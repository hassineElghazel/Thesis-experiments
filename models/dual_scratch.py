import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import AbstractEncoder
from tokenisers.visual_kmeans import KMeansVisualTokeniser
from tokenisers.text_char import CharTextTokeniser


class _TokenEncoder(nn.Module):
    """Single-modality token encoder — same architecture as SharedTokenEncoder's inner MLP."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, out_dim: int,
                 n_layers: int, max_seq: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_seq, emb_dim) * 0.02)
        layers = [nn.Linear(emb_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != 0).float().unsqueeze(-1)
        emb = self.embed(x) + self.pos[:, : x.size(1), :]
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.mlp(pooled), dim=-1)


class DualEncoderFromScratch(AbstractEncoder):
    """
    Controlled dual-encoder baseline — isolates the architectural variable vs SharedTokenEncoder.

    Key constraints (capacity-matched to SharedTokenEncoder):
      - Identical _TokenEncoder architecture for both modalities
      - Same emb_dim, hidden_dim, out_dim, n_layers, max_seq_len
      - Separate embedding tables and weights (NO weight sharing)
      - Same InfoNCE loss, AdamW, cosine LR schedule, identical hyperparameters

    Image tokens: [token_offset, token_offset + codebook_k)
    Text tokens:  [1, char_vocab)
    """

    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        # Use dual_hidden_dim (≈167) so total dual params ≈ shared encoder params.
        # The shared encoder's embedding spans the full combined vocab (193 entries),
        # while the dual has two separate smaller vocabs (65 and 128). Scaling hidden dim
        # from 256 → 167 compensates: dual/shared param ratio becomes ~1.003.
        h = cfg.dual_hidden_dim

        self.img_encoder = _TokenEncoder(
            vocab_size=cfg.codebook_k + 1,
            emb_dim=cfg.emb_dim,
            hidden=h,
            out_dim=cfg.out_dim,
            n_layers=cfg.n_layers,
            max_seq=cfg.max_seq_len,
        ).to(self.device)

        self.txt_encoder = _TokenEncoder(
            vocab_size=cfg.char_vocab,
            emb_dim=cfg.emb_dim,
            hidden=h,
            out_dim=cfg.out_dim,
            n_layers=cfg.n_layers,
            max_seq=cfg.max_seq_len,
        ).to(self.device)

        self.img_tok = KMeansVisualTokeniser(
            codebook_k=cfg.codebook_k,
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
            max_seq_len=cfg.max_seq_len,
            token_offset=cfg.token_offset,
        )
        self.txt_tok = CharTextTokeniser(
            char_vocab=cfg.char_vocab,
            max_seq_len=cfg.max_seq_len,
        )
        self._fitted = False

    @property
    def name(self) -> str:
        return "Dual Encoder (From Scratch)"

    def fit_tokeniser(self, images: list[Image.Image]) -> None:
        # Re-use the same codebook fitted by SharedTokenEncoder.
        # Pass in a pre-fitted KMeansVisualTokeniser to share it.
        self._fitted = True

    def set_tokeniser(self, img_tok: KMeansVisualTokeniser) -> None:
        """Inject a pre-fitted visual tokeniser (same codebook as SharedTokenEncoder)."""
        self.img_tok = img_tok
        self._fitted = True

    def _remap_img_tokens(self, toks: torch.Tensor) -> torch.Tensor:
        """Shift image tokens from [token_offset, …) → [1, codebook_k+1) for the smaller vocab."""
        out = toks.clone()
        mask = toks != 0
        out[mask] = toks[mask] - self.cfg.token_offset + 1
        return out

    def encode_img_tokens(self, toks: torch.Tensor) -> torch.Tensor:
        """Forward pass on raw image tokens (offset space). Returns L2-norm embeddings."""
        return self.img_encoder(self._remap_img_tokens(toks))

    def encode_txt_tokens(self, toks: torch.Tensor) -> torch.Tensor:
        return self.txt_encoder(toks)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        assert self._fitted
        self.img_encoder.eval()
        toks = torch.stack([self.img_tok.encode(img) for img in images]).to(self.device)
        with torch.no_grad():
            out = self.img_encoder(self._remap_img_tokens(toks))
        return out.cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.txt_encoder.eval()
        toks = torch.stack([self.txt_tok.encode(t) for t in texts]).to(self.device)
        with torch.no_grad():
            out = self.txt_encoder(toks)
        return out.cpu().numpy()

    def capacity_report(self, shared_encoder) -> None:
        """Assert parameter counts are within 5% of the shared encoder."""
        dual_params = (
            sum(p.numel() for p in self.img_encoder.parameters())
            + sum(p.numel() for p in self.txt_encoder.parameters())
        )
        shared_params = sum(p.numel() for p in shared_encoder.encoder.parameters())
        ratio = dual_params / shared_params
        status = "OK" if 0.95 <= ratio <= 1.05 else "WARNING — capacity mismatch!"
        print(f"  Capacity audit: dual={dual_params:,}  shared={shared_params:,}  "
              f"ratio={ratio:.3f}  {status}")
