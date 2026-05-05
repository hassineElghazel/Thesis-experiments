import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import AbstractEncoder
from tokenisers.visual_kmeans import KMeansVisualTokeniser
from tokenisers.text_char import CharTextTokeniser


class _TokenMLP(nn.Module):
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


class SharedTokenEncoder(AbstractEncoder):
    """
    Single encoder for both modalities — the core of Option B.
    Image tokens are offset into [token_offset, token_offset + codebook_k).
    Text tokens are ASCII chars in [1, char_vocab).
    Both go through the SAME embedding table and MLP.
    """

    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        total_vocab = cfg.char_vocab + cfg.codebook_k + 1
        self.encoder = _TokenMLP(
            vocab_size=total_vocab,
            emb_dim=cfg.emb_dim,
            hidden=cfg.hidden_dim,
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
        return "Option B Shared Encoder"

    def fit_tokeniser(self, images: list[Image.Image]) -> None:
        self.img_tok.fit(images)
        self._fitted = True

    def _tokenise_images(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self.img_tok.encode(img) for img in images]).to(self.device)

    def _tokenise_texts(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self.txt_tok.encode(t) for t in texts]).to(self.device)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        assert self._fitted, "Call fit_tokeniser() first"
        self.encoder.eval()
        with torch.no_grad():
            out = self.encoder(self._tokenise_images(images))
        return out.cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.encoder.eval()
        with torch.no_grad():
            out = self.encoder(self._tokenise_texts(texts))
        return out.cpu().numpy()

    def token_audit(self) -> None:
        """Log token range sanity check (img tokens must not overlap txt tokens)."""
        img_range = (self.cfg.token_offset, self.cfg.token_offset + self.cfg.codebook_k)
        txt_range = (1, self.cfg.char_vocab)
        overlap = set(range(*img_range)) & set(range(*txt_range))
        assert len(overlap) == 0, f"Token overlap detected: {overlap}"
        print(f"  Token audit OK — img [{img_range[0]}, {img_range[1]}), "
              f"txt [{txt_range[0]}, {txt_range[1]}), overlap=0")
