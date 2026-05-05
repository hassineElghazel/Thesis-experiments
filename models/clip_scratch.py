import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import AbstractEncoder


class _PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_ch: int = 3, dim: int = 256):
        super().__init__()
        n = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)           # (B, n_patches, dim)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class _ViTEncoder(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, dim: int = 256,
                 n_heads: int = 4, n_layers: int = 6, out_dim: int = 128):
        super().__init__()
        self.patch_embed = _PatchEmbed(img_size, patch_size, dim=dim)
        self.blocks = nn.Sequential(*[_TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])     # CLS token
        return F.normalize(self.proj(x), dim=-1)


class _TextTransformer(nn.Module):
    def __init__(self, vocab_size: int = 49408, max_len: int = 77, dim: int = 256,
                 n_heads: int = 4, n_layers: int = 6, out_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[_TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x) + self.pos[:, : x.size(1), :]
        x = self.blocks(x)
        x = self.norm(x[:, 0])     # first token (EOS position in CLIP convention)
        return F.normalize(self.proj(x), dim=-1)


class TinyCLIPFromScratch(AbstractEncoder):
    """
    Small ViT + TextTransformer trained from scratch on COCO with InfoNCE.
    Isolates pre-training scale from architecture.
    Architecture matches CLIP's dual-encoder paradigm but is trained on COCO only.
    """

    def __init__(self, cfg, tokeniser, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.tokeniser = tokeniser  # CLIPBPETokeniser

        self.image_encoder = _ViTEncoder(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            dim=cfg.vit_dim,
            n_heads=cfg.vit_heads,
            n_layers=cfg.vit_layers,
            out_dim=cfg.out_dim,
        ).to(self.device)

        self.text_encoder = _TextTransformer(
            vocab_size=cfg.clip_bpe_vocab,
            max_len=cfg.max_seq_len,
            dim=cfg.txt_dim,
            n_heads=cfg.txt_heads,
            n_layers=cfg.txt_layers,
            out_dim=cfg.out_dim,
        ).to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @property
    def name(self) -> str:
        return "TinyCLIP (From Scratch on COCO)"

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_encoder.eval()
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), self.cfg.batch_size):
                batch = torch.stack([
                    self.transform(img) for img in images[i:i + self.cfg.batch_size]
                ]).to(self.device)
                all_embs.append(self.image_encoder(batch).cpu())
        return torch.cat(all_embs, dim=0).numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.text_encoder.eval()
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.cfg.batch_size):
                toks = torch.stack([
                    self.tokeniser.encode(t) for t in texts[i:i + self.cfg.batch_size]
                ]).to(self.device)
                all_embs.append(self.text_encoder(toks).cpu())
        return torch.cat(all_embs, dim=0).numpy()
