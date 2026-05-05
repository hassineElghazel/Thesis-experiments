import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .base import AbstractVisualTokeniser


class VQVAEVisualTokeniser(AbstractVisualTokeniser):
    """
    Encodes images into sequences of VQ-VAE codebook indices.
    Token IDs are offset above the BPE vocab to avoid collisions.
    """

    def __init__(
        self,
        vqvae_model,
        codebook_k: int = 1024,
        img_size: int = 224,
        token_offset: int = 49408,
        device: str = "cpu",
    ):
        self.vqvae = vqvae_model
        self.codebook_k = codebook_k
        self.img_size = img_size
        self.token_offset = token_offset
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def fit(self, images: list[Image.Image]) -> None:
        pass  # VQ-VAE is trained separately via vqvae_trainer

    def encode(self, image: Image.Image) -> torch.Tensor:
        self.vqvae.eval()
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            indices = self.vqvae.encode_to_indices(x)  # (1, H*W)
        ids = (indices.squeeze(0) + self.token_offset).cpu()
        return ids.long()

    @property
    def vocab_size(self) -> int:
        return self.token_offset + self.codebook_k + 1
