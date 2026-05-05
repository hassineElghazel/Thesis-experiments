import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantiser(nn.Module):
    """Straight-through estimator VQ layer."""

    def __init__(self, codebook_k: int, latent_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.k = codebook_k
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_k, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1 / codebook_k, 1 / codebook_k)

    def forward(self, z: torch.Tensor):
        # z: (B, C, H, W)  →  flatten to (B*H*W, C)
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)

        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = dist.argmin(1)                     # (B*H*W,)
        z_q = self.embedding(indices).reshape(B, H, W, C).permute(0, 3, 1, 2)

        loss = F.mse_loss(z_q.detach(), z) * self.commitment_cost + F.mse_loss(z_q, z.detach())

        z_q_st = z + (z_q - z).detach()             # straight-through
        return z_q_st, loss, indices.reshape(B, H * W)

    def utilisation(self, indices: torch.Tensor) -> float:
        used = indices.unique().numel()
        return used / self.k


class VQVAE(nn.Module):
    """
    Small convolutional VQ-VAE for 224×224 RGB images.
    Encoder: 4 strided convolutions → 14×14 latent map
    Codebook: 1024 entries × latent_dim
    Decoder: 4 transposed convolutions → 224×224 reconstruction
    """

    def __init__(self, latent_dim: int = 64, codebook_k: int = 1024,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),   # 112
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 56
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 28
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1), # 14
        )
        self.vq = VectorQuantiser(codebook_k, latent_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),  # 28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),          # 56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),           # 112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),             # 224
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, recon_loss + vq_loss, vq_loss

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Returns codebook indices: (B, H*W) — used by VQVAEVisualTokeniser."""
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices
