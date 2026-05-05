"""
Phase 2 prerequisite: train VQ-VAE on Flickr30k (train split from HuggingFace Hub).

Run: python run.py train-vqvae
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

from configs.flickr import get_flickr_config
from models.vqvae import VQVAE
from training.vqvae_trainer import VQVAETrainer
from utils.seed import set_seed
from utils.device import get_device


def load_flickr_train_images(n_images: int = 10000, img_size: int = 224):
    from datasets import load_dataset

    print("  Loading Flickr30k (test split) from HuggingFace Hub...")
    ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
    images = []
    for ex in ds:
        try:
            img = ex["image"].convert("RGB").resize((img_size, img_size))
            images.append(img)
        except Exception:
            pass
        if len(images) >= n_images:
            break
    print(f"  Loaded {len(images)} training images")
    return images


def train_vqvae():
    cfg = get_flickr_config()
    set_seed(cfg.seed)
    device = str(get_device())
    print(f"Device: {device}")

    images = load_flickr_train_images(n_images=10000, img_size=cfg.img_size)

    model = VQVAE(
        latent_dim=cfg.vqvae_latent_dim,
        codebook_k=cfg.vqvae_codebook_k,
        commitment_cost=cfg.vqvae_commitment_cost,
    )

    vtrainer = VQVAETrainer(cfg, device=device)
    vtrainer.fit(model, images, batch_size=32)

    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "vqvae.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nVQ-VAE saved to {ckpt_path}")


if __name__ == "__main__":
    train_vqvae()
