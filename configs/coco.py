from dataclasses import dataclass
from .base import ExperimentConfig


@dataclass
class COCOConfig(ExperimentConfig):
    # overrides for COCO scale
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 128
    out_dim: int = 256
    hidden_dim: int = 512
    max_seq_len: int = 77          # CLIP BPE max length

    # VQ-VAE tokeniser
    vqvae_codebook_k: int = 1024
    vqvae_latent_dim: int = 64
    vqvae_img_tokens: int = 196    # 14×14 patches for 224px image
    vqvae_token_offset: int = 49408  # start above CLIP BPE vocab

    # text tokeniser
    clip_bpe_vocab: int = 49408

    # shared encoder total vocab = clip_bpe_vocab + vqvae_codebook_k + 1
    # dual image encoder vocab = vqvae_codebook_k + 1
    # dual text encoder vocab  = clip_bpe_vocab + 1

    # TinyCLIP architecture (clip_scratch.py)
    vit_layers: int = 6
    vit_dim: int = 256
    vit_heads: int = 4
    txt_layers: int = 6
    txt_dim: int = 256
    txt_heads: int = 4

    # VQ-VAE training
    vqvae_epochs: int = 5
    vqvae_lr: float = 1e-3
    vqvae_commitment_cost: float = 0.25

    # Flickr30k data (streamed from HuggingFace Hub)
    flickr_split: str = "test"
    flickr_n_images: int = 4000
    flickr_captions_per_image: int = 2

    output_dir: str = "outputs/coco"


def get_coco_config() -> COCOConfig:
    return COCOConfig()
