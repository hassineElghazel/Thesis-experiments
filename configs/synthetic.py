from .base import ExperimentConfig


def get_synthetic_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=42,
        batch_size=64,
        epochs=300,
        lr=2e-3,
        weight_decay=0.01,
        temperature=0.07,
        emb_dim=64,
        hidden_dim=256,
        out_dim=128,
        n_layers=3,
        codebook_k=64,
        patch_size=16,
        img_size=224,
        max_seq_len=64,
        char_vocab=128,
        token_offset=128,
        knn_k=15,
        output_dir="outputs/synthetic",
    )
