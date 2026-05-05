from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 300
    lr: float = 2e-3
    weight_decay: float = 0.01
    temperature: float = 0.07

    # shared-encoder / dual-encoder architecture
    emb_dim: int = 64
    hidden_dim: int = 256
    # dual_hidden_dim is scaled down from hidden_dim so dual encoder total params ≈ shared encoder
    # With emb_dim=64, out_dim=128, n_layers=3: h=167 gives dual/shared ratio = 1.003
    dual_hidden_dim: int = 167
    out_dim: int = 128
    n_layers: int = 3

    # visual tokeniser
    codebook_k: int = 64          # k-means codebook size (synthetic)
    patch_size: int = 16          # image patch size in pixels
    img_size: int = 224
    max_seq_len: int = 64

    # text tokeniser
    char_vocab: int = 128         # ASCII char vocabulary
    token_offset: int = 128       # image token IDs start here

    # kNN mixing rate k
    knn_k: int = 15

    # output directory
    output_dir: str = "outputs/synthetic"
