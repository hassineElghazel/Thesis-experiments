"""
Phase 2 experiment: 5-config modality gap comparison on Flickr30k.

Configs:
  1. CLIP Pre-trained (final)
  2. CLIP Pre-trained (token-level)
  3. Dual Encoder From Scratch (VQ-VAE tokens + CLIP BPE)
  4. Option B Shared Encoder    (VQ-VAE tokens + CLIP BPE, shared weights)
  5. TinyCLIP From Scratch      (ViT + TextTransformer trained on Flickr30k only)

Prerequisite: run `python run.py train-vqvae` first to produce outputs/flickr/checkpoints/vqvae.pt

Run: python run.py flickr
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from configs.flickr import get_flickr_config
from data.flickr import Flickr30kDataset
from models.vqvae import VQVAE
from models.clip_pretrained import CLIPPretrainedWrapper
from models.shared_encoder import SharedTokenEncoder
from models.dual_scratch import DualEncoderFromScratch
from models.clip_scratch import TinyCLIPFromScratch
from tokenisers.visual_vqvae import VQVAEVisualTokeniser
from tokenisers.text_clip_bpe import CLIPBPETokeniser
from tokenisers.text_char import CharTextTokeniser
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from visualisation.plotter import Plotter
from utils.seed import set_seed
from utils.device import get_device
from utils.logging import save_metrics, print_metrics_table


def _batch_infer(model, toks: torch.Tensor, batch_size: int) -> np.ndarray:
    """Run model on token tensor in batches; returns CPU numpy array."""
    parts = []
    with torch.no_grad():
        for i in range(0, len(toks), batch_size):
            parts.append(model(toks[i : i + batch_size]).cpu())
    return torch.cat(parts).numpy()


def _build_shared_enc(cfg, images, texts, img_tok, txt_tok, device):
    """Train Option B Shared Encoder with VQ-VAE image tokens + CLIP BPE text tokens."""
    total_vocab = cfg.clip_bpe_vocab + cfg.vqvae_codebook_k + 1

    import torch.nn as nn
    import torch.nn.functional as F

    class _TokenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(total_vocab, cfg.emb_dim, padding_idx=0)
            # must cover both image tokens (196) and text tokens (77)
            _max_len = max(cfg.max_seq_len, cfg.vqvae_img_tokens)
            self.pos = nn.Parameter(torch.randn(1, _max_len, cfg.emb_dim) * 0.02)
            layers = [nn.Linear(cfg.emb_dim, cfg.hidden_dim), nn.GELU()]
            for _ in range(cfg.n_layers - 1):
                layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                           nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            layers += [nn.Linear(cfg.hidden_dim, cfg.out_dim)]
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            mask = (x != 0).float().unsqueeze(-1)
            emb = self.embed(x) + self.pos[:, : x.size(1), :]
            pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
            return F.normalize(self.mlp(pooled), dim=-1)

    enc = _TokenMLP().to(device)

    print("  Tokenising Flickr30k images and texts...")
    img_toks = torch.stack([img_tok.encode(img) for img in images]).to(device)
    txt_toks = torch.stack([txt_tok.encode(t) for t in texts]).to(device)

    from training.losses import infonce
    import torch.optim as optim

    opt = optim.AdamW(enc.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    N = img_toks.size(0)
    enc.train()
    for ep in range(cfg.epochs):
        perm = torch.randperm(N)
        ep_loss, nb = 0.0, 0
        for i in range(0, N, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            ie = enc(img_toks[idx])
            te = enc(txt_toks[idx])
            loss = infonce(ie, te, cfg.temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f"  [Flickr SharedEnc] Epoch {ep+1}/{cfg.epochs} loss={ep_loss/nb:.4f}")
    enc.eval()
    ob_img = _batch_infer(enc, img_toks, cfg.batch_size)
    ob_txt = _batch_infer(enc, txt_toks, cfg.batch_size)
    return ob_img, ob_txt, enc, img_toks, txt_toks


def run_flickr():
    cfg = get_flickr_config()
    set_seed(cfg.seed)
    device = str(get_device())
    print(f"Device: {device}")

    # ── Data ──
    print("\nLoading Flickr30k...")
    dataset = Flickr30kDataset(
        split=cfg.flickr_split,
        n_images=cfg.flickr_n_images,
        captions_per_image=cfg.flickr_captions_per_image,
        img_size=cfg.img_size,
        seed=cfg.seed,
    )
    images = [dataset[i][0] for i in range(len(dataset))]
    texts  = [dataset[i][1] for i in range(len(dataset))]
    print(f"  {len(images)} image-caption pairs")

    # ── Load VQ-VAE ──
    ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "vqvae.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"VQ-VAE checkpoint not found at {ckpt_path}.\n"
            "Run: python run.py train-vqvae"
        )
    vqvae = VQVAE(latent_dim=cfg.vqvae_latent_dim, codebook_k=cfg.vqvae_codebook_k)
    vqvae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    vqvae.to(device)
    vqvae.eval()
    print(f"  VQ-VAE loaded from {ckpt_path}")

    img_tok = VQVAEVisualTokeniser(
        vqvae_model=vqvae,
        codebook_k=cfg.vqvae_codebook_k,
        img_size=cfg.img_size,
        token_offset=cfg.vqvae_token_offset,
        device=device,
    )
    txt_tok = CLIPBPETokeniser(max_seq_len=cfg.max_seq_len)

    # ── Config 1 & 2: CLIP Pre-trained ──
    print("\n" + "="*65)
    print("  Configs 1 & 2: CLIP Pre-trained")
    print("="*65)
    clip_final = CLIPPretrainedWrapper(mode="final", device=device)
    clip_token_enc = CLIPPretrainedWrapper(mode="token", device=device)
    clip_img_final, clip_txt_final = clip_final.encode_paired(images, texts)
    clip_img_token, clip_txt_token = clip_token_enc.encode_paired(images, texts)
    del clip_final.model, clip_token_enc.model

    # ── Configs 3 & 4: shared encoder + dual encoder ──
    print("\n" + "="*65)
    print("  Configs 3 & 4: Shared Encoder + Dual Encoder (From Scratch)")
    print("="*65)
    ob_img, ob_txt, shared_module, img_toks, txt_toks = _build_shared_enc(
        cfg, images, texts, img_tok, txt_tok, device
    )

    import torch.nn as nn
    import torch.nn.functional as F
    from training.losses import infonce
    import torch.optim as optim

    img_vocab = cfg.vqvae_codebook_k + 1
    txt_vocab = cfg.clip_bpe_vocab + 1

    class _SingleMLP(nn.Module):
        def __init__(self, vocab, max_len):
            super().__init__()
            self.embed = nn.Embedding(vocab, cfg.emb_dim, padding_idx=0)
            self.pos = nn.Parameter(torch.randn(1, max_len, cfg.emb_dim) * 0.02)
            layers = [nn.Linear(cfg.emb_dim, cfg.hidden_dim), nn.GELU()]
            for _ in range(cfg.n_layers - 1):
                layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                           nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
            layers += [nn.Linear(cfg.hidden_dim, cfg.out_dim)]
            self.mlp = nn.Sequential(*layers)
        def forward(self, x):
            mask = (x != 0).float().unsqueeze(-1)
            emb = self.embed(x) + self.pos[:, : x.size(1), :]
            pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
            return F.normalize(self.mlp(pooled), dim=-1)

    img_enc = _SingleMLP(img_vocab, max_len=cfg.vqvae_img_tokens).to(device)
    txt_enc = _SingleMLP(txt_vocab, max_len=cfg.max_seq_len).to(device)

    # Remap VQ-VAE tokens from [vqvae_offset, …) → [1, codebook_k+1)
    offset = cfg.vqvae_token_offset
    img_toks_remap = img_toks.clone()
    mask_nonpad = img_toks != 0
    img_toks_remap[mask_nonpad] = img_toks[mask_nonpad] - offset + 1

    txt_toks_local = txt_toks.clone()

    opt = optim.AdamW(list(img_enc.parameters()) + list(txt_enc.parameters()),
                      lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    N = img_toks_remap.size(0)
    img_enc.train(); txt_enc.train()
    for ep in range(cfg.epochs):
        perm = torch.randperm(N)
        ep_loss, nb = 0.0, 0
        for i in range(0, N, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            ie = img_enc(img_toks_remap[idx])
            te = txt_enc(txt_toks_local[idx])
            loss = infonce(ie, te, cfg.temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(img_enc.parameters()) + list(txt_enc.parameters()), 1.0
            )
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f"  [Flickr DualScratch] Epoch {ep+1}/{cfg.epochs} loss={ep_loss/nb:.4f}")
    img_enc.eval(); txt_enc.eval()
    de_img = _batch_infer(img_enc, img_toks_remap, cfg.batch_size)
    de_txt = _batch_infer(txt_enc, txt_toks_local, cfg.batch_size)

    # ── Config 5: TinyCLIP From Scratch ──
    print("\n" + "="*65)
    print("  Config 5: TinyCLIP From Scratch")
    print("="*65)
    bpe_tok = CLIPBPETokeniser(max_seq_len=cfg.max_seq_len)
    tiny_clip = TinyCLIPFromScratch(cfg, tokeniser=bpe_tok, device=device)

    print("  Pre-computing image tensors and text tokens...")
    tc_img_tensors = torch.stack([tiny_clip.transform(img) for img in images])
    tc_tok_ids = torch.stack([bpe_tok.encode(t) for t in texts])

    trainer = Trainer(cfg, device=device)
    trainer.fit_tinyclip(tiny_clip, tc_img_tensors, tc_tok_ids)

    # Reuse pre-computed tensors for inference (skip 8k extra PIL transforms,
    # and stream batches to MPS so we never allocate the full 4.5 GB tensor).
    tiny_clip.image_encoder.eval()
    tiny_clip.text_encoder.eval()
    img_parts, txt_parts = [], []
    with torch.no_grad():
        for i in range(0, tc_img_tensors.size(0), cfg.batch_size):
            ib = tc_img_tensors[i : i + cfg.batch_size].to(device, non_blocking=True)
            img_parts.append(tiny_clip.image_encoder(ib).cpu())
        tc_tok_ids_dev = tc_tok_ids.to(device)
        for i in range(0, tc_tok_ids_dev.size(0), cfg.batch_size):
            tb = tc_tok_ids_dev[i : i + cfg.batch_size]
            txt_parts.append(tiny_clip.text_encoder(tb).cpu())
    tc_img = torch.cat(img_parts).numpy()
    tc_txt = torch.cat(txt_parts).numpy()

    # ── Evaluate ──
    print("\n" + "="*65)
    print("  Results")
    print("="*65)
    evaluator = Evaluator(knn_k=cfg.knn_k)
    configs_eval = [
        {"name": "CLIP Pre-trained (final)", "img_emb": clip_img_final, "txt_emb": clip_txt_final},
        {"name": "CLIP Pre-trained (token)", "img_emb": clip_img_token, "txt_emb": clip_txt_token},
        {"name": "Dual Encoder (scratch)",   "img_emb": de_img,         "txt_emb": de_txt},
        {"name": "Option B Shared Enc",      "img_emb": ob_img,         "txt_emb": ob_txt},
        {"name": "TinyCLIP (scratch Flickr)","img_emb": tc_img,         "txt_emb": tc_txt},
    ]
    results = evaluator.run(configs_eval)
    for cfg_e in configs_eval:
        cfg_e["metrics"] = results[cfg_e["name"]]

    print_metrics_table(
        {cfg_e["name"]: {k: v for k, v in results[cfg_e["name"]].items() if not k.startswith("_")}
         for cfg_e in configs_eval}
    )

    save_metrics(
        {name: {k: v for k, v in m.items() if not k.startswith("_")}
         for name, m in results.items()},
        out_dir=cfg.output_dir,
    )

    print("\nGenerating figures...")
    plotter = Plotter(out_dir=cfg.output_dir)
    plotter.figure2_anisotropy(configs_eval)

    print(f"\nDone. Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    run_flickr()
