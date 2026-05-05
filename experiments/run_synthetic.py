"""
Phase 1 experiment: 4-config modality gap comparison on synthetic data.

Configs:
  1. CLIP Pre-trained (final embeddings)
  2. CLIP Pre-trained (token-level embeddings)
  3. Dual Encoder From Scratch  ← NEW: fixes the domain-mismatch confound
  4. Option B Shared Encoder

Run: python run.py synthetic
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from configs.synthetic import get_synthetic_config
from data.synthetic import SyntheticDataset, CONCEPTS
from models.clip_pretrained import CLIPPretrainedWrapper
from models.shared_encoder import SharedTokenEncoder
from models.dual_scratch import DualEncoderFromScratch
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from visualisation.plotter import Plotter
from utils.seed import set_seed
from utils.device import get_device
from utils.logging import save_metrics, print_metrics_table


def run_synthetic():
    cfg = get_synthetic_config()
    set_seed(cfg.seed)
    device = str(get_device())
    print(f"Device: {device}")

    # ── Data ──
    print("\nGenerating synthetic data...")
    dataset = SyntheticDataset(n_concepts=12, n_variations=30,
                               img_size=cfg.img_size, seed=cfg.seed)
    images = dataset.images
    texts = dataset.texts
    concept_ids = dataset.concept_ids
    N = len(images)
    print(f"  {N} pairs, {len(CONCEPTS)} concepts")

    # ── Config 1 & 2: CLIP Pre-trained ──
    print("\n" + "="*65)
    print("  Config 1 & 2: CLIP Pre-trained")
    print("="*65)
    clip_final = CLIPPretrainedWrapper(mode="final", device=device)
    clip_token = CLIPPretrainedWrapper(mode="token", device=device)

    clip_img_final, clip_txt_final = clip_final.encode_paired(images, texts)
    clip_img_token, clip_txt_token = clip_token.encode_paired(images, texts)

    # Free CLIP models from memory
    del clip_final.model, clip_token.model
    torch.cuda.empty_cache() if device == "cuda" else None

    # ── Config 3 & 4: train shared encoder first (to share the visual codebook) ──
    print("\n" + "="*65)
    print("  Config 4: Option B Shared Encoder")
    print("="*65)
    shared_enc = SharedTokenEncoder(cfg, device=device)
    shared_enc.img_tok.fit(images)
    shared_enc.token_audit()
    shared_enc._fitted = True

    img_toks = torch.stack([shared_enc.img_tok.encode(img) for img in images])
    txt_toks = torch.stack([shared_enc.txt_tok.encode(t) for t in texts])

    trainer = Trainer(cfg, device=device)
    trainer.fit_shared(shared_enc, img_toks, txt_toks)

    ob_img = shared_enc.encode_images(images)
    ob_txt = shared_enc.encode_texts(texts)

    # ── Config 3: Dual Encoder From Scratch ──
    print("\n" + "="*65)
    print("  Config 3: Dual Encoder From Scratch")
    print("="*65)
    dual_enc = DualEncoderFromScratch(cfg, device=device)
    dual_enc.set_tokeniser(shared_enc.img_tok)  # same codebook — only weight sharing differs
    dual_enc.capacity_report(shared_enc)

    trainer.fit_dual(dual_enc, img_toks, txt_toks)

    de_img = dual_enc.encode_images(images)
    de_txt = dual_enc.encode_texts(texts)

    # ── Evaluate ──
    print("\n" + "="*65)
    print("  Results")
    print("="*65)
    evaluator = Evaluator(knn_k=cfg.knn_k)
    configs_eval = [
        {"name": "CLIP Pre-trained (final)", "img_emb": clip_img_final, "txt_emb": clip_txt_final},
        {"name": "CLIP Pre-trained (token)", "img_emb": clip_img_token, "txt_emb": clip_txt_token},
        {"name": "Dual Encoder (scratch)", "img_emb": de_img, "txt_emb": de_txt},
        {"name": "Option B Shared Enc",    "img_emb": ob_img, "txt_emb": ob_txt},
    ]
    results = evaluator.run(configs_eval)
    for cfg_e in configs_eval:
        cfg_e["metrics"] = results[cfg_e["name"]]

    print_metrics_table(
        {cfg_e["name"]: {k: v for k, v in results[cfg_e["name"]].items() if not k.startswith("_")}
         for cfg_e in configs_eval}
    )

    ratios = evaluator.ratios_vs_baseline(results, "CLIP Pre-trained (final)")
    print("\nRatios vs CLIP Pre-trained (final):")
    for name, r in ratios.items():
        print(f"  {name}: centroid={r.get('centroid_distance',0):.3f}  "
              f"jsd={r.get('js_divergence',0):.3f}  "
              f"mix={r.get('knn_mixing_rate',0):.3f}  "
              f"pairedcos={r.get('paired_cosine_sim',0):.3f}")

    save_metrics(
        {name: {k: v for k, v in m.items() if not k.startswith("_")}
         for name, m in results.items()},
        out_dir=cfg.output_dir,
    )

    # ── Figures ──
    print("\nGenerating figures...")
    concept_names = [
        f"{c['text'].split()[1]}\n{c['text'].split()[2]}" for c in CONCEPTS
    ]
    plotter = Plotter(out_dir=cfg.output_dir)
    plotter.figure1_main_comparison(configs_eval, concept_ids, n_concepts=len(CONCEPTS))
    plotter.figure2_anisotropy(configs_eval)
    plotter.figure3_per_concept(configs_eval, concept_ids, concept_names)

    print(f"\nDone. Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    run_synthetic()
