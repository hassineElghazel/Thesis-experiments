# Summary of Multi-Modal Architectures and Modality Gap Analysis

This document summarizes four architectural configurations tested on the Flickr30k dataset to analyze the "modality gap" between image and text embeddings. 

## 1. Model Architectures

### CLIP Pre-trained
*   **Architecture:** Dual encoder using Vision Transformer (ViT-B/32) for images and a 12-layer Transformer for text. Uses the final projection-layer outputs.
*   **Training:** Contrastive learning on ~400M image-text pairs (WIT dataset). ~150M parameters.

### Dual Encoder From Scratch (Baseline)
*   **Architecture:** Two completely independent 3-layer MLP sub-encoders. Images are tokenized via KMeans (65 vocab), and text via ASCII characters (128 vocab). 
*   **Capacity:** Hidden dimension of 167 to match the parameter count of Option B.
*   **Training:** Trained from scratch on 8,000 Flickr30k pairs using InfoNCE loss. Shares *zero* parameters between modalities.

### Option B — Shared Token Encoder
*   **Architecture:** A single, shared 3-layer MLP encoder (hidden dimension 256) processes both image and text tokens using a unified vocabulary of 193 tokens.
*   **Training:** Identical hyper-parameters and data to the Dual Encoder. Forces both modalities into a unified representation space.

### TinyCLIP From Scratch
*   **Architecture:** A miniature clone of CLIP (ViT + Transformer dual encoders).
*   **Training:** Trained from scratch strictly on the 8k Flickr30k pairs to isolate architecture from pre-training scale.

---

## 2. Experimental Results (Flickr30k)

| Model | Centroid Distance ↓ | Paired Cosine ↑ | JS Divergence ↓ | k-NN Mixing ↑ |
| :--- | :--- | :--- | :--- | :--- |
| **CLIP Pre-trained (final)** | 0.806 | 0.311 | 0.021 | 0.000 |
| **Dual Scratch** | 0.761 | 0.668 | 0.002 | ~0.000 |
| **Option B Shared** | 0.384 | 0.874 | 0.075 | 0.002 |
| **TinyCLIP Scratch** | 1.068 | 0.393 | 0.220 | 0.000 |

### Key Metrics Explained
*   **Centroid Distance:** Distance between the average image embedding and average text embedding (measures the modality gap).
*   **Paired Cosine Similarity:** How close matching image-text pairs are.
*   **JS Divergence:** Differences in the overall statistical distribution of the two modalities.
*   **k-NN Mixing Rate:** The fraction of a point's nearest neighbors that belong to the *other* modality.

---

## 3. Core Conclusions

1.  **The Modality Gap is Persistent:** Standard CLIP (trained on 400M pairs) and the Dual Scratch model (trained on 8k pairs) produce nearly identical centroid distances (~0.80). Scaling up data and model size does not naturally eliminate the gap.
2.  **Weight Sharing is the Key Variable:** Option B (Shared Encoder) reduced the centroid distance by roughly 50% compared to the Dual Scratch baseline while drastically improving paired cosine similarity. This proves that shared architecture, rather than data capacity, fundamentally alters the geometry of the gap.
3.  **Reduction is Not Elimination:** Despite the centroids moving much closer in Option B, the k-NN mixing rate remains near zero across all models. This indicates that while the overall modality clouds are drawn closer together, they remain locally separated and distinguishable in the embedding space.
4.  **TinyCLIP Underperformed:** The TinyCLIP model had the worst centroid gap and paired cosine. However, because ViTs are highly data-hungry, it is difficult to determine whether this failure is due to the dual-transformer architecture itself or simply undertraining on a small (8k) dataset.
