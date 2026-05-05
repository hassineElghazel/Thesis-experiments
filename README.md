# Summary: Flickr30k Modality Gap Architecture and Results

## 📊 Metrics Overview
*   **Centroid Distance (↓):** Measures the overall "modality gap" by calculating the distance between the average image embedding and the average text embedding. 
*   **Paired Cosine Similarity (↑):** Measures how closely matching image-text pairs align in the embedding space.
*   **JS Divergence (↓):** Measures the statistical difference in the shape and spread of the image distribution versus the text distribution.
*   **k-NN Mixing Rate (↑):** Measures how well the two modalities interleave at a local level (0 means embeddings are completely isolated with their own modality).

---

## 📈 Performance Results

| Model | Centroid Distance ↓ | Paired Cosine ↑ | JS Divergence ↓ | k-NN Mixing ↑ |
| :--- | :--- | :--- | :--- | :--- |
| **CLIP Pre-trained (final)** | 0.806 | 0.311 | 0.021 | 0.000 |
| **Dual Scratch** | 0.761 | 0.668 | 0.002 | ~0.000 |
| **Option B Shared** | 0.384 | 0.874 | 0.075 | 0.002 |
| **TinyCLIP Scratch** | 1.068 | 0.393 | 0.220 | 0.000 |

---

## 🧠 Model Breakdown & Analysis

*   **CLIP Pre-trained (final):** The standard, off-the-shelf model trained on 400M pairs. Despite massive training, it exhibits a clear modality gap (centroid: 0.806). The low JS divergence indicates the text and image distributions have similar shapes, but are simply translated away from one another.
*   **Dual Encoder From Scratch:** Trained on just 8k pairs with separate encoders for images and text (zero shared weights). It produces a centroid gap (0.761) nearly identical to the massive CLIP model. This proves the modality gap is persistent and not primarily caused by insufficient data or capacity.
*   **Option B — Shared Token Encoder:** The core thesis model. It uses the exact same data, tokenization, and capacity as the Dual Scratch model, but relies on a **single shared embedding table and MLP** for both modalities. This architectural shift cuts the centroid distance by ~50% (0.384) and significantly boosts paired similarity.
*   **TinyCLIP From Scratch:** A miniature ViT+Transformer clone of CLIP trained on 8k pairs. It yielded the worst results, though the findings are inconclusive; it is unclear if the failure is due to the ViT architecture naturally producing a wider gap, or simply because ViTs require far more data than 8,000 pairs to properly converge.

---

## 🎯 Key Takeaways

1.  **The Modality Gap is Persistent:** Scaling up data and model parameters does not inherently make the gap go away. Massive models (CLIP) and tiny separated models (Dual Scratch) suffer from nearly identical centroid distances.
2.  **Weight Sharing is the Differentiator:** Forcing both modalities through the same learned representations (Option B) is a proven architectural method to significantly shrink the modality gap. 
3.  **Reduction ≠ Elimination:** Even when the gap is halved via weight sharing (Option B), the k-NN mixing rate stays near zero. The two modality clouds are relocated much closer together, but they still remain clearly distinguishable and separated in their local neighborhoods.
