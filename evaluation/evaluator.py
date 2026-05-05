import numpy as np
from PIL import Image
from metrics.gap import GapMetrics


class Evaluator:
    """
    Runs GapMetrics across multiple encoder configurations.
    Each config is a dict:
        {
          "name": str,
          "img_emb": np.ndarray,   # (N, D) L2-normalised
          "txt_emb": np.ndarray,   # (N, D) L2-normalised
        }
    """

    def __init__(self, knn_k: int = 15):
        self.metrics = GapMetrics(knn_k=knn_k)

    def run(self, configs: list[dict]) -> dict[str, dict]:
        """
        Returns {config_name: metrics_dict} for all configs.
        """
        results = {}
        for cfg in configs:
            name = cfg["name"]
            print(f"\nEvaluating: {name}")
            img_emb = cfg["img_emb"]
            txt_emb = cfg["txt_emb"]
            m = self.metrics.compute_all(img_emb, txt_emb)
            results[name] = m
            print(f"  centroid={m['centroid_distance']:.4f}  "
                  f"jsd={m['js_divergence']:.4f}  "
                  f"mix={m['knn_mixing_rate']:.4f}  "
                  f"pairedcos={m['paired_cosine_sim']:.4f}")
        return results

    def ratios_vs_baseline(
        self, results: dict[str, dict], baseline_name: str
    ) -> dict[str, dict]:
        """Compute ratio of each metric relative to baseline config."""
        base = results[baseline_name]
        ratios = {}
        for name, m in results.items():
            if name == baseline_name:
                continue
            ratios[name] = {}
            for k, v in m.items():
                if k.startswith("_"):
                    continue
                bv = base.get(k, 1.0)
                ratios[name][k] = v / (bv + 1e-12)
        return ratios
