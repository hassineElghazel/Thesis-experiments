import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity as sk_cos


class GapMetrics:
    """Computes all six modality-gap metrics from Yu et al. (2026)."""

    def __init__(self, knn_k: int = 15, jsd_n_sample: int = 200, jsd_n_bins: int = 60):
        self.knn_k = knn_k
        self.jsd_n_sample = jsd_n_sample
        self.jsd_n_bins = jsd_n_bins

    def centroid_distance(self, e1: np.ndarray, e2: np.ndarray) -> float:
        return float(np.linalg.norm(e1.mean(0) - e2.mean(0)))

    def covariance_stats(self, emb: np.ndarray) -> tuple[float, float, np.ndarray]:
        centered = emb - emb.mean(0)
        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(eigvals, 1e-12)
        kappa = float(eigvals[0] / eigvals[-1])
        log_r = np.log(np.arange(1, len(eigvals) + 1))
        log_e = np.log(eigvals)
        alpha = float(-np.polyfit(log_r, log_e, 1)[0])
        return kappa, alpha, eigvals

    def js_divergence_cosine(
        self, e1: np.ndarray, e2: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        n = min(self.jsd_n_sample, len(e1), len(e2))
        idx1 = np.random.choice(len(e1), n, replace=False)
        idx2 = np.random.choice(len(e2), n, replace=False)
        c1 = sk_cos(e1[idx1])
        c2 = sk_cos(e2[idx2])
        triu = np.triu_indices(n, k=1)
        s1, s2 = c1[triu], c2[triu]
        bins = np.linspace(-1, 1, self.jsd_n_bins + 1)
        h1, _ = np.histogram(s1, bins=bins)
        h2, _ = np.histogram(s2, bins=bins)
        h1 = h1 / (h1.sum() + 1e-12)
        h2 = h2 / (h2.sum() + 1e-12)
        m = (h1 + h2) / 2

        def kl(p, q):
            safe_p = np.where(p > 0, p, 1.0)  # avoid log(0); np.where evaluates both branches
            return np.sum(np.where(p > 0, safe_p * np.log(safe_p / (q + 1e-12)), 0.0))

        jsd = float(0.5 * kl(h1, m) + 0.5 * kl(h2, m))
        return jsd, s1, s2

    def knn_mixing_rate(self, e1: np.ndarray, e2: np.ndarray) -> float:
        combined = np.vstack([e1, e2])
        labels = np.array([0] * len(e1) + [1] * len(e2))
        nbrs = NearestNeighbors(n_neighbors=self.knn_k + 1, metric="cosine").fit(combined)
        _, idx = nbrs.kneighbors(combined)
        return float(
            np.mean([np.mean(labels[idx[i, 1:]] != labels[i]) for i in range(len(combined))])
        )

    def paired_cosine_sim(self, e1: np.ndarray, e2: np.ndarray) -> float:
        return float(np.mean(np.sum(e1 * e2, axis=1)))

    def compute_all(
        self, img_emb: np.ndarray, txt_emb: np.ndarray
    ) -> dict:
        centroid = self.centroid_distance(img_emb, txt_emb)
        ki, ai, evi = self.covariance_stats(img_emb)
        kt, at, evt = self.covariance_stats(txt_emb)
        jsd, cs_img, cs_txt = self.js_divergence_cosine(img_emb, txt_emb)
        mixing = self.knn_mixing_rate(img_emb, txt_emb)
        paired = self.paired_cosine_sim(img_emb, txt_emb)
        return {
            "centroid_distance": centroid,
            "kappa_image": ki,
            "kappa_text": kt,
            "alpha_image": ai,
            "alpha_text": at,
            "js_divergence": jsd,
            "knn_mixing_rate": mixing,
            "paired_cosine_sim": paired,
            # raw eigenvalues for plotting
            "_eigvals_image": evi.tolist(),
            "_eigvals_text": evt.tolist(),
            "_cosine_sim_image": cs_img.tolist(),
            "_cosine_sim_text": cs_txt.tolist(),
        }
