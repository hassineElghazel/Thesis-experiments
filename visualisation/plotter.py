import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

BG = "#F8F9FA"
CONFIG_COLORS = [
    "#E63946",  # CLIP final
    "#457B9D",  # CLIP token
    "#2A9D8F",  # Dual scratch
    "#F4A261",  # Option B
    "#9B59B6",  # TinyCLIP scratch
]
CONCEPT_CMAP = plt.cm.tab10


def _sty(ax, title: str) -> None:
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bbb")
    ax.tick_params(colors="#444", labelsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1a1a2e", pad=8)
    ax.yaxis.grid(True, color="#E0E0E0", lw=0.6)
    ax.set_axisbelow(True)


class Plotter:
    def __init__(self, out_dir: str = "outputs/synthetic"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def figure1_main_comparison(
        self,
        configs: list[dict],
        concept_ids: np.ndarray,
        n_concepts: int,
    ) -> None:
        """
        Figure 1: PCA panels + bar chart + eigenspectrum panels + cosine KDE.
        configs: list of {name, img_emb, txt_emb, metrics}
        """
        n_cfgs = len(configs)
        cc = CONCEPT_CMAP(np.linspace(0, 1, n_concepts))
        colors = CONFIG_COLORS[:n_cfgs]

        # Layout: PCA panels in row 0, eigenspectra + KDE in row 1
        # Row 0: n_cfgs PCA panels + 1 bar chart
        # Row 1: n_cfgs eigenspectrum panels + 1 KDE
        n_cols = max(n_cfgs + 1, 4)
        fig = plt.figure(figsize=(6 * n_cols, 12), facecolor=BG)
        fig.suptitle(
            "Modality Gap Comparison — All Configurations",
            fontsize=14, fontweight="bold", y=0.99, color="#1a1a2e",
        )
        gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.42, wspace=0.32,
                               left=0.04, right=0.97, top=0.93, bottom=0.06)

        # PCA panels
        for ci, cfg in enumerate(configs):
            ax = fig.add_subplot(gs[0, ci])
            _sty(ax, f"{chr(65+ci)} · PCA — {cfg['name']}")
            N = len(cfg["img_emb"])
            pca = PCA(2)
            proj = pca.fit_transform(np.vstack([cfg["img_emb"], cfg["txt_emb"]]))
            pi, pt = proj[:N], proj[N:]
            for cid in range(n_concepts):
                m = concept_ids == cid
                ax.scatter(pi[m, 0], pi[m, 1], c=[cc[cid]], marker="o", s=20, alpha=0.7)
                ax.scatter(pt[m, 0], pt[m, 1], c=[cc[cid]], marker="^", s=22, alpha=0.7)
            ci_mean, ct_mean = pi.mean(0), pt.mean(0)
            ax.annotate("", xy=ct_mean, xytext=ci_mean,
                        arrowprops=dict(arrowstyle="->", color="#333", lw=2))
            m = cfg["metrics"]
            ax.text(0.03, 0.96,
                    f"Gap={m['centroid_distance']:.4f}\nMix={m['knn_mixing_rate']:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.8, ec="#ccc"))
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        # Bar chart
        ax_bar = fig.add_subplot(gs[0, n_cfgs])
        _sty(ax_bar, f"{chr(65+n_cfgs)} · Metrics (normalised to first config=1)")
        metric_keys = ["centroid_distance", "js_divergence"]
        metric_labels = ["Centroid\nDist", "JS\nDiverg."]
        base_vals = [configs[0]["metrics"][k] for k in metric_keys]
        x = np.arange(len(metric_keys))
        w = 0.8 / n_cfgs
        for ci, cfg in enumerate(configs):
            norm_vals = [cfg["metrics"][k] / (bv + 1e-12) for k, bv in zip(metric_keys, base_vals)]
            bars = ax_bar.bar(x + ci * w - (n_cfgs - 1) * w / 2, norm_vals, w,
                              color=colors[ci], alpha=0.85, label=cfg["name"], edgecolor="w")
        ax_bar.axhline(1, color="#555", ls="--", lw=1.2, alpha=0.5)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_labels, fontsize=9)
        ax_bar.set_ylabel("Relative magnitude")
        ax_bar.legend(fontsize=7, loc="upper right")

        # Eigenspectrum panels
        for ci, cfg in enumerate(configs):
            ax = fig.add_subplot(gs[1, ci])
            _sty(ax, f"Eigenspectrum — {cfg['name']}")
            evi = np.array(cfg["metrics"]["_eigvals_image"])
            evt = np.array(cfg["metrics"]["_eigvals_text"])
            top = min(50, len(evi), len(evt))
            ax.plot(range(1, top + 1), evi[:top], color=colors[ci], lw=2,
                    label=f"Img α={cfg['metrics']['alpha_image']:.2f}")
            ax.plot(range(1, top + 1), evt[:top], color=colors[ci], lw=2, ls="--",
                    label=f"Txt α={cfg['metrics']['alpha_text']:.2f}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Eigenvalue")
            ax.legend(fontsize=8)

        # Cosine KDE
        ax_kde = fig.add_subplot(gs[1, n_cfgs])
        _sty(ax_kde, "Cosine Sim KDE")
        xs = np.linspace(-0.5, 1.0, 400)
        for ci, cfg in enumerate(configs):
            cs_img = np.array(cfg["metrics"]["_cosine_sim_image"])
            cs_txt = np.array(cfg["metrics"]["_cosine_sim_text"])
            try:
                y_img = gaussian_kde(cs_img, bw_method=0.06)(xs)
                y_txt = gaussian_kde(cs_txt, bw_method=0.06)(xs)
                ax_kde.plot(xs, y_img, color=colors[ci], lw=1.8, label=f"{cfg['name']} img")
                ax_kde.fill_between(xs, y_img, alpha=0.1, color=colors[ci])
            except Exception:
                pass
        ax_kde.set_xlabel("Cosine Similarity")
        ax_kde.set_ylabel("Density")
        ax_kde.legend(fontsize=7, loc="upper left")

        path = os.path.join(self.out_dir, "figure1_main_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved: {path}")
        plt.close(fig)

    def figure2_anisotropy(self, configs: list[dict]) -> None:
        """Figure 2: Residual covariance eigenvalue decay for all configs."""
        n = len(configs)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor=BG)
        if n == 1:
            axes = [axes]
        fig.suptitle("Anisotropy: Residual Covariance After Centroid Subtraction",
                     fontsize=13, fontweight="bold", color="#1a1a2e")
        colors = CONFIG_COLORS[:n]
        for ax, cfg, col in zip(axes, configs, colors):
            _sty(ax, cfg["name"])
            evi = np.array(cfg["metrics"]["_eigvals_image"])
            evt = np.array(cfg["metrics"]["_eigvals_text"])
            top = min(40, len(evi), len(evt))
            ax.plot(range(1, top + 1), evi[:top] / evi[0], color=col, lw=2, label="Img")
            ax.plot(range(1, top + 1), evt[:top] / evt[0], color=col, lw=2, ls="--", label="Txt")
            ax.axhline(1 / top, color="#999", lw=1, ls=":", label="Isotropic")
            ax.set_yscale("log")
            ax.set_xlabel("Rank")
            ax.set_ylabel("λₖ/λ₁")
            ax.legend(fontsize=8)
            ki = cfg["metrics"]["kappa_image"]
            kt = cfg["metrics"]["kappa_text"]
            ax.text(0.97, 0.95, f"Img: κ={ki:.0f}\nTxt: κ={kt:.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="w", alpha=0.8))
        plt.tight_layout()
        path = os.path.join(self.out_dir, "figure2_anisotropy.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved: {path}")
        plt.close(fig)

    def figure3_per_concept(
        self,
        configs: list[dict],
        concept_ids: np.ndarray,
        concept_names: list[str],
    ) -> None:
        """Figure 3: Per-concept paired cosine similarity bar chart."""
        n_concepts = len(concept_names)
        n_cfgs = len(configs)
        colors = CONFIG_COLORS[:n_cfgs]

        fig, ax = plt.subplots(figsize=(max(14, 2 * n_concepts), 5), facecolor=BG)
        means = [np.mean([
            np.mean(np.sum(cfg["img_emb"][concept_ids == cid] *
                           cfg["txt_emb"][concept_ids == cid], axis=1))
            for cid in range(n_concepts)
        ]) for cfg in configs]
        title = "Per-Concept Paired Cosine Similarity  |  " + "  ".join(
            f"{cfg['name']} mean={m:.3f}" for cfg, m in zip(configs, means)
        )
        _sty(ax, title)
        x = np.arange(n_concepts)
        w = 0.8 / n_cfgs
        for ci, (cfg, col) in enumerate(zip(configs, colors)):
            per_concept = [
                np.mean(np.sum(cfg["img_emb"][concept_ids == cid] *
                               cfg["txt_emb"][concept_ids == cid], axis=1))
                for cid in range(n_concepts)
            ]
            ax.bar(x + ci * w - (n_cfgs - 1) * w / 2, per_concept, w,
                   color=col, alpha=0.85, label=cfg["name"], edgecolor="w")
        ax.set_xticks(x)
        ax.set_xticklabels(concept_names, fontsize=8)
        ax.set_ylabel("Paired Cosine Sim")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = os.path.join(self.out_dir, "figure3_per_concept.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved: {path}")
        plt.close(fig)
