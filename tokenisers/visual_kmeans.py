import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from .base import AbstractVisualTokeniser


class KMeansVisualTokeniser(AbstractVisualTokeniser):
    """
    Encodes images as sequences of k-means cluster IDs over patch colour/texture features.
    Token IDs are offset so they don't collide with text token IDs.
    """

    def __init__(
        self,
        codebook_k: int = 64,
        patch_size: int = 16,
        img_size: int = 224,
        max_seq_len: int = 64,
        token_offset: int = 128,
        seed: int = 42,
    ):
        self.codebook_k = codebook_k
        self.patch_size = patch_size
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.token_offset = token_offset
        self.seed = seed
        self.km: KMeans | None = None

    def _patch_features(self, img: Image.Image) -> np.ndarray:
        arr = np.array(img.resize((self.img_size, self.img_size)), dtype=np.float32) / 255.0
        feats = []
        for r in range(0, self.img_size, self.patch_size):
            for c in range(0, self.img_size, self.patch_size):
                patch = arr[r : r + self.patch_size, c : c + self.patch_size, :]
                feats.append(np.concatenate([patch.mean(axis=(0, 1)), patch.std(axis=(0, 1))]))
        return np.array(feats)

    def fit(self, images: list[Image.Image], n_fit: int = 100) -> None:
        all_feats = np.concatenate([self._patch_features(img) for img in images[:n_fit]])
        self.km = KMeans(
            n_clusters=self.codebook_k, n_init=5, max_iter=200, random_state=self.seed
        )
        self.km.fit(all_feats)

    def encode(self, image: Image.Image) -> torch.Tensor:
        assert self.km is not None, "Call .fit() before .encode()"
        feats = self._patch_features(image)
        ids = self.km.predict(feats) + self.token_offset
        ids = ids[: self.max_seq_len].tolist()
        ids += [0] * (self.max_seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return self.token_offset + self.codebook_k + 1
