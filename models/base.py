from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class AbstractEncoder(ABC):
    """
    Common interface for all configs.
    Each encoder must return L2-normalised float32 embeddings.
    """

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Returns (N, D) float32 array of L2-normalised image embeddings."""
        ...

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Returns (N, D) float32 array of L2-normalised text embeddings."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
