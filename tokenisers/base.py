from abc import ABC, abstractmethod
from PIL import Image
import torch


class AbstractVisualTokeniser(ABC):
    @abstractmethod
    def fit(self, images: list[Image.Image]) -> None:
        ...

    @abstractmethod
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Returns a 1-D LongTensor of token IDs."""
        ...


class AbstractTextTokeniser(ABC):
    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Returns a 1-D LongTensor of token IDs."""
        ...
