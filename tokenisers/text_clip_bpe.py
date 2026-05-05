import torch
from transformers import CLIPTokenizer
from .base import AbstractTextTokeniser


class CLIPBPETokeniser(AbstractTextTokeniser):
    """
    Wraps the CLIP BPE tokeniser (49408-token vocab).
    The tokeniser is purely a discretiser — it does not borrow CLIP's embedding space.
    """

    VOCAB_SIZE = 49408

    def __init__(self, max_seq_len: int = 77):
        self.max_seq_len = max_seq_len
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def encode(self, text: str) -> torch.Tensor:
        out = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return out["input_ids"].squeeze(0)

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE
