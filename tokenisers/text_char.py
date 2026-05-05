import torch
from .base import AbstractTextTokeniser


class CharTextTokeniser(AbstractTextTokeniser):
    """ASCII character-level tokeniser (vocab 0–127, 0 = padding)."""

    def __init__(self, char_vocab: int = 128, max_seq_len: int = 64):
        self.char_vocab = char_vocab
        self.max_seq_len = max_seq_len

    def encode(self, text: str) -> torch.Tensor:
        ids = [min(ord(ch), self.char_vocab - 1) for ch in text[: self.max_seq_len]]
        ids += [0] * (self.max_seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return self.char_vocab
