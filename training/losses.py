import torch
import torch.nn.functional as F


def infonce(e1: torch.Tensor, e2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = (e1 @ e2.T) / temperature
    labels = torch.arange(len(e1), device=e1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
