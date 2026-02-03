import torch
import torch.nn as nn


class MajorityBaseline(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer("majority_class", torch.tensor(1, dtype=torch.long))

    def set_majority_class(self, class_index: int) -> None:
        self.majority_class.fill_(class_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        logits = torch.full(
            (batch, self.n_classes), float("-inf"), device=x.device, dtype=x.dtype
        )
        logits[:, self.majority_class.item()] = 0.0
        return logits
