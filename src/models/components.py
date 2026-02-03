from typing import Optional
import torch
from torch import nn
from src import config as cfg


class BiN(nn.Module):
    def __init__(self, d1: int, t1: int) -> None:
        super().__init__()
        self.t1 = t1
        self.d1 = d1
        self.B1 = nn.Parameter(torch.zeros(t1, 1))
        self.l1 = nn.Parameter(torch.empty(t1, 1))
        nn.init.xavier_normal_(self.l1)
        self.B2 = nn.Parameter(torch.zeros(d1, 1))
        self.l2 = nn.Parameter(torch.empty(d1, 1))
        nn.init.xavier_normal_(self.l2)
        self.y1 = nn.Parameter(torch.tensor([0.5]))
        self.y2 = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.y1[0] < 0:
            self.y1 = nn.Parameter(torch.tensor([0.01], device=x.device))
        if self.y2[0] < 0:
            self.y2 = nn.Parameter(torch.tensor([0.01], device=x.device))

        T2 = torch.ones([self.t1, 1], device=x.device)
        x2 = torch.mean(x, dim=2).unsqueeze(2)
        std = torch.std(x, dim=2).unsqueeze(2).clamp(min=1e-4)
        Z2 = (x - x2 @ T2.T) / (std @ T2.T)
        X2 = self.l2 @ T2.T * Z2 + self.B2 @ T2.T

        T1 = torch.ones([self.d1, 1], device=x.device)
        x1 = torch.mean(x, dim=1).unsqueeze(2)
        std = torch.std(x, dim=1).unsqueeze(2).clamp(min=1e-4)
        op1 = (x1 @ T1.T).permute(0, 2, 1)
        op2 = (std @ T1.T).permute(0, 2, 1)
        z1 = (x - op1) / op2
        X1 = T1 @ self.l1.T * z1 + T1 @ self.B1.T

        return self.y1 * X1 + self.y2 * X2


class MLP(nn.Module):
    def __init__(self, start_dim: int, hidden_dim: int, final_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.gelu(self.fc(x))
        x = self.fc2(x)
        if x.shape[2] == residual.shape[2]:
            x = x + residual
        return self.gelu(self.layer_norm(x))


def sinusoidal_positional_embedding(seq_size: int, dim: int, n: float = 10000.0,
                                    device: Optional[str] = None) -> torch.Tensor:
    positions = torch.arange(0, seq_size).unsqueeze(1)
    denominators = torch.pow(n, 2 * torch.arange(0, dim // 2) / dim)
    embeddings = torch.zeros(seq_size, dim)
    embeddings[:, 0::2] = torch.sin(positions / denominators)
    embeddings[:, 1::2] = torch.cos(positions / denominators)
    return embeddings.to(device or cfg.DEVICE, non_blocking=True)
