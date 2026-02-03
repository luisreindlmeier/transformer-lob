from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config as cfg
from src.models.components import MLP


class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q(x), self.k(x), self.v(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(hidden_dim * num_heads, num_heads, batch_first=True, device=cfg.DEVICE)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = x
        q, k, v = self.qkv(x)
        x, att = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        x = self.w0(x) + res
        x = self.mlp(self.norm(x))
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, att


class DecayAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1,
                 init_decay: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        
        init_raw = torch.log(torch.exp(torch.tensor(init_decay)) - 1)
        self.lambda_raw = nn.Parameter(torch.full((n_heads,), init_raw.item()))
        
        positions = torch.arange(max_seq_len)
        self.register_buffer('distances', torch.abs(positions[:, None] - positions[None, :]).float())
    
    @property
    def lambda_decay(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        lambdas = self.lambda_decay.view(1, self.n_heads, 1, 1)
        decay_mask = torch.exp(-lambdas * self.distances[:T, :T].unsqueeze(0).unsqueeze(0))
        attn = attn * decay_mask
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class DecayTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int, max_seq_len: int,
                 init_decay: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = DecayAttention(hidden_dim, num_heads, max_seq_len, dropout=0.0, init_decay=init_decay)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        res = x
        x = self.attention(self.norm1(x)) + res
        res = x
        x = self.mlp(self.norm2(x))
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, None
