from typing import Iterable, Sequence, Optional
import torch
import torch.nn as nn
from einops import rearrange
from src.models.components import BiN


def resolve_feature_indices(n_features: int, mode: str, indices: Optional[Sequence[int]],
                           event_count: Optional[int], lob_count: Optional[int]) -> Optional[Iterable[int]]:
    if indices is not None:
        return indices
    if mode == "all":
        return None
    if event_count is None or lob_count is None:
        raise ValueError("event_count and lob_count required for feature_mode != 'all'")
    if mode == "events":
        return range(0, event_count)
    if mode == "lob":
        return range(event_count, event_count + lob_count)
    if mode == "no-raw":
        return list(range(3, event_count)) + list(range(event_count, event_count + lob_count))
    raise ValueError(f"Unknown feature_mode: {mode}")


class BaseLiTTransformer(nn.Module):
    def __init__(self, n_features: int, window: int, feature_mode: str, feature_indices: Optional[Sequence[int]],
                 event_count: Optional[int], lob_count: Optional[int], d_model: int, n_heads: int,
                 n_layers: int, num_classes: int, dropout: float,
                 use_bin: bool = False, use_event_embed: bool = False, use_mean_pool: bool = False):
        super().__init__()
        self.window = window
        self.use_bin = use_bin
        self.use_event_embed = use_event_embed
        self.use_mean_pool = use_mean_pool

        if use_event_embed and n_features > 41:
            self.order_type_embedder = nn.Embedding(3, 1)
            input_features = n_features
        else:
            self.order_type_embedder = None
            input_features = n_features

        indices = resolve_feature_indices(n_features, feature_mode.lower(), feature_indices, event_count, lob_count)
        if indices is None:
            self.feature_indices = None
            if use_bin:
                self.bin_layer = BiN(input_features, window)
            else:
                self.bin_layer = None
        else:
            self.register_buffer("feature_indices", torch.tensor(list(indices), dtype=torch.long))
            input_features = len(indices)
            if use_bin:
                self.bin_layer = BiN(input_features, window)
            else:
                self.bin_layer = None

        self.input_proj = nn.Linear(input_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                                                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, window + 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_classes))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order_type_embedder is not None and x.shape[-1] > 41:
            continuous = torch.cat([x[:, :, :41], x[:, :, 42:]], dim=2)
            order_type = x[:, :, 41].long().clamp(0, 2)
            order_emb = self.order_type_embedder(order_type)
            x = torch.cat([continuous, order_emb], dim=2)
        if self.feature_indices is not None:
            x = torch.index_select(x, dim=2, index=self.feature_indices)
        if self.bin_layer is not None:
            x = rearrange(x, "b s f -> b f s")
            x = self.bin_layer(x)
            x = rearrange(x, "b f s -> b s f")
        B = x.size(0)
        x = self.input_proj(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_emb
        x = self.encoder(x)
        if self.use_mean_pool:
            pooled = (x[:, 0] + x[:, 1:].mean(dim=1)) / 2
            return self.head(self.norm(pooled))
        return self.head(self.norm(x[:, 0]))


class LiTTransformer(BaseLiTTransformer):
    def __init__(self, n_features: int, window: int, feature_mode: str = "all",
                 feature_indices: Optional[Sequence[int]] = None, event_count: Optional[int] = None,
                 lob_count: Optional[int] = None, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, num_classes: int = 3, dropout: float = 0.2,
                 use_bin: bool = False, use_event_embed: bool = False, use_mean_pool: bool = False):
        super().__init__(n_features, window, feature_mode, feature_indices, event_count, lob_count,
                        d_model, n_heads, n_layers, num_classes, dropout,
                        use_bin=use_bin, use_event_embed=use_event_embed, use_mean_pool=use_mean_pool)
