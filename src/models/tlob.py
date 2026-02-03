import torch
from torch import nn
from einops import rearrange
from src import config as cfg
from src.models.components import BiN, sinusoidal_positional_embedding
from src.models.attention import TransformerLayer, DecayTransformerLayer


class BaseTLOB(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, seq_size: int, num_features: int,
                 num_heads: int, is_sin_emb: bool, use_decay: bool = False,
                 init_decay: float = 0.1, device: str = "cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.use_decay = use_decay
        
        self.order_type_embedder = nn.Embedding(3, 1)
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        
        if is_sin_emb:
            self.pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim, device=device)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))

        self.layers = nn.ModuleList()
        LayerClass = DecayTransformerLayer if use_decay else TransformerLayer
        
        for i in range(num_layers):
            if i != num_layers - 1:
                if use_decay:
                    self.layers.append(LayerClass(hidden_dim, num_heads, hidden_dim, seq_size, init_decay))
                    self.layers.append(LayerClass(seq_size, num_heads, seq_size, hidden_dim, init_decay))
                else:
                    self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                    self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
            else:
                if use_decay:
                    self.layers.append(LayerClass(hidden_dim, num_heads, hidden_dim // 4, seq_size, init_decay))
                    self.layers.append(LayerClass(seq_size, num_heads, seq_size // 4, hidden_dim // 4, init_decay))
                else:
                    self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim // 4))
                    self.layers.append(TransformerLayer(seq_size, num_heads, seq_size // 4))
        
        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, 3))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] > 41:
            continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
            order_type = input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x = input
        
        x = rearrange(x, 'b s f -> b f s')
        x = self.norm_layer(x)
        x = rearrange(x, 'b f s -> b s f')
        x = self.emb_layer(x) + self.pos_encoder
        
        for layer in self.layers:
            x, _ = layer(x)
            x = x.permute(0, 2, 1)
        
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x


class TLOB(BaseTLOB):
    def __init__(self, hidden_dim: int = cfg.HIDDEN_DIM, num_layers: int = cfg.NUM_LAYERS,
                 seq_size: int = cfg.SEQ_SIZE, num_features: int = 46,
                 num_heads: int = cfg.NUM_HEADS, is_sin_emb: bool = cfg.IS_SIN_EMB) -> None:
        super().__init__(hidden_dim, num_layers, seq_size, num_features, num_heads,
                        is_sin_emb, use_decay=False)


class TLOBDecay(BaseTLOB):
    def __init__(self, hidden_dim: int = cfg.HIDDEN_DIM, num_layers: int = cfg.NUM_LAYERS,
                 seq_size: int = cfg.SEQ_SIZE, num_features: int = 46,
                 num_heads: int = cfg.NUM_HEADS, is_sin_emb: bool = True,
                 init_decay: float = 0.1, device: str = "cpu") -> None:
        super().__init__(hidden_dim, num_layers, seq_size, num_features, num_heads,
                        is_sin_emb, use_decay=True, init_decay=init_decay, device=device)

    def get_decay_rates(self) -> dict:
        return {f"layer_{i}_{'feature' if i % 2 == 0 else 'temporal'}": layer.attention.lambda_decay.detach().cpu()
                for i, layer in enumerate(self.layers)}

    def get_all_lambda_raw(self) -> list:
        return [layer.attention.lambda_raw for layer in self.layers]
