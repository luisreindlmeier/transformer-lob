import torch.nn as nn

N_LOB_FEATURES = 40


class DeepLOB(nn.Module):
    def __init__(self, n_features: int, num_classes: int = 3, dropout: float = 0.3, gru_dim: int = 128):
        super().__init__()
        lob_dim = min(N_LOB_FEATURES, n_features)
        self.lob_dim = lob_dim
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2)), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2)), nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(),
        )
        conv_out_dim = 64 * lob_dim
        self.proj = nn.Linear(conv_out_dim, gru_dim)
        self.gru = nn.GRU(input_size=gru_dim, hidden_size=gru_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_dim, num_classes)

    def forward(self, x):
        B, T, F = x.shape
        x = x[:, :, : self.lob_dim].contiguous()
        x = x.unsqueeze(1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)
        x = self.proj(x)
        _, h = self.gru(x)
        h = self.dropout(h.squeeze(0))
        return self.fc(h)
