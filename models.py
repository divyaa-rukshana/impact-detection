import torch
import torch.nn as nn


class HIKNetLike(nn.Module):
    """Simplified version of the paper's design: temporal convs applied per
    channel, then a 2D conv that fuses the 6 channels (late fusion)."""

    def __init__(self, n_out=1, filters=32, k=15, dropout=0.4):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, filters, (1, k), padding=(0, k // 2)), nn.ReLU(),
            nn.Conv2d(filters, filters, (1, k), padding=(0, k // 2)), nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(filters, filters * 2, (6, 5), padding=(0, 2)), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * 2, n_out)

    def forward(self, x):                      # x: (B, T, C)
        x = x.permute(0, 2, 1).unsqueeze(1)    # (B, 1, C, T)
        x = self.temporal(x)                   # (B, F, C, T/2)
        x = self.fuse(x)                       # (B, 2F, 1, T/2)
        x = x.mean(dim=(2, 3))                 # global average pool
        return self.fc(self.drop(x))


class ConvBiGRU(nn.Module):
    """Your alternative: conv stem shortens the sequence 4x, then a
    bidirectional GRU models the timeline."""

    def __init__(self, n_out=1, hidden=64, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.gru = nn.GRU(32, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_out)

    def forward(self, x):                      # x: (B, T, C)
        x = self.stem(x.permute(0, 2, 1))      # (B, 32, T/4)
        out, _ = self.gru(x.permute(0, 2, 1))  # (B, T/4, 2*hidden)
        pooled = torch.cat([out.mean(1), out.max(1).values], dim=1)
        return self.fc(self.drop(pooled))