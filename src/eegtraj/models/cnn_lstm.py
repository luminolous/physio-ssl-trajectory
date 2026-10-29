"""CNN-LSTM decoder: temporal conv -> spatial conv -> LSTM -> linear (baselines.md: cnn-lstm).

Input (B, n_chans, n_times) is a causal window ending at the target sample, so 'same'
padding inside the window uses only past EEG. The LSTM reads the window left to right
and the last hidden state predicts the 3D target at the window's last sample.
"""

from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(self, n_chans: int, n_times: int, n_outputs: int = 3, n_temporal: int = 8,
                 temporal_kernel: int = 25, n_spatial: int = 16, pool: int = 2, hidden: int = 32,
                 lstm_layers: int = 1, drop_prob: float = 0.25):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, n_temporal, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(n_temporal),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(n_temporal, n_spatial, (n_chans, 1), bias=False),
            nn.BatchNorm2d(n_spatial),
            nn.ELU(),
            nn.AvgPool2d((1, pool)),
            nn.Dropout(drop_prob),
        )
        self.lstm = nn.LSTM(n_spatial, hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=drop_prob if lstm_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_chans, n_times) -> (B, n_outputs)."""
        h = self.spatial(self.temporal(x.unsqueeze(1)))  # (B, n_spatial, 1, T')
        h = h.squeeze(2).transpose(1, 2)  # (B, T', n_spatial)
        out, _ = self.lstm(h)
        return self.head(out[:, -1])
