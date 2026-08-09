"""A deliberately tiny CNN for TB/normal classification.

Only ~60k parameters (vs 500k+ for the old model), so it trains on CPU
in seconds and fits on any machine. Global average pooling keeps the
classifier small no matter the input size.
"""

import torch.nn as nn


class TBCNN(nn.Module):
    def __init__(self, in_channels=1, hidden=(32, 64, 64), num_classes=2, dropout=0.5):
        super().__init__()
        blocks = []
        for out_channels in hidden:
            blocks.append(self._block(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(cfg):
    return TBCNN()
