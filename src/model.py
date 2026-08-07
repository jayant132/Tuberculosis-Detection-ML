import torch
import torch.nn as nn


# Custom lightweight CNN: the dataset is small and imbalanced, grayscale, and no
# torchvision is installed — cheaper to train than a pretrained ResNet and less
# likely to overfit than a deep network.


class TBCNN(nn.Module):
    def __init__(self, input_size, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 128),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, input_size, input_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg):
    return TBCNN(input_size=cfg["image_size"])
