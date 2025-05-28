import torch.nn as nn
from torch import Tensor


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return x


if __name__ == "__main__":
    import torchinfo

    model = AlexNet()
    torchinfo.summary(model, (1, 3, 224, 224))
