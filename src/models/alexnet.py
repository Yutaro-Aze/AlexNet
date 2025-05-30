from torch import Tensor, nn


class AlexNet(nn.Module):
    """AlexNet model architecture.

    the paper: "ImageNet Classification with Deep Convolutional Neural Networks".

    Args:
        num_classes (int): Number of classes for classification. Default is 1000.

    """

    def __init__(self, num_classes: int = 1000) -> None:
        """Initialize the AlexNet model."""
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        conv2: nn.Conv2d = self.conv2[0]
        conv4: nn.Conv2d = self.conv4[0]
        conv5: nn.Conv2d = self.conv5[0]
        conv_change_list = [conv2, conv4, conv5]
        for conv in conv_change_list:
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the AlexNet model.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W) where N is batch size,
                        C is number of channels, H is height, and W is width.

        Returns:
            Tensor: Output tensor after passing through the model.

        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    import torchinfo

    model = AlexNet()
    torchinfo.summary(model, (1, 3, 224, 224))
