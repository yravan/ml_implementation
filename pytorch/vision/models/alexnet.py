"""
AlexNet
=======

AlexNet model architecture from "ImageNet Classification with Deep Convolutional
Neural Networks" - https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

The groundbreaking 2012 paper that sparked the deep learning revolution in computer vision.

Key innovations:
- ReLU activation (faster training than tanh)
- Dropout for regularization
- Data augmentation
- GPU training with model parallelism
- Local Response Normalization (LRN)

Architecture:
    Input: 224x224x3
    Conv1: 96 filters, 11x11, stride 4 -> 55x55x96
    MaxPool: 3x3, stride 2 -> 27x27x96
    Conv2: 256 filters, 5x5, pad 2 -> 27x27x256
    MaxPool: 3x3, stride 2 -> 13x13x256
    Conv3: 384 filters, 3x3, pad 1 -> 13x13x384
    Conv4: 384 filters, 3x3, pad 1 -> 13x13x384
    Conv5: 256 filters, 3x3, pad 1 -> 13x13x256
    MaxPool: 3x3, stride 2 -> 6x6x256
    Flatten: 9216
    FC1: 4096 (with dropout)
    FC2: 4096 (with dropout)
    FC3: num_classes
"""
import math

from torch.nn import (
    Module,
    Sequential,
    ReLU,
    MaxPool2d,
    Conv2d,
    Dropout,
    Linear,
    AdaptiveAvgPool2d,
)


class AlexNet(Module):
    """
    AlexNet model.

    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv_layers = Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = Sequential(
            Dropout(p=self.dropout),
            Linear(256 * 6 * 6, 4096),
            ReLU(),
            Dropout(p=self.dropout),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, num_classes),
        )
        self.apply(AlexNet._init_weights)

    @classmethod
    def _init_weights(cls, module):
        if isinstance(module, (Conv2d, Linear)):
            # Kaiming init for ReLU networks
            kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                zeros_(module.bias)



    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        features = self.conv_layers(x)
        features = features.reshape(features.shape[0], -1)
        output = self.classifier(features)
        return output


def alexnet(num_classes: int = 1000, dropout: float = 0.5, **kwargs) -> AlexNet:
    """
    AlexNet model.

    Args:
        num_classes: Number of output classes
        dropout: Dropout probability

    Returns:
        AlexNet model instance
    """
    return AlexNet(num_classes=num_classes, dropout=dropout, **kwargs)
