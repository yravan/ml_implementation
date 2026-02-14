"""
ResNet
======

ResNet model architectures from "Deep Residual Learning for Image Recognition"
https://arxiv.org/abs/1512.03385

Key innovation: Skip connections (residual connections) that allow training
of very deep networks by addressing the vanishing gradient problem.

The residual block computes: y = F(x) + x
where F(x) is the residual mapping to be learned.

Variants:
    ResNet-18:  [2, 2, 2, 2] BasicBlocks
    ResNet-34:  [3, 4, 6, 3] BasicBlocks
    ResNet-50:  [3, 4, 6, 3] Bottlenecks
    ResNet-101: [3, 4, 23, 3] Bottlenecks
    ResNet-152: [3, 8, 36, 3] Bottlenecks

Wide ResNet variants use wider bottleneck layers (more channels).
"""

from typing import List, Optional, Type, Union
from python.nn_core import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Linear,
    kaiming_normal_,
    ones_,
    zeros_,
)


class BasicBlock(Module):
    """
    Basic residual block for ResNet-18/34.

    Structure:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU

    Args:
        inplanes: Input channels
        planes: Output channels
        stride: Stride for first conv
        downsample: Downsample module for skip connection
    """
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
    ):
        super().__init__()
        self.layers = Sequential(
            Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(planes),
            ReLU(),
            Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(planes),
        )
        self.downsample = downsample
        self.relu = ReLU()

    def forward(self, x):
        """Forward with residual connection."""
        out = self.layers(x)
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += x
        return self.relu(out)


class Bottleneck(Module):
    """
    Bottleneck block for ResNet-50/101/152.

    Structure:
        x -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> (+x) -> ReLU

    The 1x1 convs reduce then restore dimensionality, making the 3x3 conv cheaper.

    Args:
        inplanes: Input channels
        planes: Bottleneck channels (output = planes * expansion)
        stride: Stride for 3x3 conv
        downsample: Downsample module for skip connection
    """
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
    ):
        super().__init__()
        self.layers = Sequential(
            Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(planes),
            ReLU(),
            Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(planes),
            ReLU(),
            Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(planes * self.expansion),
        )
        self.downsample = downsample

        self.relu = ReLU()

    def forward(self, x):
        """Forward with residual connection."""
        out = self.layers(x)
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += x
        return self.relu(out)


class ResNet(Module):
    """
    ResNet model.

    Args:
        block: Block type (BasicBlock or Bottleneck)
        layers: Number of blocks in each layer [layer1, layer2, layer3, layer4]
        num_classes: Number of output classes
        zero_init_residual: Zero-init the last BN in each block
        width_per_group: Base width for Wide ResNet
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        width_per_group: int = 64,
    ):
        super().__init__()
        self.inplanes = 64
        self.base_width = width_per_group

        self.stage_0 = Sequential(
            Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(self.inplanes),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block,64, layers[0])
        self.layer2 = self._make_layer(block,128, layers[1], stride=2)
        self.layer3 = self._make_layer(block,256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)

        self.apply(lambda x: ResNet._init_weights(x, zero_init_residual))


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes, blocks, stride=1):
        """Create a ResNet layer."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(blocks - 1):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))
        return Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        x = self.stage_0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    @classmethod
    def _init_weights(cls, module, zero_init_residual):
        if isinstance(module, (Conv2d, Linear)):
            # Kaiming init for ReLU networks
            kaiming_normal_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)

        elif isinstance(module, BatchNorm2d):
            ones_(module.weight)
            zeros_(module.bias)

        # Zero-init last BN in each residual block
        if zero_init_residual:
            if isinstance(module, Bottleneck):
                # Last BN is the 8th layer (index 7) in the Sequential
                zeros_(module.layers._modules["7"].weight)
            elif isinstance(module, BasicBlock):
                # Last BN is the 5th layer (index 4)
                zeros_(module.layers._modules["4"].weight)


# Model constructors
def resnet18(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def resnet34(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet50(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet101(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

def resnet152(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-152 model."""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)

def wide_resnet50_2(num_classes: int = 1000, **kwargs) -> ResNet:
    """Wide ResNet-50-2 model (2x wider bottleneck)."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, width_per_group=128, **kwargs)

def wide_resnet101_2(num_classes: int = 1000, **kwargs) -> ResNet:
    """Wide ResNet-101-2 model (2x wider bottleneck)."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, width_per_group=128, **kwargs)
