"""
Vision Models
=============

Pre-trained model architectures for computer vision tasks.

Classification Models
---------------------
- AlexNet: alexnet
- VGG: vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
- ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
- Wide ResNet: wide_resnet50_2, wide_resnet101_2
- ResNeXt: resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
- DenseNet: densenet121, densenet161, densenet169, densenet201
- SqueezeNet: squeezenet1_0, squeezenet1_1
- GoogLeNet: googlenet
- Inception: inception_v3
- MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- ShuffleNet: shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
- MNASNet: mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
- EfficientNet: efficientnet_b0 through efficientnet_b7
- EfficientNetV2: efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
- RegNet: regnet_x_*, regnet_y_*
- ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
- VisionTransformer: vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
- SwinTransformer: swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
- MaxVit: maxvit_t

Detection Models (in detection/)
--------------------------------
- Faster R-CNN: fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
- FCOS: fcos_resnet50_fpn
- RetinaNet: retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
- SSD: ssd300_vgg16, ssdlite320_mobilenet_v3_large
- Mask R-CNN: maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
- Keypoint R-CNN: keypointrcnn_resnet50_fpn

Segmentation Models (in segmentation/)
--------------------------------------
- FCN: fcn_resnet50, fcn_resnet101
- DeepLabV3: deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
- LRASPP: lraspp_mobilenet_v3_large

Video Models (in video/)
------------------------
- Video ResNet: r3d_18, mc3_18, r2plus1d_18
- Video S3D: s3d
- Video MViT: mvit_v1_b, mvit_v2_s
- Video SwinTransformer: swin3d_t, swin3d_s, swin3d_b
"""

# Classification models
from .alexnet import alexnet, AlexNet
from .vgg import (
    vgg11, vgg13, vgg16, vgg19,
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
    VGG
)
from .resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    wide_resnet50_2, wide_resnet101_2,
    ResNet, BasicBlock, Bottleneck
)
from .resnext import (
    resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
)
from .densenet import (
    densenet121, densenet161, densenet169, densenet201,
    DenseNet
)
from .squeezenet import squeezenet1_0, squeezenet1_1, SqueezeNet
from .googlenet import googlenet, GoogLeNet
from .inception import inception_v3, InceptionV3
from .mobilenet import (
    mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
    MobileNetV2, MobileNetV3
)
from .shufflenet import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0,
    shufflenet_v2_x1_5, shufflenet_v2_x2_0,
    ShuffleNetV2
)
from .mnasnet import (
    mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3,
    MNASNet
)
from .efficientnet import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l,
    EfficientNet
)
from .regnet import (
    regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf,
    regnet_x_8gf, regnet_x_16gf, regnet_x_32gf,
    regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf,
    regnet_y_8gf, regnet_y_16gf, regnet_y_32gf, regnet_y_128gf,
    RegNet
)
from .convnext import (
    convnext_tiny, convnext_small, convnext_base, convnext_large,
    ConvNeXt
)
from .vision_transformer import (
    vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14,
    VisionTransformer
)
from .swin_transformer import (
    swin_t, swin_s, swin_b,
    swin_v2_t, swin_v2_s, swin_v2_b,
    SwinTransformer
)
from .maxvit import maxvit_t, MaxViT

# Submodules
from . import detection
from . import segmentation
from . import video

# Model registry utilities
from ._api import list_models, get_model, get_model_weights

__all__ = [
    # AlexNet
    'alexnet', 'AlexNet',
    # VGG
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'VGG',
    # ResNet
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'wide_resnet50_2', 'wide_resnet101_2',
    'ResNet', 'BasicBlock', 'Bottleneck',
    # ResNeXt
    'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
    # DenseNet
    'densenet121', 'densenet161', 'densenet169', 'densenet201', 'DenseNet',
    # SqueezeNet
    'squeezenet1_0', 'squeezenet1_1', 'SqueezeNet',
    # GoogLeNet
    'googlenet', 'GoogLeNet',
    # Inception
    'inception_v3', 'InceptionV3',
    # MobileNet
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'MobileNetV2', 'MobileNetV3',
    # ShuffleNet
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'ShuffleNetV2',
    # MNASNet
    'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'MNASNet',
    # EfficientNet
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'EfficientNet',
    # RegNet
    'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',
    'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf',
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf',
    'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', 'regnet_y_128gf', 'RegNet',
    # ConvNeXt
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'ConvNeXt',
    # Vision Transformer
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'VisionTransformer',
    # Swin Transformer
    'swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b', 'SwinTransformer',
    # MaxViT
    'maxvit_t', 'MaxViT',
    # Submodules
    'detection', 'segmentation', 'video',
    # Utilities
    'list_models', 'get_model', 'get_model_weights',
]



