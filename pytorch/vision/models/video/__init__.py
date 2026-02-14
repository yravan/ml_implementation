"""
Video Classification Models
===========================

Models for video understanding and action recognition.

3D ConvNet-based:
- R3D: 3D ResNets
- MC3: Mixed 3D/2D convolutions
- R(2+1)D: Factorized 3D convolutions
- S3D: Separable 3D convolutions

Transformer-based:
- MViT: Multiscale Vision Transformer
- Video Swin: Video Swin Transformer
"""

from .resnet import r3d_18, mc3_18, r2plus1d_18, VideoResNet
from .s3d import s3d, S3D
from .mvit import mvit_v1_b, mvit_v2_s, MViT
from .swin_transformer import swin3d_t, swin3d_s, swin3d_b, SwinTransformer3D

__all__ = [
    'r3d_18', 'mc3_18', 'r2plus1d_18', 'VideoResNet',
    's3d', 'S3D',
    'mvit_v1_b', 'mvit_v2_s', 'MViT',
    'swin3d_t', 'swin3d_s', 'swin3d_b', 'SwinTransformer3D',
]
