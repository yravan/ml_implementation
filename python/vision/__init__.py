"""
Vision Package
==============

A torchvision-like package for our NumPy-based deep learning framework.

This package provides:
- Pre-defined model architectures (classification, detection, segmentation, video)
- Image transforms for data augmentation and preprocessing
- Vision-specific operations (NMS, RoI pooling, etc.)
- Dataset utilities

Example:
    >>> from python.vision import models, transforms
    >>>
    >>> # Load a model
    >>> model = models.resnet50(num_classes=1000)
    >>>
    >>> # Create a transform pipeline
    >>> transform = transforms.Compose([
    ...     transforms.Resize(256),
    ...     transforms.CenterCrop(224),
    ...     transforms.ToTensor(),
    ...     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    ...                          std=[0.229, 0.224, 0.225])
    ... ])
"""

from . import models
from . import transforms
from . import ops

__all__ = ['models', 'transforms', 'ops']
