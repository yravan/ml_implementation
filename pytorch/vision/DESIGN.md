# Vision Package Design

A torchvision-like package for our NumPy-based deep learning framework.

## Package Structure

```
python/vision/
├── __init__.py
├── models/
│   ├── __init__.py
│   │
│   ├── # ===== CLASSIFICATION MODELS =====
│   ├── alexnet.py          # AlexNet (2012) - the OG deep CNN
│   ├── vgg.py              # VGG11, VGG13, VGG16, VGG19 (with/without batch norm)
│   ├── resnet.py           # ResNet18, 34, 50, 101, 152 + Wide ResNet
│   ├── densenet.py         # DenseNet121, 161, 169, 201
│   ├── squeezenet.py       # SqueezeNet 1.0, 1.1
│   ├── inception.py        # Inception v3, GoogLeNet
│   ├── mobilenet.py        # MobileNetV2, MobileNetV3 (Small/Large)
│   ├── shufflenet.py       # ShuffleNet V2
│   ├── efficientnet.py     # EfficientNet B0-B7, EfficientNetV2
│   ├── resnext.py          # ResNeXt50_32x4d, ResNeXt101_32x8d, etc.
│   ├── mnasnet.py          # MNASNet 0.5, 0.75, 1.0, 1.3
│   ├── convnext.py         # ConvNeXt Tiny/Small/Base/Large
│   ├── regnet.py           # RegNetX, RegNetY variants
│   ├── vision_transformer.py  # ViT-B/16, ViT-B/32, ViT-L/16, etc.
│   ├── swin_transformer.py    # Swin-T, Swin-S, Swin-B, Swin-L
│   ├── maxvit.py           # MaxViT
│   │
│   ├── # ===== DETECTION MODELS =====
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── faster_rcnn.py     # Faster R-CNN with various backbones
│   │   ├── mask_rcnn.py       # Mask R-CNN (instance segmentation)
│   │   ├── retinanet.py       # RetinaNet with FPN
│   │   ├── ssd.py             # SSD300, SSD512, SSDLite
│   │   ├── fcos.py            # FCOS (Fully Convolutional One-Stage)
│   │   ├── anchor_utils.py    # Anchor generation utilities
│   │   ├── roi_heads.py       # Region of Interest pooling/heads
│   │   └── rpn.py             # Region Proposal Network
│   │
│   ├── # ===== SEGMENTATION MODELS =====
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── fcn.py             # FCN with ResNet backbone
│   │   ├── deeplabv3.py       # DeepLabV3, DeepLabV3+
│   │   ├── lraspp.py          # Lite R-ASPP (MobileNetV3)
│   │   ├── unet.py            # U-Net (classic medical imaging)
│   │   └── segmentation_utils.py
│   │
│   ├── # ===== VIDEO MODELS =====
│   ├── video/
│   │   ├── __init__.py
│   │   ├── r3d.py             # 3D ResNets (R3D-18, MC3, R2+1D)
│   │   ├── s3d.py             # S3D
│   │   ├── mvit.py            # MViT (Multiscale Vision Transformer)
│   │   └── swin3d.py          # Video Swin Transformer
│   │
│   └── # ===== COMMON BUILDING BLOCKS =====
│       _utils.py              # Model utilities, weight loading
│       _api.py                # list_models(), get_model(), etc.
│
├── transforms/
│   ├── __init__.py
│   ├── v2/                    # Modern transforms API
│   │   ├── __init__.py
│   │   │
│   │   ├── # ===== GEOMETRY TRANSFORMS =====
│   │   ├── _geometry.py
│   │   │   # - Resize, RandomResize
│   │   │   # - CenterCrop, RandomCrop, RandomResizedCrop, FiveCrop, TenCrop
│   │   │   # - Pad, RandomPad
│   │   │   # - RandomHorizontalFlip, RandomVerticalFlip
│   │   │   # - RandomRotation, RandomAffine
│   │   │   # - RandomPerspective, ElasticTransform
│   │   │   # - ScaleJitter, RandomShortestSize, RandomZoomOut
│   │   │
│   │   ├── # ===== COLOR TRANSFORMS =====
│   │   ├── _color.py
│   │   │   # - Grayscale, RandomGrayscale, RGB
│   │   │   # - ColorJitter (brightness, contrast, saturation, hue)
│   │   │   # - RandomPhotometricDistort
│   │   │   # - RandomChannelPermutation
│   │   │   # - RandomInvert, RandomPosterize, RandomSolarize
│   │   │   # - RandomAdjustSharpness, RandomAutocontrast, RandomEqualize
│   │   │   # - GaussianBlur, GaussianNoise
│   │   │
│   │   ├── # ===== TYPE CONVERSION =====
│   │   ├── _type_conversion.py
│   │   │   # - ToTensor, ToImage, ToPILImage, ToDtype
│   │   │   # - ConvertImageDtype
│   │   │
│   │   ├── # ===== NORMALIZATION =====
│   │   ├── _normalize.py
│   │   │   # - Normalize (mean/std)
│   │   │   # - ClampBoundingBoxes, SanitizeBoundingBoxes
│   │   │
│   │   ├── # ===== COMPOSITION =====
│   │   ├── _container.py
│   │   │   # - Compose
│   │   │   # - RandomApply, RandomChoice, RandomOrder
│   │   │
│   │   ├── # ===== AUTO AUGMENT =====
│   │   ├── _auto_augment.py
│   │   │   # - AutoAugment (ImageNet, CIFAR10, SVHN policies)
│   │   │   # - RandAugment
│   │   │   # - TrivialAugmentWide
│   │   │   # - AugMix
│   │   │
│   │   ├── # ===== AUGMENTATION MIXING =====
│   │   ├── _augment.py
│   │   │   # - CutMix, MixUp
│   │   │   # - RandomErasing (Cutout)
│   │   │
│   │   ├── # ===== MISC =====
│   │   ├── _misc.py
│   │   │   # - Identity, Lambda
│   │   │   # - LinearTransformation (whitening)
│   │   │   # - PermuteDimensions, TransposeDimensions
│   │   │
│   │   └── functional.py      # Functional interface for all transforms
│   │
│   └── functional.py          # Legacy functional transforms
│
├── ops/                       # Vision-specific operations
│   ├── __init__.py
│   ├── boxes.py               # Box utilities (IoU, NMS, etc.)
│   ├── roi_align.py           # RoI Align
│   ├── roi_pool.py            # RoI Pooling
│   ├── ps_roi_align.py        # Position-Sensitive RoI Align
│   ├── deform_conv.py         # Deformable Convolution
│   ├── focal_loss.py          # Focal Loss
│   └── misc.py                # FrozenBatchNorm2d, etc.
│
├── datasets/                  # Dataset loaders (optional, for later)
│   ├── __init__.py
│   ├── cifar.py               # CIFAR-10, CIFAR-100
│   ├── mnist.py               # MNIST, FashionMNIST
│   ├── imagenet.py            # ImageNet
│   ├── coco.py                # COCO detection/segmentation
│   ├── voc.py                 # Pascal VOC
│   └── folder.py              # ImageFolder for custom datasets
│
└── utils/
    ├── __init__.py
    ├── draw.py                # draw_bounding_boxes, draw_segmentation_masks
    ├── make_grid.py           # make_grid for visualizing batches
    └── save_image.py          # save_image utility
```

## Model Details

### Classification Models (by era)

| Model | Year | Key Innovation | Variants |
|-------|------|----------------|----------|
| AlexNet | 2012 | ReLU, Dropout, GPU training | - |
| VGG | 2014 | Deep stacks of 3x3 convs | 11, 13, 16, 19 (±BN) |
| GoogLeNet | 2014 | Inception modules | - |
| ResNet | 2015 | Skip connections | 18, 34, 50, 101, 152 |
| DenseNet | 2016 | Dense connections | 121, 161, 169, 201 |
| SqueezeNet | 2016 | Fire modules (squeeze/expand) | 1.0, 1.1 |
| ResNeXt | 2017 | Grouped convolutions | 50_32x4d, 101_32x8d |
| MobileNetV2 | 2018 | Inverted residuals | - |
| ShuffleNetV2 | 2018 | Channel shuffle | 0.5x, 1.0x, 1.5x, 2.0x |
| EfficientNet | 2019 | Compound scaling | B0-B7 |
| RegNet | 2020 | Design space search | X/Y variants |
| ViT | 2020 | Pure transformer | B/16, B/32, L/16, L/32 |
| ConvNeXt | 2022 | Modernized ConvNet | T, S, B, L |
| Swin | 2021 | Hierarchical ViT | T, S, B, L |
| MaxViT | 2022 | Multi-axis attention | T, S, B |

### Detection Models

| Model | Type | Backbone Options |
|-------|------|------------------|
| Faster R-CNN | Two-stage | ResNet50-FPN, MobileNetV3 |
| Mask R-CNN | Instance seg | ResNet50-FPN |
| RetinaNet | One-stage | ResNet50-FPN |
| SSD | One-stage | VGG16, MobileNetV3 |
| FCOS | Anchor-free | ResNet50-FPN |

### Segmentation Models

| Model | Type | Backbone Options |
|-------|------|------------------|
| FCN | Semantic | ResNet50, ResNet101 |
| DeepLabV3 | Semantic | ResNet50, ResNet101, MobileNetV3 |
| LRASPP | Semantic (lite) | MobileNetV3 |
| U-Net | Semantic | Custom encoder-decoder |

## Transform Categories

### 1. Geometry Transforms
```python
# Resizing
Resize(size, interpolation='bilinear', antialias=True)
RandomResize(min_size, max_size)
RandomShortestSize(min_size, max_size)

# Cropping
CenterCrop(size)
RandomCrop(size, padding=None, pad_if_needed=False)
RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3/4, 4/3))
FiveCrop(size)
TenCrop(size, vertical_flip=False)

# Flipping
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)

# Rotation & Affine
RandomRotation(degrees, interpolation='nearest', expand=False)
RandomAffine(degrees, translate=None, scale=None, shear=None)
RandomPerspective(distortion_scale=0.5, p=0.5)
ElasticTransform(alpha=50.0, sigma=5.0)

# Padding
Pad(padding, fill=0, padding_mode='constant')
RandomZoomOut(fill=0, side_range=(1.0, 4.0), p=0.5)
```

### 2. Color Transforms
```python
# Basic
Grayscale(num_output_channels=1)
RandomGrayscale(p=0.1)
RGB()

# Color adjustments
ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
RandomPhotometricDistort(brightness, contrast, saturation, hue, p=0.5)

# Pixel operations
RandomInvert(p=0.5)
RandomPosterize(bits, p=0.5)
RandomSolarize(threshold, p=0.5)
RandomAdjustSharpness(sharpness_factor, p=0.5)
RandomAutocontrast(p=0.5)
RandomEqualize(p=0.5)

# Noise & Blur
GaussianBlur(kernel_size, sigma=(0.1, 2.0))
GaussianNoise(mean=0.0, sigma=0.1)
```

### 3. Composition
```python
Compose([transform1, transform2, ...])
RandomApply(transforms, p=0.5)
RandomChoice(transforms)
RandomOrder(transforms)
```

### 4. Auto Augmentation
```python
AutoAugment(policy='imagenet')  # 'imagenet', 'cifar10', 'svhn'
RandAugment(num_ops=2, magnitude=9)
TrivialAugmentWide(num_magnitude_bins=31)
AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)
```

### 5. Mixing Augmentations
```python
CutMix(alpha=1.0)
MixUp(alpha=0.2)
RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
```

### 6. Normalization
```python
Normalize(mean, std, inplace=False)
ToDtype(dtype, scale=False)
```

## Implementation Priority

### Phase 1: Core Classification (implement first)
1. `resnet.py` - Most important backbone
2. `vgg.py` - Simple, good for teaching
3. `alexnet.py` - Historical importance
4. Geometry transforms: Resize, Crop, Flip
5. Normalize transform

### Phase 2: Modern Architectures
1. `mobilenet.py` - Efficient inference
2. `efficientnet.py` - SOTA efficiency
3. `densenet.py` - Dense connections
4. Color transforms: ColorJitter, Grayscale

### Phase 3: Transformers
1. `vision_transformer.py` - ViT
2. `swin_transformer.py` - Hierarchical ViT
3. `convnext.py` - Modern ConvNet

### Phase 4: Detection & Segmentation
1. `segmentation/fcn.py`
2. `segmentation/unet.py`
3. `detection/faster_rcnn.py`
4. Box utilities in `ops/`

### Phase 5: Advanced
1. Auto augment policies
2. CutMix, MixUp
3. Video models
4. Deformable convolutions

## Example Usage

```python
from python.vision import models, transforms

# Classification
model = models.resnet50(num_classes=1000)
model = models.vit_b_16(num_classes=1000)

# Transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Detection
model = models.detection.fasterrcnn_resnet50_fpn(num_classes=91)

# Segmentation
model = models.segmentation.fcn_resnet50(num_classes=21)
```

## Notes

- All models should work with our `Tensor` and `Module` classes
- Transforms operate on numpy arrays by default
- Weight loading will use numpy `.npz` files (not PyTorch `.pth`)
- Focus on clean, readable implementations over optimization
