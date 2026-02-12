"""
Optimization Module
===================

Comprehensive optimization toolkit for training neural networks.

This module provides:

1. **Optimizers** (optimizers.py)
   - SGD, SGDW: Stochastic Gradient Descent with momentum and weight decay
   - Adam, AdamW, NAdam, RAdam: Adaptive moment estimation variants
   - RMSprop, Adagrad, Adadelta: Classic adaptive methods
   - LAMB, LARS: Large-batch training optimizers
   - Lion, Muon: Novel optimizers

2. **Loss Functions** (losses.py)
   - MSE, MAE, Huber: Regression losses
   - CrossEntropy, BCE, NLL: Classification losses
   - Focal, LabelSmoothing: Handling class imbalance
   - Triplet, Contrastive, InfoNCE: Metric learning
   - KLDiv, Dice: Distribution and segmentation losses

3. **Learning Rate Schedulers** (schedulers.py)
   - StepLR, MultiStepLR: Step-based decay
   - CosineAnnealingLR, OneCycleLR: Cyclic schedules
   - WarmupLR, LinearLR: Warmup and linear decay
   - ReduceLROnPlateau: Adaptive scheduling
   - LRFinder: Learning rate range test

4. **Gradient Utilities** (gradient_utils.py)
   - clip_grad_norm_, clip_grad_value_: Gradient clipping
   - GradientAccumulator: Simulate larger batch sizes
   - GradScaler: Mixed-precision training support
   - GradientMonitor: Training analysis

Typical Usage:
    >>> from python.optimization import Adam, CrossEntropyLoss, CosineAnnealingLR
    >>>
    >>> # Setup optimizer and scheduler
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
    >>> loss_fn = CrossEntropyLoss()
    >>>
    >>> # Training loop
    >>> for epoch in range(100):
    ...     for batch in dataloader:
    ...         loss = loss_fn(model(batch.x), batch.y)
    ...         grads = compute_gradients(loss)
    ...         optimizer.step(grads)
    ...     scheduler.step()
"""

# Optimizers
from .optimizers import (
    # Base
    Optimizer,
    # SGD family
    SGD, SGDW,
    # RMSprop/Adagrad family
    RMSprop, Adagrad, Adadelta,
    # Adam family
    Adam, AdamW, NAdam, RAdam, Adafactor,
    # Large-batch
    LAMB, LARS,
    # Novel
    Lion, Muon,
)

# Loss Functions
from .losses import (
    # Regression
    MSELoss, MAELoss, HuberLoss, SmoothL1Loss, RMSELoss,
    # Classification
    CrossEntropyLoss, BinaryCrossEntropyLoss, BCEWithLogitsLoss, NLLLoss,
    FocalLoss,
    # Sequence
    CTCLoss,
    # Metric Learning
    TripletLoss, ContrastiveLoss, InfoNCELoss,
    # Distribution
    KLDivLoss, DiceLoss,
)
# Learning Rate Schedulers
from .schedulers import (
    # Base
    LRScheduler,
    # Step-based
    StepLR, MultiStepLR,
    # Continuous
    ExponentialLR, LinearLR, PolynomialLR,
    # Cosine
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    # Cyclic
    CyclicLR, OneCycleLR,
    # Adaptive
    ReduceLROnPlateau,
    # Warmup
    WarmupLR, WarmupCosineSchedule,
    # Composite
    SequentialLR, ChainedScheduler,
    # Utilities
    LRFinder,
    get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
)

# Gradient Utilities
from .gradient_utils import (
    # Clipping
    clip_grad_norm_, clip_grad_value_, GradientClipper,
    # Accumulation
    GradientAccumulator,
    # Scaling
    GradScaler,
    # Analysis
    compute_gradient_norm, compute_gradient_stats,
    detect_gradient_anomaly, GradientMonitor,
    # Regularization
    gradient_penalty, spectral_norm,
    # Fisher/Hessian
    compute_fisher_information, compute_hessian_vector_product,
    # Utilities
    flatten_gradients, unflatten_gradients,
    zero_gradients, scale_gradients, add_gradient_noise,
)

__all__ = [
    # Optimizers
    'Optimizer',
    'SGD', 'SGDW',
    'RMSprop', 'Adagrad', 'Adadelta',
    'Adam', 'AdamW', 'NAdam', 'RAdam', 'Adafactor',
    'LAMB', 'LARS',
    'Lion', 'Muon',
    'sgd_step', 'adam_step', 'adamw_step',

    # Losses
    'Loss',
    'MSELoss', 'MAELoss', 'HuberLoss', 'SmoothL1Loss', 'RMSELoss',
    'CrossEntropyLoss', 'BinaryCrossEntropyLoss', 'BCEWithLogitsLoss', 'NLLLoss',
    'FocalLoss',
    'CTCLoss',
    'TripletLoss', 'ContrastiveLoss', 'InfoNCELoss',
    'KLDivLoss', 'DiceLoss',
    'mse_loss', 'mae_loss', 'cross_entropy_loss', 'binary_cross_entropy_loss',
    'kl_div_loss', 'triplet_loss',
    'logsumexp', 'softmax', 'log_softmax',

    # Schedulers
    'LRScheduler',
    'StepLR', 'MultiStepLR',
    'ExponentialLR', 'LinearLR', 'PolynomialLR',
    'CosineAnnealingLR', 'CosineAnnealingWarmRestarts',
    'CyclicLR', 'OneCycleLR',
    'ReduceLROnPlateau',
    'WarmupLR', 'WarmupCosineSchedule',
    'SequentialLR', 'ChainedScheduler',
    'LRFinder',
    'get_cosine_schedule_with_warmup', 'get_linear_schedule_with_warmup',

    # Gradient Utils
    'clip_grad_norm_', 'clip_grad_value_', 'GradientClipper',
    'GradientAccumulator',
    'GradScaler',
    'compute_gradient_norm', 'compute_gradient_stats',
    'detect_gradient_anomaly', 'GradientMonitor',
    'gradient_penalty', 'spectral_norm',
    'compute_fisher_information', 'compute_hessian_vector_product',
    'flatten_gradients', 'unflatten_gradients',
    'zero_gradients', 'scale_gradients', 'add_gradient_noise',
]
