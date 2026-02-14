"""
Consolidated Normalization Module

This module contains all normalization layers for deep learning:
- BatchNorm1d, BatchNorm2d: Batch normalization for various input shapes
- LayerNorm, LayerNormTransformer: Layer normalization (essential for Transformers)
- GroupNorm, InstanceNorm: Group and instance normalization
- RMSNorm, RMSNormTransformer: Root Mean Square normalization
- SpectralNorm, SpectralNormConv2d, SpectralNormLinear: Weight normalization for GANs

Each module includes comprehensive documentation and implementation hints.
"""

from python.foundations import Tensor, convert_to_function
from typing import Optional, Tuple, Union, List
import numpy as np
from .module import Module, Parameter
from . import normalization_functional

# ============================================================================
# BATCH NORMALIZATION
# ============================================================================

class BatchNorm1d(Module):
    """
    Batch Normalization for 1D inputs.

    Shapes:
    - Input: (N, C) or (N, C, L)
    - Output: same as input

    Parameters:
    -----------
    num_features: int
        Number of features/channels (C)
    eps: float
        Small constant for numerical stability
    momentum: float
        Momentum for exponential moving average of running statistics
    affine: bool
        Whether to learn γ and β parameters
    track_running_stats: bool
        Whether to track running mean and variance
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        if not self.affine:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

        if self.track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features))
            self.register_buffer('running_var', np.ones(num_features))
            self.register_buffer('num_batches_tracked', 0)
        self.batch_norm1d = convert_to_function(normalization_functional.BatchNorm1d)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of Batch Normalization.

        Args:
            x: Input Tensor of shape (N, C) or (N, C, L)

        Returns:
            y: Normalized output of same shape as input
            updates running mean and variance in place

        Implementation Notes:
        - If training=True: Uses batch statistics
        - If training=False: Uses running statistics
        - Stores intermediate values for backward pass via autograd
        """
        return self.batch_norm1d(x, self.weight, self.bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps)


    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}"


class BatchNorm2d(Module):
    """
    Batch Normalization for 2D inputs (images).

    Shapes:
    - Input: (N, C, H, W)
    - Output: (N, C, H, W)

    Normalization is performed over batch and spatial dimensions (0, 2, 3),
    with separate parameters for each channel.

    Parameters:
    -----------
    num_features: int
        Number of channels (C)
    eps: float
        Small constant for numerical stability
    momentum: float
        Momentum for exponential moving average of running statistics
    affine: bool
        Whether to learn γ and β parameters
    track_running_stats: bool
        Whether to track running mean and variance
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        if not self.affine:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

        if self.track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features, dtype=np.float32))
            self.register_buffer('running_var', np.ones(num_features, dtype=np.float32))
            self.register_buffer('num_batches_tracked', 0)
        self.batch_norm2d = convert_to_function(normalization_functional.BatchNorm2d)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of BatchNorm2d.

        Args:
            x: Input Tensor of shape (N, C, H, W)

        Returns:
            y: Normalized output of shape (N, C, H, W)
            updates running mean and variance in place

        Key Points:
        - Normalize over dimensions (0, 2, 3), keeping dimension 1 (channels)
        - Resulting mean/var shape: (C,)
        - Apply learnable affine transform with reshaping for broadcasting
        """
        return self.batch_norm2d(x, gamma=self.weight, beta=self.bias, running_mean=self.running_mean, running_var=self.running_var, training=self.training, momentum=self.momentum, eps=self.eps)

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}"


BatchNorm = BatchNorm1d


# ============================================================================
# LAYER NORMALIZATION
# ============================================================================

class LayerNorm(Module):
    """
    Layer Normalization.

    Normalizes over the feature dimensions (last d dimensions) independently
    for each sample and position.

    Shapes:
    - Input: (*, normalized_shape)
    - Output: (*, normalized_shape)

    Parameters:
    -----------
    normalized_shape: int or tuple of ints
        Shape of the features to normalize over

    eps: float
        Small constant for numerical stability (default: 1e-5)

    elementwise_affine: bool
        Whether to learn affine parameters γ and β (default: True)

    Learnable Parameters:
    --------------------
    weight (γ): shape = normalized_shape, initialized to ones
    bias (β): shape = normalized_shape, initialized to zeros
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        elif not isinstance(normalized_shape, (list, tuple)):
            raise TypeError(
                f"normalized_shape should be int or list, got {type(normalized_shape)}"
            )

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = Parameter(np.ones(self.normalized_shape))
        self.bias = Parameter(np.zeros(self.normalized_shape))
        if not self.elementwise_affine:
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        self.layer_norm = convert_to_function(normalization_functional.LayerNorm)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of LayerNorm.

        Args:
            x: Input Tensor of shape (..., *normalized_shape)

        Returns:
            y: Normalized output of same shape as input

        Algorithm:
        1. Verify input shape ends with normalized_shape
        2. Compute mean over normalized dimensions
        3. Compute variance over normalized dimensions
        4. Normalize: x̂ = (x - mean) / sqrt(var + eps)
        5. Apply affine: y = γ * x̂ + β
        """
        return self.layer_norm(x, gamma=self.weight, beta=self.bias, normalized_shape=self.normalized_shape, eps=self.eps)

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class LayerNormTransformer(Module):
    """
    Specialized LayerNorm for Transformer use cases.

    This is identical to LayerNorm but documented for the specific context
    of Transformer architectures where it's essential for stability.
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
    ):
        """Initialize LayerNorm for Transformer use."""
        super().__init__()
        self.layernorm = LayerNorm(normalized_shape, eps=eps, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply LayerNorm in Transformer context."""
        return self.layernorm.forward(x)

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"normalized_shape={self.layernorm.normalized_shape}, eps={self.layernorm.eps}, elementwise_affine={self.layernorm.elementwise_affine}"


# ============================================================================
# GROUP NORMALIZATION
# ============================================================================

class GroupNorm(Module):
    """
    Group Normalization.

    Divides channels into groups and normalizes within each group independently.
    Works well with small batch sizes and is batch-size independent.

    Shapes:
    - Input: (N, C, *) where * can be any spatial dimensions
    - Output: (N, C, *)

    Parameters:
    -----------
    num_groups: int
        Number of groups to divide channels into
        Constraint: num_channels % num_groups == 0

    num_channels: int
        Total number of channels (C)

    eps: float
        Small constant for numerical stability (default: 1e-5)

    affine: bool
        Whether to learn γ and β parameters (default: True)

    Learnable Parameters:
    --------------------
    weight (γ): shape (num_channels,)
    bias (β): shape (num_channels,)
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine


        self.weight = Parameter(np.ones(num_channels))
        self.bias = Parameter(np.zeros(num_channels))

        if not self.affine:
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        self.group_norm = convert_to_function(normalization_functional.GroupNorm)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of GroupNorm.

        Args:
            x: Input Tensor of shape (N, C, ...) where C = num_channels

        Returns:
            y: Normalized output of same shape as input

        Algorithm:
        1. Get original shape (N, C, *spatial_dims)
        2. Reshape to (N, G, C//G, *spatial_dims) to make groups explicit
        3. Compute mean over all dimensions except N and G
        4. Compute variance similarly
        5. Normalize within each group
        6. Reshape back to (N, C, *spatial_dims)
        7. Apply per-channel affine transformation
        """
        return self.group_norm(x, gamma=self.weight, beta=self.bias, num_groups=self.num_groups, eps=self.eps)

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"num_groups={self.num_groups}, num_channels={self.num_channels}, eps={self.eps}, affine={self.affine}"


class InstanceNorm(Module):
    """
    Instance Normalization (special case: GroupNorm with num_groups=num_channels).

    Each channel is normalized independently over spatial dimensions.
    Commonly used in style transfer and generative models.

    Shapes:
    - Input: (N, C, H, W)
    - Output: (N, C, H, W)

    This is equivalent to GroupNorm(num_channels, num_channels)
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Initialize InstanceNorm (GroupNorm with one channel per group)."""
        super().__init__()
        self.groupnorm = GroupNorm(
            num_groups=num_channels,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply InstanceNorm (equivalent to per-channel normalization)."""
        return self.groupnorm.forward(x)

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"num_channels={self.groupnorm.num_channels}, eps={self.groupnorm.eps}, affine={self.groupnorm.affine}"


# ============================================================================
# RMS NORMALIZATION
# ============================================================================

class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster alternative to LayerNorm that only uses RMS scaling
    without centering. Used extensively in modern large language models
    like LLaMA, PaLM, and other state-of-the-art architectures.

    Shapes:
    - Input: (*, normalized_shape)
    - Output: (*, normalized_shape)

    Parameters:
    -----------
    normalized_shape: int or tuple of ints
        Shape of the feature dimension(s) to normalize over

    eps: float
        Small constant for numerical stability (default: 1e-6)

    Learnable Parameters:
    --------------------
    weight (γ): shape = normalized_shape, initialized to ones
    Note: Unlike LayerNorm, no bias parameter (β)

    Key Properties:
    - No train/eval distinction
    - No running statistics
    - Identical behavior in training and inference
    - Simpler than LayerNorm (no mean centering)
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-6,
    ):
        """
        Initialize RMSNorm.

        Args:
            normalized_shape: int or tuple, number of features to normalize
            eps: float, small constant for stability
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        elif not isinstance(normalized_shape, tuple):
            raise TypeError(f"normalized_shape should be int or tuple, got {type(normalized_shape)}")

        if any(s <= 0 for s in normalized_shape):
            raise ValueError(f"normalized_shape must be all positive, got {normalized_shape}")

        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = Parameter(np.ones(normalized_shape))
        self.rms_norm = convert_to_function(normalization_functional.RMSNorm)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x: Input Tensor of shape (..., normalized_shape)

        Returns:
            y: Normalized output of same shape as input

        Algorithm:
        1. Verify last dimension matches normalized_shape
        2. Compute mean of squares over last dimension: mean(x²)
        3. Compute RMS: sqrt(mean(x²))
        4. Add epsilon for numerical stability
        5. Normalize: x̂ = x / (RMS + ε)
        6. Scale by learned weight: y = γ * x̂

        Implementation Notes:
        - DO NOT subtract mean before normalizing (key difference from LayerNorm)
        - Compute mean(x²) carefully to avoid numerical overflow
        """
        return self.rms_norm(x, gamma=self.weight, normalized_shape=self.normalized_shape, eps=self.eps)


class RMSNormTransformer(Module):
    """
    Specialized RMSNorm for Transformer use cases.

    This is identical to RMSNorm but documented for Transformer architectures,
    which are the primary use case for RMSNorm in modern deep learning.

    In LLaMA and other modern LLMs, RMSNorm is applied:
    - Before attention layers
    - Before feed-forward layers
    - Using Pre-LN configuration for stability
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-6,
    ):
        """Initialize RMSNorm for Transformer use."""
        super().__init__()
        self.rmsnorm = RMSNorm(normalized_shape, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm in Transformer context."""
        return self.rmsnorm.forward(x)

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"normalized_shape={self.rmsnorm.normalized_shape}, eps={self.rmsnorm.eps}"


# ============================================================================
# SPECTRAL NORMALIZATION
# ============================================================================

class SpectralNorm(Module):
    """
    Spectral Normalization for GAN discriminators.

    Normalizes the weight matrix W to have spectral norm (largest singular value)
    equal to 1, constraining the Lipschitz constant of the layer.

    This is a wrapper applied to weight matrices in discriminator layers.

    Parameters:
    -----------
    weight: np.ndarray
        Weight matrix to normalize, shape (out_features, in_features)

    n_power_iterations: int
        Number of power iteration steps per forward pass (default: 1)

    eps: float
        Small constant for numerical stability (default: 1e-12)

    Key Properties:
    - Applied to weight matrices (not activations)
    - Makes discriminator layer 1-Lipschitz
    - Stabilizes GAN training
    - Minimal computational overhead
    - Works with any optimizer
    """

    def __init__(
        self,
        weight: np.ndarray,
        n_power_iterations: int = 1,
        eps: float = 1e-12,
    ):
        """
        Initialize Spectral Normalization.

        Args:
            weight: Weight matrix to normalize
            n_power_iterations: Number of power iterations (usually 1-2)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = weight
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        if weight.ndim == 1:
            self.height = 1
            self.width = weight.shape[0]
        else:
            self.height = weight.shape[0]
            self.width = np.prod(weight.shape[1:])

        self.register_buffer('u', np.random.normal(0, 1, (self.height,)))

    def forward(self) -> np.ndarray:
        """
        Apply spectral normalization to weight matrix.

        Returns:
            weight_normalized: Weight matrix normalized by spectral norm

        Algorithm:
        1. Reshape weight to 2D: (height, width)
        2. Power iteration (1-2 times):
           a. v = W^T u / ||W^T u||
           b. u = W v / ||W v||
        3. Compute spectral norm: sigma = u^T W v
        4. Normalize: W_normalized = W / sigma
        5. Update u for next iteration
        6. Return normalized weight

        Implementation Notes:
        - Reshape weight to 2D before power iteration
        - Keep weight 2D during computation
        - Reshape back to original shape before returning
        - Update u in-place for gradient tracking
        """
        raise NotImplementedError(
            "Implement spectral normalization forward:\n"
            "1. Reshape weight to 2D (height, width)\n"
            "2. For i in range(n_power_iterations):\n"
            "   a. v = W^T @ u\n"
            "   b. v = v / (||v|| + eps)    # normalize v\n"
            "   c. u = W @ v\n"
            "   d. u = u / (||u|| + eps)    # normalize u\n"
            "3. Compute spectral norm:\n"
            "   sigma = u @ W @ v\n"
            "   OR sigma = u @ (W @ v)    # more numerically stable\n"
            "4. Normalize weight:\n"
            "   W_normalized = W / sigma\n"
            "5. Store u for next iteration:\n"
            "   self.u = u  # update buffer\n"
            "6. Reshape W_normalized back to original shape\n"
            "7. Return W_normalized"
        )

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"height={self.height}, width={self.width}, n_power_iterations={self.n_power_iterations}, eps={self.eps}"


class SpectralNormConv2d(Module):
    """
    Spectral Normalization wrapper for 2D Convolution.

    Applies spectral normalization to the convolutional layer's weight matrix.

    Shape:
    - Conv2d weight shape: (out_channels, in_channels, kernel_h, kernel_w)
    - Reshaped for SpectralNorm: (out_channels, in_channels * kernel_h * kernel_w)

    Parameters:
    -----------
    conv_weight: np.ndarray
        Convolution weight of shape (out_channels, in_channels, kernel_h, kernel_w)

    n_power_iterations: int
        Number of power iterations (default: 1)
    """

    def __init__(
        self,
        conv_weight: np.ndarray,
        n_power_iterations: int = 1,
    ):
        """
        Initialize Spectral Norm for Conv2d layer.

        Args:
            conv_weight: Conv2d weight, shape (out_channels, in_channels, kh, kw)
            n_power_iterations: Power iterations per forward pass
        """
        super().__init__()
        self.original_shape = conv_weight.shape
        self.out_channels = conv_weight.shape[0]

        reshaped_weight = conv_weight.reshape(conv_weight.shape[0], -1)
        self.spec_norm = SpectralNorm(reshaped_weight, n_power_iterations)

    def forward(self) -> np.ndarray:
        """
        Get normalized convolution weight.

        Returns:
            weight_normalized: Shape (out_channels, in_channels, kernel_h, kernel_w)

        Implementation:
        1. Get normalized weight from spectral norm object
           (shape: (out_channels, in_channels * kh * kw))
        2. Reshape back to original shape:
           (out_channels, in_channels, kernel_h, kernel_w)
        3. Return reshaped normalized weight
        """
        raise NotImplementedError(
            "Implement spectral norm for conv2d:\n"
            "1. Get normalized weight from spectral norm object\n"
            "   (shape: (out_channels, in_channels * kh * kw))\n"
            "2. Reshape back to original shape:\n"
            "   (out_channels, in_channels, kernel_h, kernel_w)\n"
            "3. Return reshaped normalized weight"
        )

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"original_shape={self.original_shape}, out_channels={self.out_channels}"


class SpectralNormLinear(Module):
    """
    Spectral Normalization wrapper for Linear layer.

    Applies spectral normalization to linear layer weight matrix.

    Shape:
    - Linear weight shape: (out_features, in_features)
    - This is already the right shape for spectral norm

    Parameters:
    -----------
    linear_weight: np.ndarray
        Linear weight, shape (out_features, in_features)

    n_power_iterations: int
        Power iterations per forward pass (default: 1)
    """

    def __init__(
        self,
        linear_weight: np.ndarray,
        n_power_iterations: int = 1,
    ):
        """
        Initialize Spectral Norm for Linear layer.

        Args:
            linear_weight: Linear weight, shape (out_features, in_features)
            n_power_iterations: Power iterations per forward pass
        """
        super().__init__()
        self.spec_norm = SpectralNorm(linear_weight, n_power_iterations)

    def forward(self) -> np.ndarray:
        """
        Get normalized linear weight.

        Returns:
            weight_normalized: Shape (out_features, in_features)
        """
        return self.spec_norm.forward()

    def extra_repr(self) -> str:
        """Extra info for __repr__."""
        return f"height={self.spec_norm.height}, width={self.spec_norm.width}"


# =============================================================================
# Additional Normalization Layers
# =============================================================================

class BatchNorm3d(Module):
    """
    3D Batch Normalization for volumetric data.

    Input shape: (N, C, D, H, W)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features))
            self.register_buffer('running_var', np.ones(num_features))
            self.register_buffer('num_batches_tracked', np.array(0))

    def forward(self, x):
        raise NotImplementedError("TODO: Implement BatchNorm3d forward")


class InstanceNorm1d(Module):
    """Instance Normalization for 1D inputs (sequences)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))

    def forward(self, x):
        raise NotImplementedError("TODO: Implement InstanceNorm1d forward")


class InstanceNorm2d(Module):
    """Instance Normalization for 2D inputs (images)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))

    def forward(self, x):
        raise NotImplementedError("TODO: Implement InstanceNorm2d forward")


class InstanceNorm3d(Module):
    """Instance Normalization for 3D inputs (volumes)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))

    def forward(self, x):
        raise NotImplementedError("TODO: Implement InstanceNorm3d forward")


class LocalResponseNorm(Module):
    """
    Local Response Normalization (LRN).

    Used in AlexNet. Normalizes across nearby channels.
    """

    def __init__(
        self,
        size: int,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0
    ):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        raise NotImplementedError("TODO: Implement LocalResponseNorm forward")
