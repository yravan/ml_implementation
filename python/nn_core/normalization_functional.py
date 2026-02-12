"""
Normalization Functional Operations
====================================

This module provides functional operations for normalization layers.
Function classes handle the forward/backward computation with np.ndarray,
while Module classes in normalization.py wrap these for Tensor operations.

Function Classes:
    - BatchNorm1d: 1D batch normalization functional
    - BatchNorm2d: 2D batch normalization functional
    - LayerNorm: Layer normalization functional
    - GroupNorm: Group normalization functional
    - InstanceNorm2d: Instance normalization functional
    - RMSNorm: Root mean square normalization functional

Helper Functions:
    - batch_norm, layer_norm, group_norm, instance_norm, rms_norm
"""
from math import prod

import numpy as np
from typing import Tuple, Optional, Union

from python.foundations import Function, convert_to_function, _no_grad


# =============================================================================
# Batch Normalization Function Classes
# =============================================================================

class BatchNorm1d(Function):
    """
    Batch Normalization 1D functional operation.

    Normalizes over batch and (optionally) spatial dimensions for 2D/3D input.

    Math:
        y = (x - mean) / sqrt(var + eps) * gamma + beta

    Running statistics are updated during training:
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var
    """

    def forward(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: Optional[np.ndarray] = None,
        running_var: Optional[np.ndarray] = None,
        training: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5
    ) -> np.ndarray:
        """
        Compute batch normalization.

        Args:
            x: Input (batch_size, num_features) or (batch_size, num_features, length)
            gamma: Scale parameter (num_features,)
            beta: Shift parameter (num_features,)
            running_mean: Running mean for inference
            running_var: Running variance for inference
            training: Whether in training mode
            momentum: Momentum for running stats update
            eps: Small constant for numerical stability

        Returns:
            output
            updates running mean and variance in place
        """
        if training:
            if x.ndim == 2:
                mean = x.mean(axis=0)
                var = x.var(axis=0)
                norm_x = (x - mean[None,:]) / np.sqrt(var[None,:] + eps)
                output = norm_x * gamma[None,:] + beta[None,:]
            elif x.ndim == 3:
                mean = x.mean(axis=(0,2))
                var = x.var(axis=(0,2))
                norm_x = (x - mean[None,:,None]) / np.sqrt(var[None,:,None] + eps)
                output = norm_x * gamma[None,:,None] + beta[None,:,None]
            else:
                raise RuntimeError("Invalid input for BatchNorm1d", x.shape)

            if running_mean is not None:
                running_mean *= momentum
                running_mean += (1 - momentum) * mean
            else:
                running_mean = mean.copy()
            if running_var is not None:
                running_var *= momentum
                running_var += (1 - momentum) * var
            else:
                running_var = var.copy()
        else:
            if x.ndim == 2:
                norm_x = (x - running_mean[None,:]) / np.sqrt(running_var[None,:] + eps)
                output = norm_x * gamma[None,:] + beta[None,:]
            elif x.ndim == 3:
                norm_x = (x - running_mean[None,:,None]) / np.sqrt(running_var[None,:,None] + eps)
                output = norm_x * gamma[None,:,None] + beta[None,:,None]
            else:
                raise RuntimeError("Invalid input for BatchNorm1d", x.shape)

        global _no_grad
        if not _no_grad:
            self.norm_x = norm_x
            self.gamma = gamma
            self.beta = beta
            if training:
                self.var = var
                self.mean = mean
            else:
                self.mean = running_mean
                self.var = running_var
            self.eps = eps

        return output



    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for batch normalization.

        Returns:
            Tuple of (grad_x, grad_gamma, grad_beta)
        """
        if grad_output.ndim == 2:
            grad_beta = grad_output.sum(axis=0)
            grad_gamma = (self.norm_x * grad_output).sum(axis=0)
            N = grad_output.shape[0]
            grad_x = (self.gamma[None,:] / (N * np.sqrt(self.var[None,:] + self.eps))) * (
                N * grad_output - grad_output.sum(axis=0,  keepdims=True) - self.norm_x * (grad_output * self.norm_x).sum(axis=0, keepdims=True)
            )
        elif grad_output.ndim == 3:
            grad_beta = grad_output.sum(axis=(0,2))
            grad_gamma = (self.norm_x * grad_output).sum(axis=(0,2))
            N = grad_output.shape[0] * grad_output.shape[2]
            grad_x = (self.gamma[None,:, None] / (N * np.sqrt(self.var[None,:, None] + self.eps))) * (
                N * grad_output - grad_output.sum(axis=(0,2), keepdims=True) - self.norm_x * (grad_output * self.norm_x).sum(axis=(0,2), keepdims=True)
            )
        else:
            raise RuntimeError("Invalid gradient for BatchNorm1d", grad_output.shape)

        return grad_x, grad_gamma, grad_beta



class BatchNorm2d(Function):
    """
    Batch Normalization 2D functional operation.

    Normalizes over batch and spatial dimensions for 4D input (images).

    Math:
        y = (x - mean) / sqrt(var + eps) * gamma + beta

    Statistics computed per channel over (N, H, W) dimensions.
    """

    def forward(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: Optional[np.ndarray] = None,
        running_var: Optional[np.ndarray] = None,
        training: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D batch normalization.

        Args:
            x: Input (batch_size, channels, height, width)
            gamma: Scale parameter (channels,)
            beta: Shift parameter (channels,)
            running_mean: Running mean for inference
            running_var: Running variance for inference
            training: Whether in training mode
            momentum: Momentum for running stats update
            eps: Small constant for numerical stability

        Returns:
            output
            updates running mean and variance in place
        """
        if not x.ndim == 4:
            raise RuntimeError("Invalid input for BatchNorm2d", x.shape)

        if training:
            mean = x.mean(axis=(0,2,3))
            var = x.var(axis=(0,2,3))
            norm_x = (x - mean[None,:,None, None]) / np.sqrt(var[None,:,None,None] + eps)
            output = norm_x * gamma[None,:, None, None] + beta[None,:,None,None]
            if running_mean is not None:
                running_mean *= momentum
                running_mean += (1 - momentum) * mean
            else:
                running_mean = mean.copy()
            if running_var is not None:
                running_var *= momentum
                running_var += (1 - momentum) * var
            else:
                running_var = var.copy()
        else:
            norm_x = (x - running_mean[None,:,None, None]) / np.sqrt(running_var[None,:,None,None] + eps)
            output = norm_x * gamma[None,:, None, None] + beta[None,:,None,None]

        global _no_grad
        if not _no_grad:
            if training:
                self.mean = mean
                self.var = var
            else:
                self.mean = running_mean
                self.var = running_var
            self.norm_x = norm_x
            self.gamma = gamma
            self.beta = beta
            self.eps = eps

        return output


    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for 2D batch normalization.

        Returns:
            Tuple of (grad_x, grad_gamma, grad_beta)
        """
        if grad_output.ndim == 4:
            grad_beta = grad_output.sum(axis=(0,2,3))
            grad_gamma = (self.norm_x * grad_output).sum(axis=(0,2,3))
            N = grad_output.shape[0] * grad_output.shape[2] * grad_output.shape[3]
            grad_x = (self.gamma[None,:, None, None] / (N * np.sqrt(self.var[None,:, None, None] + self.eps))) * (
                N * grad_output - grad_output.sum(axis=(0,2,3),  keepdims=True) - self.norm_x * (grad_output * self.norm_x).sum(axis=(0,2,3), keepdims=True)
            )
        else:
            raise RuntimeError("Invalid gradient for BatchNorm1d", grad_output.shape)
        return grad_x, grad_gamma, grad_beta


# =============================================================================
# Layer Normalization Function Class
# =============================================================================

class LayerNorm(Function):
    """
    Layer Normalization functional operation.

    Normalizes over the last D dimensions (typically feature dimensions).
    Unlike BatchNorm, LayerNorm normalizes each sample independently.

    Math:
        y = (x - mean) / sqrt(var + eps) * gamma + beta

    Mean and variance computed over normalized_shape dimensions.
    """

    def forward(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-5
    ) -> np.ndarray:
        """
        Compute layer normalization.

        Args:
            x: Input of any shape
            gamma: Scale parameter (normalized_shape)
            beta: Shift parameter (normalized_shape)
            normalized_shape: Shape of dimensions to normalize over
            eps: Small constant for numerical stability

        Returns:
            Normalized output (same shape as input)
        """
        normalized_shape = tuple(normalized_shape)
        D = len(normalized_shape)
        if x.shape[x.ndim - D :] != normalized_shape:
            raise RuntimeError("Invalid normalization shape for LayerNorm", x.shape, normalized_shape)
        axes = tuple(range(x.ndim - D, x.ndim))

        mean = x.mean(axis=tuple(axes),keepdims=True)
        var = x.var(axis=tuple(axes),keepdims=True)
        norm_x = (x - mean) / np.sqrt(var + eps)
        output = norm_x * gamma + beta # broadcasting automatically works bc it's over the last dimension

        global _no_grad
        if not _no_grad:
            self.mean = mean
            self.var = var
            self.norm_x = norm_x
            self.gamma = gamma
            self.beta = beta
            self.eps = eps
            self.axes = axes

        return output



    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for layer normalization.

        Returns:
            Tuple of (grad_x, grad_gamma, grad_beta)
        """
        sum_axes = tuple(i for i in range(grad_output.ndim) if i not in self.axes)
        grad_beta = grad_output.sum(axis=sum_axes)
        grad_gamma = (self.norm_x * grad_output).sum(axis=sum_axes)
        N = prod(grad_output.shape[i] for i in self.axes)
        grad_x = (self.gamma / (N * np.sqrt(self.var + self.eps))) * (
            N * grad_output - grad_output.sum(axis=self.axes,  keepdims=True) - self.norm_x * (grad_output * self.norm_x).sum(axis=self.axes, keepdims=True)
        )
        return grad_x, grad_gamma, grad_beta


# =============================================================================
# Group Normalization Function Class
# =============================================================================

class GroupNorm(Function):
    """
    Group Normalization functional operation.

    Divides channels into groups and normalizes within each group.
    Combines benefits of LayerNorm (batch-independent) and BatchNorm (channel-wise).

    Math:
        y = (x - mean) / sqrt(var + eps) * gamma + beta

    Statistics computed per group over (C/G, H, W) dimensions.
    """

    def forward(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        num_groups: int,
        eps: float = 1e-5
    ) -> np.ndarray:
        """
        Compute group normalization.

        Args:
            x: Input (batch_size, channels, *spatial_dims)
            gamma: Scale parameter (channels,)
            beta: Shift parameter (channels,)
            num_groups: Number of groups to divide channels into
            eps: Small constant for numerical stability

        Returns:
            Normalized output (same shape as input)
        """
        B, C, *spatial_dims = x.shape
        if C % num_groups != 0:
            raise RuntimeError("Invalid number of groups", x.shape, num_groups)
        x_grouped = x.reshape((B, num_groups, C//num_groups, *spatial_dims))
        mean_grouped = x_grouped.mean(axis=tuple(range(2,x_grouped.ndim)), keepdims=True)
        var_grouped = x_grouped.var(axis=tuple(range(2,x_grouped.ndim)), keepdims=True)

        x_norm_grouped = (x_grouped - mean_grouped) / np.sqrt(var_grouped + eps)
        x_norm = x_norm_grouped.reshape((B, C, *spatial_dims))

        output = x_norm * gamma.reshape((1, C, *tuple(1 for _ in range(2,x.ndim)))) + beta.reshape((1, C, *tuple(1 for _ in range(2,x.ndim))))

        global _no_grad
        if not _no_grad:
            self.mean_grouped = mean_grouped
            self.var_grouped = var_grouped
            self.x_norm_grouped = x_norm_grouped
            self.eps = eps
            self.gamma = gamma.reshape((1, C, *tuple(1 for _ in range(2,x.ndim))))
            self.beta = beta.reshape((1, C, *tuple(1 for _ in range(2,x.ndim))))

        return output


    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for group normalization.

        Returns:
            Tuple of (grad_x, grad_gamma, grad_beta)
        """
        B, C, *spatial_dims = grad_output.shape
        num_groups = self.x_norm_grouped.shape[1]
        x_norm = self.x_norm_grouped.reshape((B, C, *spatial_dims))
        sum_axes = (0,) + tuple(range(2, grad_output.ndim))
        grad_beta = grad_output.sum(axis=sum_axes)
        grad_gamma = (x_norm * grad_output).sum(axis=sum_axes)

        grad_output_grouped = grad_output.reshape((B, num_groups, C//num_groups, *spatial_dims))
        gamma_grouped = self.gamma.reshape((1, num_groups, C // num_groups, *self.gamma.shape[2:]))
        beta_grouped = self.beta.reshape((1, num_groups, C // num_groups, *self.beta.shape[2:]))

        N = prod(grad_output_grouped.shape[i] for i in range(2, grad_output_grouped.ndim))
        grad_x_grouped = (gamma_grouped / (N * np.sqrt(self.var_grouped + self.eps))) * (
            N * grad_output_grouped - grad_output_grouped.sum(axis=tuple(range(2,grad_output_grouped.ndim)),  keepdims=True) - self.x_norm_grouped * (grad_output_grouped * self.x_norm_grouped).sum(axis=tuple(range(2,grad_output_grouped.ndim)), keepdims=True)
        )

        grad_x = grad_x_grouped.reshape((B, C, *spatial_dims))
        return grad_x, grad_gamma, grad_beta


# =============================================================================
# RMS Normalization Function Class
# =============================================================================

class RMSNorm(Function):
    """
    Root Mean Square Normalization functional operation.

    A simplified normalization that doesn't center the data (no mean subtraction).
    Used in models like LLaMA.

    Math:
        y = x / sqrt(mean(x^2) + eps) * gamma
    """

    def forward(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-6
    ) -> np.ndarray:
        """
        Compute RMS normalization.

        Args:
            x: Input of any shape
            gamma: Scale parameter (last dimension size)
            eps: Small constant for numerical stability

        Returns:
            Normalized output (same shape as input)
        """
        normalized_shape = tuple(normalized_shape)
        D = len(normalized_shape)
        if x.shape[x.ndim - D :] != normalized_shape:
            raise RuntimeError("Invalid normalization shape for RMSNorm", x.shape, normalized_shape)
        axes = tuple(range(x.ndim - D, x.ndim))
        rms = np.sqrt((x**2).mean(axis=axes, keepdims=True) + eps)
        norm_x = x / rms
        output = norm_x * gamma
        global _no_grad
        if not _no_grad:
            self.norm_x = norm_x
            self.rms = rms
            self.gamma = gamma
            self.axes = axes
        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for RMS normalization.

        Returns:
            Tuple of (grad_x, grad_gamma)
        """
        sum_axes = tuple(i for i in range(grad_output.ndim) if i not in self.axes)
        grad_gamma = grad_output.sum(axis=sum_axes)

        g = grad_output * self.gamma
        grad_x = (1 / self.rms) * (
            g - self.norm_x * (g * self.norm_x).mean(axis=-1, keepdims=True)
        )

        return grad_x, grad_gamma



class SpectralNorm(Function):
    """
    Root Mean Square Normalization functional operation.

    A simplified normalization that doesn't center the data (no mean subtraction).
    Used in models like LLaMA.

    Math:
        y = x / sqrt(mean(x^2) + eps) * gamma
    """

    def forward(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-6
    ) -> np.ndarray:
        """
        Compute RMS normalization.

        Args:
            x: Input of any shape
            gamma: Scale parameter (last dimension size)
            eps: Small constant for numerical stability

        Returns:
            Normalized output (same shape as input)
        """
        normalized_shape = tuple(normalized_shape)
        D = len(normalized_shape)
        if x.shape[x.ndim - D :] != normalized_shape:
            raise RuntimeError("Invalid normalization shape for RMSNorm", x.shape, normalized_shape)
        axes = tuple(range(x.ndim - D, x.ndim))
        rms = np.sqrt((x**2).mean(axis=axes, keepdims=True) + eps)
        norm_x = x / rms
        output = norm_x * gamma
        global _no_grad
        if not _no_grad:
            self.norm_x = norm_x
            self.rms = rms
            self.gamma = gamma
            self.axes = axes
        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for RMS normalization.

        Returns:
            Tuple of (grad_x, grad_gamma)
        """
        sum_axes = tuple(i for i in range(grad_output.ndim) if i not in self.axes)
        grad_gamma = grad_output.sum(axis=sum_axes)

        g = grad_output * self.gamma
        grad_x = (1 / self.rms) * (
            g - self.norm_x * (g * self.norm_x).mean(axis=-1, keepdims=True)
        )

        return grad_x, grad_gamma


# =============================================================================
# Functional Interfaces
# =============================================================================

batch_norm_1d = convert_to_function(BatchNorm1d)
batch_norm_2d = convert_to_function(BatchNorm2d)
layer_norm = convert_to_function(LayerNorm)
group_norm = convert_to_function(GroupNorm)
rms_norm = convert_to_function(RMSNorm)
