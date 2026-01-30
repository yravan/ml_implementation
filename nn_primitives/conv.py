"""
Convolutional layers: Conv2D and MaxPool2D with forward and backward passes.

Convention: NCHW format (batch, channels, height, width).

Conv2D forward (naive implementation):
    For each output position (i, j) and output channel c_out:
        Y[n, c_out, i, j] = Σ_{c_in} Σ_{kh} Σ_{kw}
            X[n, c_in, i*stride+kh, j*stride+kw] * W[c_out, c_in, kh, kw] + b[c_out]

Output dimensions:
    H_out = (H_in + 2*pad - kernel_h) // stride + 1
    W_out = (W_in + 2*pad - kernel_w) // stride + 1
"""

import numpy as np


def conv2d_forward(X, Weight, b, stride=1, pad=0):
    """
    Forward pass for 2D convolution.

    Parameters:
        X: np.ndarray of shape (N, C_in, H, W) - Input feature maps.
        W: np.ndarray of shape (C_out, C_in, kH, kW) - Convolution filters.
        b: np.ndarray of shape (C_out,) - Bias per output channel.
        stride: int - Stride of the convolution.
        pad: int - Zero-padding added to both sides of input.

    Returns:
        Y: np.ndarray of shape (N, C_out, H_out, W_out) - Output feature maps.
        cache: tuple (X_padded, W, b, stride, pad) - Stored for backward.
    """
    N, C_in, H, W = X.shape
    C_out, C_in, kH, kW = Weight.shape
    H_out = (H + 2 * pad - (kH)) // stride + 1
    W_out = (W + 2 * pad - (kW)) // stride + 1
    padded_image = np.zeros((N, C_in, H + 2 * pad, W + 2 * pad))
    padded_image[:, :, pad: H + pad, pad: W + pad] = X

    h_indices = np.arange(H_out)[:, None] * stride + np.arange(kH)[None, :] # H_out x kH
    h_indices = h_indices[:, None, :, None] # H_out x 1 x kH x 1
    w_indices = np.arange(W_out)[:, None] * stride + np.arange(kW)[None, :] # W_out x kW
    w_indices = w_indices[None, :, None, :] # 1 x W_out x 1 x kW

    patches = padded_image[:, :, h_indices, w_indices] # N x C_in x H_out x W_out x kH x kW
    filter = Weight.reshape((C_out, -1))
    patches = np.transpose(patches, (0, 2, 3, 1, 4, 5))
    patches = patches.reshape((N, H_out, W_out, -1))
    patches = patches @ filter.T + b[None, None, None, :]
    patches = np.transpose(patches, (0, 3, 1, 2))
    Y = patches
    cache = None
    return Y, cache


def conv2d_backward(d_out, cache):
    """
    Backward pass for 2D convolution.

    Computes gradients w.r.t. input, weights, and bias.

    Key formulas (for each element):
        ∂L/∂W[co, ci, kh, kw] = Σ_n Σ_i Σ_j d_out[n,co,i,j] * X_pad[n,ci,i*s+kh,j*s+kw]
        ∂L/∂b[co] = Σ_n Σ_i Σ_j d_out[n, co, i, j]
        ∂L/∂X_pad[n,ci,i*s+kh,j*s+kw] += Σ_co d_out[n,co,i,j] * W[co,ci,kh,kw]

    Parameters:
        d_out: np.ndarray of shape (N, C_out, H_out, W_out) - Upstream gradient.
        cache: tuple from conv2d_forward.

    Returns:
        d_X: np.ndarray of shape (N, C_in, H, W) - Gradient w.r.t. input (unpadded).
        d_W: np.ndarray of shape (C_out, C_in, kH, kW) - Gradient w.r.t. filters.
        d_b: np.ndarray of shape (C_out,) - Gradient w.r.t. bias.
    """
    d_X = None
    d_W = None
    d_b = None
    return d_X, d_W, d_b


def max_pool2d_forward(X, pool_size=2, stride=2):
    """
    Forward pass for 2D max pooling.

    Parameters:
        X: np.ndarray of shape (N, C, H, W) - Input.
        pool_size: int - Size of the pooling window.
        stride: int - Stride of the pooling window.

    Returns:
        Y: np.ndarray of shape (N, C, H_out, W_out) - Pooled output.
        cache: tuple (X, pool_size, stride) - Stored for backward.
    """
    Y = None
    cache = (X, pool_size, stride)
    return Y, cache


def max_pool2d_backward(d_out, cache):
    """
    Backward pass for 2D max pooling.

    Gradient is routed only to the position of the max in each window.
    All other positions get zero gradient.

    Parameters:
        d_out: np.ndarray of shape (N, C, H_out, W_out) - Upstream gradient.
        cache: tuple from max_pool2d_forward.

    Returns:
        d_X: np.ndarray of shape (N, C, H, W) - Gradient w.r.t. input.
    """
    d_X = None
    return d_X
