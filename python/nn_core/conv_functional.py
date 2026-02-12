"""
Convolutional Functional Operations
====================================

This module provides functional operations for convolution layers.
Function classes handle the forward/backward computation with np.ndarray,
while Module classes in conv.py wrap these for Tensor operations.

Function Classes:
    - Conv1d: 1D convolution functional
    - Conv2d: 2D convolution functional
    - ConvTranspose2d: 2D transposed convolution functional
    - DepthwiseConv2d: Depthwise 2D convolution functional

Helper Functions:
    - im2col_2d: Image to column transformation
    - col2im_2d: Column to image transformation
    - conv1d, conv2d, conv_transpose2d: Functional interfaces
"""

import numpy as np
from typing import List, Tuple, Union, Optional

from python.foundations import Function, convert_to_function, _no_grad
from python.nn_core.pooling_functional import compute_pooling_output_size

# =============================================================================
# Helper Functions
# =============================================================================

def im2col_2d(
    x: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int]
) -> np.ndarray:
    """
    Convert image patches to columns for efficient convolution.

    Takes sliding windows from input and arranges them as columns.
    This converts convolution into matrix multiplication.

    Args:
        x: Input tensor (batch, channels, height, width)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: (stride_h, stride_w)
        padding: (padding_h, padding_w)
        dilation: (dilation_h, dilation_w)

    Returns:
        cols: Column matrix (batch, channels*kernel_h*kernel_w, out_h*out_w)
    """
    B, C, H, W = x.shape
    H_out, W_out = calculate_output_shape((H,W), (kernel_h, kernel_w), stride, padding, dilation)
    h_indices = np.arange(H_out)[:, None] * stride[0] + np.arange(kernel_h)[None, :] * dilation[0]
    w_indices = np.arange(W_out)[:, None] * stride[1] + np.arange(kernel_w)[None, :] * dilation[1]

    patches = x[:, :, h_indices[:, :, None, None], w_indices[None, None, :, :]] # B, C, H_out, Kh, W_out, Kw
    patches = patches.transpose((0, 1, 3, 5, 2, 4))
    patches = patches.reshape((B, C * kernel_h * kernel_w, -1))
    return patches


def col2im_2d(
    cols: np.ndarray,
    x_shape: Tuple[int, ...],
    kernel_h: int,
    kernel_w: int,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int]
) -> np.ndarray:
    """
    Convert columns back to image (inverse of im2col).

    Used in backward pass to convert gradient columns to gradient image.

    Args:
        cols: Column matrix (batch, channels*kernel_h*kernel_w, out_h*out_w)
        x_shape: Original input shape (batch, channels, height, width)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: (stride_h, stride_w)
        padding: (padding_h, padding_w)
        dilation: (dilation_h, dilation_w)

    Returns:
        x: Reconstructed tensor (batch, channels, height, width)
    """
    B, C, H, W = x_shape
    H_out, W_out = calculate_output_shape(
        (H, W), (kernel_h, kernel_w), stride, padding, dilation
    )

    h_indices = (
        np.arange(H_out)[:, None] * stride[0]
        + np.arange(kernel_h)[None, :] * dilation[0]
    )  # H_out, Kh
    w_indices = (
        np.arange(W_out)[:, None] * stride[1]
        + np.arange(kernel_w)[None, :] * dilation[1]
    )  # W_out, Kw

    # Linear indices into the (H, W) plane
    hw_linear = (
        h_indices[:, :, None, None] * W + w_indices[None, None, :, :]
    )  # H_out, Kh, W_out, Kw

    # Reshape cols to (B, C, Kh, Kw, H_out, W_out), reorder to match hw_linear
    patches = cols.reshape(B, C, kernel_h, kernel_w, H_out, W_out)
    patches = patches.transpose(0, 1, 4, 2, 5, 3)  # B, C, H_out, Kh, W_out, Kw

    # Compute full 1D indices into (B*C*H*W)
    bc_offset = np.arange(B * C).reshape(-1, 1) * (H * W)  # (B*C, 1)
    hw_flat = hw_linear.ravel()  # (H_out*Kh*W_out*Kw,)
    all_indices = (bc_offset + hw_flat).ravel()  # (B*C*H_out*Kh*W_out*Kw,)

    flat_patches = patches.reshape(B * C, -1).ravel()  # (B*C*H_out*Kh*W_out*Kw,)

    output = np.zeros(B * C * H * W)
    np.add.at(output, all_indices, flat_patches)
    return output.reshape(B, C, H, W)



def calculate_output_shape(
    input_shape: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1)
) -> Tuple[int, int]:
    """
    Calculate output spatial dimensions for 2D convolution.

    Formula:
        H_out = floor((H_in + 2*pad_h - dil_h*(K_h - 1) - 1) / stride_h) + 1
        W_out = floor((W_in + 2*pad_w - dil_w*(K_w - 1) - 1) / stride_w) + 1
    """
    h_in, w_in = input_shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation

    h_out = ((h_in + 2*p_h - d_h*(k_h - 1) - 1) // s_h) + 1
    w_out = ((w_in + 2*p_w - d_w*(k_w - 1) - 1) // s_w) + 1

    return (h_out, w_out)



def calculate_transposed_output_shape(
    input_shape: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    output_padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1)
) -> Tuple[int, int]:
    """
    Calculate output shape for transposed convolution.

    Formula:
        H_out = (H_in - 1)*stride_h - 2*pad_h + dil_h*(K_h - 1) + out_pad_h + 1
    """
    h_in, w_in = input_shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    op_h, op_w = output_padding
    d_h, d_w = dilation

    h_out = (h_in - 1)*s_h - 2*p_h + d_h*(k_h - 1) + op_h + 1
    w_out = (w_in - 1)*s_w - 2*p_w + d_w*(k_w - 1) + op_w + 1

    return (h_out, w_out)


def calculate_receptive_field(layer_configs: List[dict]) -> int:
    """
    Calculate cumulative receptive field for stacked conv layers.

    Args:
        layer_configs: List of dicts with keys: kernel_size, stride, dilation

    Returns:
        Total receptive field size
    """
    rf = 1
    for config in layer_configs:
        k = config.get('kernel_size', 3)
        s = config.get('stride', 1)
        d = config.get('dilation', 1)
        rf += (k - 1) * s * d
    return rf


def count_conv_parameters(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    groups: int = 1,
    bias: bool = True
) -> int:
    """Count learnable parameters in a Conv2d layer."""
    k_h, k_w = kernel_size
    num_weights = out_channels * (in_channels // groups) * k_h * k_w
    num_bias = out_channels if bias else 0
    return num_weights + num_bias


def count_depthwise_separable_parameters(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),
    bias: bool = True
) -> Tuple[int, int, float]:
    """
    Compare parameters: standard conv vs depthwise separable.

    Returns:
        (standard_params, depthwise_sep_params, reduction_factor)
    """
    k_h, k_w = kernel_size

    # Standard convolution
    standard = out_channels * in_channels * k_h * k_w
    if bias:
        standard += out_channels

    # Depthwise separable
    depthwise = in_channels * k_h * k_w
    if bias:
        depthwise += in_channels
    pointwise = in_channels * out_channels
    if bias:
        pointwise += out_channels
    depthwise_sep = depthwise + pointwise

    reduction = standard / depthwise_sep if depthwise_sep > 0 else 1.0

    return standard, depthwise_sep, reduction




# =============================================================================
# Function Classes
# =============================================================================

class Conv1d(Function):
    """
    1D Convolution functional operation for autograd.

    Forward:
        y[n, c_out, l] = Σ_{c_in} Σ_{k} x[n, c_in, l*stride + k] * w[c_out, c_in, k] + b[c_out]
    """
    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1):
        B, C, L = x.shape
        C_out, _, K = weight.shape

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
        L_padded = x.shape[2]
        L_out = (L_padded - dilation * (K - 1) - 1) // stride + 1

        indices = np.arange(L_out)[:, None] * stride + np.arange(K) * dilation  # L_out, K
        patches = x[:, :, indices]  # B, C, L_out, K
        cols = patches.transpose(0, 2, 1, 3).reshape(B, L_out, -1)  # B, L_out, C*K

        W_mat = weight.reshape(C_out, -1)  # C_out, C*K
        output = cols @ W_mat.T  # B, L_out, C_out
        if bias is not None:
            output += bias
        output = output.transpose(0, 2, 1)  # B, C_out, L_out

        global _no_grad
        if not _no_grad:
            self.cols = cols
            self.weight = weight
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.K = K
            self.x_shape = x.shape  # padded shape
            self.has_bias = bias is not None

        return output

    def backward(self, grad_output):
        B, C_out, L_out = grad_output.shape
        grad_out_flat = grad_output.transpose(0, 2, 1)  # B, L_out, C_out
        W_mat = self.weight.reshape(C_out, -1)           # C_out, C*K

        # grad_bias
        grad_bias = grad_output.sum(axis=(0, 2)) if self.has_bias else None

        # grad_weight: grad_out.T @ cols, summed over batch
        grad_W_mat = np.einsum('blo,blk->ok', grad_out_flat, self.cols)  # C_out, C*K
        grad_weight = grad_W_mat.reshape(self.weight.shape)

        # grad_cols: grad_out @ W_mat
        grad_cols = grad_out_flat @ W_mat  # B, L_out, C*K

        # col2im_1d: scatter grad_cols back to input
        B, C, L_padded = self.x_shape
        C_in = C
        grad_patches = grad_cols.reshape(B, L_out, C_in, self.K).transpose(0, 2, 1, 3)  # B, C, L_out, K

        indices = np.arange(L_out)[:, None] * self.stride + np.arange(self.K) * self.dilation  # L_out, K
        bc_offset = np.arange(B * C_in).reshape(-1, 1) * L_padded
        all_indices = (bc_offset + indices.ravel()).ravel()

        grad_x = np.zeros(B * C_in * L_padded)
        np.add.at(grad_x, all_indices, grad_patches.reshape(B * C_in, -1).ravel())
        grad_x = grad_x.reshape(B, C_in, L_padded)

        # Strip padding
        if self.padding > 0:
            grad_x = grad_x[:, :, self.padding:-self.padding]

        return grad_x, grad_weight, grad_bias


class Conv2d(Function):
    """
    2D Convolution functional operation for autograd.

    Forward:
        y[n,c_out,h,w] = Σ_{c_in,kh,kw} x[n,c_in,h*s+kh*d,w*s+kw*d] * w[c_out,c_in,kh,kw] + b[c_out]
    """

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert x.ndim == 4
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)

        B, C, H, W = x.shape
        C_out = weight.shape[0]
        kh, kw = weight.shape[-2:]
        H_out, W_out = calculate_output_shape((H, W), (kh, kw), stride, padding, dilation)

        if padding[0] > 0 or padding[1] > 0:
            x = np.pad(x, ((0,0), (0,0), (padding[0], padding[0]), (padding[1], padding[1])))

        C_per_group = C // groups
        C_out_per_group = C_out // groups

        cols_list = []
        outputs = []
        for g in range(groups):
            x_g = x[:, g * C_per_group:(g + 1) * C_per_group]
            w_g = weight[g * C_out_per_group:(g + 1) * C_out_per_group]

            cols_g = im2col_2d(x_g, kh, kw, stride, (0, 0), dilation)  # B, C_per_group*kh*kw, N
            W_mat_g = w_g.reshape(C_out_per_group, -1)

            out_g = np.einsum("oi,bin->bon", W_mat_g, cols_g)  # B, C_out_per_group, N

            outputs.append(out_g)
            cols_list.append(cols_g)

        output = np.concatenate(outputs, axis=1).reshape(B, C_out, H_out, W_out)

        if bias is not None:
            output += bias[None, :, None, None]

        global _no_grad
        if not _no_grad:
            self.cols_list = cols_list
            self.weight = weight
            self.x_shape = x.shape
            self.kh = kh
            self.kw = kw
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.has_bias = bias is not None

        return output

    def backward(self, grad_output):
        B, C_out, H_out, W_out = grad_output.shape
        groups = self.groups
        C = self.x_shape[1]
        C_per_group = C // groups
        C_out_per_group = C_out // groups

        grad_weight = np.zeros_like(self.weight)
        grad_x = np.zeros(self.x_shape)

        for g in range(groups):
            grad_out_g = grad_output[:, g * C_out_per_group:(g + 1) * C_out_per_group]
            grad_out_flat = grad_out_g.reshape(B, C_out_per_group, -1)  # B, C_out_per_group, N

            w_g = self.weight[g * C_out_per_group:(g + 1) * C_out_per_group]
            W_mat_g = w_g.reshape(C_out_per_group, -1)
            cols_g = self.cols_list[g]

            # grad_weight for this group
            grad_W_g = np.einsum('bon,bin->oi', grad_out_flat, cols_g)
            grad_weight[g * C_out_per_group:(g + 1) * C_out_per_group] = grad_W_g.reshape(w_g.shape)

            # grad_cols for this group
            grad_cols_g = np.einsum('oi,bon->bin', W_mat_g, grad_out_flat)

            # col2im for this group's channels
            x_g_shape = (B, C_per_group, self.x_shape[2], self.x_shape[3])
            grad_x_g = col2im_2d(grad_cols_g, x_g_shape, self.kh, self.kw,
                                 self.stride, (0, 0), self.dilation)
            grad_x[:, g * C_per_group:(g + 1) * C_per_group] = grad_x_g

        # Strip padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            grad_x = grad_x[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]

        if self.has_bias:
            grad_bias = grad_output.sum(axis=(0, 2, 3))
            return grad_x, grad_weight, grad_bias
        else:
            return grad_x, grad_weight, None



class ConvTranspose2d(Function):
    """
    2D Transposed Convolution (Deconvolution) functional operation.

    The forward pass is mathematically equivalent to the backward pass
    of Conv2d with respect to the input.
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1
    ) -> np.ndarray:
        """
        Compute transposed 2D convolution.

        Args:
            x: Input (batch_size, in_channels, height, width)
            weight: Kernel (in_channels, out_channels/groups, kernel_h, kernel_w)
            bias: Optional bias (out_channels,)
            stride: Convolution stride
            padding: Zero-padding
            output_padding: Additional padding for output
            dilation: Kernel dilation
            groups: Number of groups

        Returns:
            Upsampled output (batch_size, out_channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement ConvTranspose2d forward\n"
            "Hint: Input dilation approach:\n"
            "  1. Dilate input: insert (stride-1) zeros between elements\n"
            "  2. Pad with kernel_size - 1 - original_padding\n"
            "  3. Perform regular convolution with flipped weights"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute gradients for transposed 2D convolution.

        Returns:
            Tuple of (grad_x, grad_weight, grad_bias)
        """
        raise NotImplementedError(
            "TODO: Implement ConvTranspose2d backward\n"
            "Hint: The backward of transposed conv is regular conv"
        )


class DepthwiseConv2d(Function):
    """
    Depthwise 2D Convolution functional operation.

    Performs separate convolution for each input channel.
    Equivalent to grouped convolution with groups=in_channels.
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1
    ) -> np.ndarray:
        """
        Compute depthwise 2D convolution.

        Args:
            x: Input (batch_size, channels, height, width)
            weight: Kernel (channels, 1, kernel_h, kernel_w)
            bias: Optional bias (channels,)

        Returns:
            Output (batch_size, channels, height_out, width_out)
        """
        raise NotImplementedError(
            "TODO: Implement DepthwiseConv2d forward\n"
            "Hint: Process each channel independently or use Conv2d with groups=in_channels"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for depthwise 2D convolution."""
        raise NotImplementedError("TODO: Implement DepthwiseConv2d backward")


class PointwiseConv2d(Function):
    """
    Pointwise (1x1) Convolution functional operation.

    Mixes information across channels without spatial computation.
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute pointwise (1×1) convolution.

        Args:
            x: Input (batch_size, in_channels, height, width)
            weight: Kernel (out_channels, in_channels, 1, 1)
            bias: Optional bias (out_channels,)

        Returns:
            Output (batch_size, out_channels, height, width)
        """
        raise NotImplementedError(
            "TODO: Implement PointwiseConv2d forward\n"
            "Hint: 1×1 conv is just matrix multiply per spatial position:\n"
            "  # Reshape: (B, C_in, H, W) -> (B*H*W, C_in)\n"
            "  # Multiply: output = input @ weight.squeeze().T + bias\n"
            "  # Reshape back: (B*H*W, C_out) -> (B, C_out, H, W)"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for pointwise convolution."""
        raise NotImplementedError("TODO: Implement PointwiseConv2d backward")


# =============================================================================
# Conv3d Function
# =============================================================================

class Conv3d(Function):
    """
    3D Convolution functional operation.

    Used for volumetric data (video, 3D medical imaging, etc.).
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1
    ) -> np.ndarray:
        """
        Compute 3D convolution.

        Args:
            x: Input (batch_size, in_channels, depth, height, width)
            weight: Kernel (out_channels, in_channels//groups, kernel_d, kernel_h, kernel_w)
            bias: Optional bias (out_channels,)
            stride: Stride for each dimension
            padding: Padding for each dimension
            dilation: Dilation for each dimension
            groups: Number of groups for grouped convolution

        Returns:
            Output (batch_size, out_channels, depth_out, height_out, width_out)

        TODO: Implement Conv3d forward
        Hint: Extend the im2col approach to 3D:
            1. For each 3D patch in input, flatten to vector
            2. Stack all patches into a matrix
            3. Multiply with flattened weight matrix
            4. Reshape to output volume
        """
        raise NotImplementedError(
            "TODO: Implement Conv3d forward\n"
            "Hint: Extend im2col_2d to im2col_3d:\n"
            "  1. Extract 3D patches using sliding window over (D, H, W)\n"
            "  2. Reshape patches: (batch, out_d*out_h*out_w, C_in*kD*kH*kW)\n"
            "  3. Reshape weights: (C_out, C_in*kD*kH*kW)\n"
            "  4. Matrix multiply: output = patches @ weights.T + bias\n"
            "  5. Reshape: (batch, out_d, out_h, out_w, C_out) -> (batch, C_out, out_d, out_h, out_w)"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute gradients for 3D convolution.

        Returns:
            grad_x: Gradient w.r.t. input
            grad_weight: Gradient w.r.t. weight
            grad_bias: Gradient w.r.t. bias (or None)

        TODO: Implement Conv3d backward
        Hint: Similar to Conv2d backward but in 3D
        """
        raise NotImplementedError(
            "TODO: Implement Conv3d backward\n"
            "Hint: Apply col2im_3d to convert gradient columns back to volume"
        )


# =============================================================================
# ConvTranspose1d Function
# =============================================================================

class ConvTranspose1d(Function):
    """
    1D Transposed Convolution (Deconvolution) functional operation.

    Used for upsampling 1D signals (audio, time series).
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1
    ) -> np.ndarray:
        """
        Compute 1D transposed convolution.

        Args:
            x: Input (batch_size, in_channels, length)
            weight: Kernel (in_channels, out_channels//groups, kernel_size)
            bias: Optional bias (out_channels,)
            stride: Upsampling factor
            padding: Padding to remove from output
            output_padding: Additional size added to output
            dilation: Dilation factor
            groups: Number of groups for grouped convolution

        Returns:
            Output (batch_size, out_channels, length_out)
            where length_out = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

        TODO: Implement ConvTranspose1d forward
        """
        raise NotImplementedError(
            "TODO: Implement ConvTranspose1d forward\n"
            "Hint: Transposed conv is the gradient of regular conv:\n"
            "  1. Insert (stride-1) zeros between input elements\n"
            "  2. Pad with (kernel_size - 1 - padding) zeros\n"
            "  3. Convolve with flipped kernel\n"
            "Or use the col2im approach directly"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute gradients for 1D transposed convolution.

        TODO: Implement ConvTranspose1d backward
        Hint: The backward of transposed conv is regular conv
        """
        raise NotImplementedError(
            "TODO: Implement ConvTranspose1d backward\n"
            "Hint: grad_x = conv1d(grad_output, weight)"
        )


# =============================================================================
# ConvTranspose3d Function
# =============================================================================

class ConvTranspose3d(Function):
    """
    3D Transposed Convolution (Deconvolution) functional operation.

    Used for upsampling volumetric data (3D image generation, video upsampling).
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1
    ) -> np.ndarray:
        """
        Compute 3D transposed convolution.

        Args:
            x: Input (batch_size, in_channels, depth, height, width)
            weight: Kernel (in_channels, out_channels//groups, kernel_d, kernel_h, kernel_w)
            bias: Optional bias (out_channels,)
            stride: Upsampling factor for each dimension
            padding: Padding to remove from output
            output_padding: Additional size added to output
            dilation: Dilation factor for each dimension
            groups: Number of groups

        Returns:
            Output (batch_size, out_channels, depth_out, height_out, width_out)

        TODO: Implement ConvTranspose3d forward
        """
        raise NotImplementedError(
            "TODO: Implement ConvTranspose3d forward\n"
            "Hint: Similar to ConvTranspose2d but extended to 3D"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for 3D transposed convolution."""
        raise NotImplementedError(
            "TODO: Implement ConvTranspose3d backward\n"
            "Hint: The backward of transposed conv is regular conv"
        )


# =============================================================================
# DepthwiseSeparableConv2d Function
# =============================================================================

class DepthwiseSeparableConv2d(Function):
    """
    Depthwise Separable 2D Convolution functional operation.

    Combines depthwise convolution followed by pointwise convolution.
    Used in efficient architectures like MobileNet.
    """

    def forward(
        self,
        x: np.ndarray,
        depthwise_weight: np.ndarray,
        pointwise_weight: np.ndarray,
        depthwise_bias: Optional[np.ndarray] = None,
        pointwise_bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1
    ) -> np.ndarray:
        """
        Compute depthwise separable convolution.

        Args:
            x: Input (batch_size, in_channels, height, width)
            depthwise_weight: Depthwise kernel (in_channels, 1, kernel_h, kernel_w)
            pointwise_weight: Pointwise kernel (out_channels, in_channels, 1, 1)
            depthwise_bias: Optional depthwise bias (in_channels,)
            pointwise_bias: Optional pointwise bias (out_channels,)
            stride: Stride (applied in depthwise conv)
            padding: Padding (applied in depthwise conv)
            dilation: Dilation (applied in depthwise conv)

        Returns:
            Output (batch_size, out_channels, height_out, width_out)

        TODO: Implement DepthwiseSeparableConv2d forward
        Hint: output = pointwise_conv(depthwise_conv(x))
        """
        raise NotImplementedError(
            "TODO: Implement DepthwiseSeparableConv2d forward\n"
            "Hint: Chain depthwise and pointwise convolutions:\n"
            "  1. depthwise_out = depthwise_conv2d(x, depthwise_weight, depthwise_bias, stride, padding)\n"
            "  2. output = pointwise_conv2d(depthwise_out, pointwise_weight, pointwise_bias)"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute gradients for depthwise separable convolution.

        Returns:
            grad_x: Gradient w.r.t. input
            grad_depthwise_weight: Gradient w.r.t. depthwise kernel
            grad_pointwise_weight: Gradient w.r.t. pointwise kernel
            grad_depthwise_bias: Gradient w.r.t. depthwise bias (or None)
            grad_pointwise_bias: Gradient w.r.t. pointwise bias (or None)

        TODO: Implement DepthwiseSeparableConv2d backward
        """
        raise NotImplementedError(
            "TODO: Implement DepthwiseSeparableConv2d backward\n"
            "Hint: Backprop through pointwise first, then through depthwise"
        )


# =============================================================================
# DilatedConv2d Function
# =============================================================================

class DilatedConv2d(Function):
    """
    Dilated (Atrous) 2D Convolution functional operation.

    Increases receptive field without increasing parameters or computation.
    Used in semantic segmentation (DeepLab, etc.).
    """

    def forward(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 2
    ) -> np.ndarray:
        """
        Compute dilated 2D convolution.

        Args:
            x: Input (batch_size, in_channels, height, width)
            weight: Kernel (out_channels, in_channels, kernel_h, kernel_w)
            bias: Optional bias (out_channels,)
            stride: Convolution stride
            padding: Input padding
            dilation: Spacing between kernel elements (dilation > 1 for atrous conv)

        Returns:
            Output (batch_size, out_channels, height_out, width_out)

        Note: Effective kernel size = dilation * (kernel_size - 1) + 1

        TODO: Implement DilatedConv2d forward
        Hint: This is just Conv2d with dilation parameter
        """
        raise NotImplementedError(
            "TODO: Implement DilatedConv2d forward\n"
            "Hint: Use Conv2d.forward with dilation parameter, or\n"
            "modify im2col_2d to skip (dilation-1) elements between samples"
        )

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute gradients for dilated 2D convolution."""
        raise NotImplementedError(
            "TODO: Implement DilatedConv2d backward\n"
            "Hint: Similar to Conv2d backward but with dilation"
        )


# =============================================================================
# Functional Interfaces (using convert_to_function)
# =============================================================================

conv1d = convert_to_function(Conv1d)
conv2d = convert_to_function(Conv2d)
conv3d = convert_to_function(Conv3d)
conv_transpose1d = convert_to_function(ConvTranspose1d)
conv_transpose2d = convert_to_function(ConvTranspose2d)
conv_transpose3d = convert_to_function(ConvTranspose3d)
depthwise_conv2d = convert_to_function(DepthwiseConv2d)
pointwise_conv2d = convert_to_function(PointwiseConv2d)
depthwise_separable_conv2d = convert_to_function(DepthwiseSeparableConv2d)
dilated_conv2d = convert_to_function(DilatedConv2d)
