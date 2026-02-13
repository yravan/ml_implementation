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


# ── buffer helper (same as conv_functional) ──────────────────────────────────

def _get_buf(obj, name, shape, dtype):
    buf = getattr(obj, name, None)
    if buf is None or buf.shape != shape or buf.dtype != dtype:
        buf = np.empty(shape, dtype=dtype)
        setattr(obj, name, buf)
    return buf


# =============================================================================
# C Extension loader for BatchNorm
# =============================================================================
import ctypes

_bn_lib = None
_f32p = ctypes.POINTER(ctypes.c_float)

def _load_bn_c():
    global _bn_lib
    if _bn_lib is not None:
        return _bn_lib

    import ctypes, subprocess, pathlib, os
    _f32p = ctypes.POINTER(ctypes.c_float)
    _ci = ctypes.c_int
    _cf = ctypes.c_float

    src = pathlib.Path(__file__).parent / "_batchnorm_c.c"
    so  = pathlib.Path(__file__).parent / "_batchnorm_c.so"

    if not src.exists():
        return None

    needs_compile = not so.exists() or os.path.getmtime(src) > os.path.getmtime(so)

    if needs_compile:
        import platform
        if platform.system() == "Darwin":
            omp_prefix = None
            for prefix in ["/opt/homebrew", "/usr/local"]:
                if os.path.exists(f"{prefix}/opt/libomp/lib/libomp.dylib"):
                    omp_prefix = f"{prefix}/opt/libomp"
                    break
            if omp_prefix:
                cmd = [
                    "clang", "-O3", "-mcpu=native", "-ffast-math", "-fno-finite-math-only",
                    "-Xpreprocessor", "-fopenmp",
                    f"-I{omp_prefix}/include", f"-L{omp_prefix}/lib", "-lomp",
                    "-shared", "-fPIC", "-o", str(so), str(src),
                ]
            else:
                cmd = [
                    "clang", "-O3", "-mcpu=native", "-ffast-math", "-fno-finite-math-only",
                    "-shared", "-fPIC", "-o", str(so), str(src),
                ]
        else:
            cmd = [
                "gcc", "-O3", "-march=native", "-ffast-math", "-fno-finite-math-only",
                "-fopenmp", "-shared", "-fPIC", "-o", str(so), str(src), "-lm",
            ]

        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    lib = ctypes.CDLL(str(so))

    # forward train: x, out, norm_x, gamma, beta, mean, var, B, C, S, eps
    lib.batchnorm_forward_train.argtypes = [_f32p]*7 + [_ci]*3 + [_cf]
    lib.batchnorm_forward_train.restype = None

    # forward eval: x, out, gamma, beta, running_mean, running_var, B, C, S, eps
    lib.batchnorm_forward_eval.argtypes = [_f32p]*6 + [_ci]*3 + [_cf]
    lib.batchnorm_forward_eval.restype = None

    # backward: grad_out, norm_x, gamma, var, grad_x, grad_gamma, grad_beta, B, C, S, eps
    lib.batchnorm_backward.argtypes = [_f32p]*7 + [_ci]*3 + [_cf]
    lib.batchnorm_backward.restype = None

    _bn_lib = lib
    return lib


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
        if x.ndim not in (2, 3):
            raise RuntimeError("Invalid input for BatchNorm1d", x.shape)

        lib = _load_bn_c()
        use_c = lib is not None and x.dtype == np.float32
        ndim = x.ndim
        B, C = x.shape[:2]
        S = x.shape[2] if ndim == 3 else 1

        if training:
            if use_c:
                # C kernel expects (B, C, S) layout — reshape 2D to 3D
                x_c = np.ascontiguousarray(x.reshape(B, C, S))
                out = np.empty_like(x_c)
                norm_x = np.empty_like(x_c)
                mean = np.empty(C, dtype=np.float32)
                var = np.empty(C, dtype=np.float32)

                lib.batchnorm_forward_train(
                    x_c.ctypes.data_as(_f32p),
                    out.ctypes.data_as(_f32p),
                    norm_x.ctypes.data_as(_f32p),
                    gamma.ctypes.data_as(_f32p),
                    beta.ctypes.data_as(_f32p),
                    mean.ctypes.data_as(_f32p),
                    var.ctypes.data_as(_f32p),
                    ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(S),
                    ctypes.c_float(eps),
                )
                # Reshape back to original ndim
                out = out.reshape(x.shape)
                norm_x = norm_x.reshape(x.shape)
                output = out
            else:
                if ndim == 2:
                    mean = x.mean(axis=0)
                    var = x.var(axis=0)
                    norm_x = (x - mean[None,:]) / np.sqrt(var[None,:] + eps)
                    output = norm_x * gamma[None,:] + beta[None,:]
                else:  # ndim == 3
                    mean = x.mean(axis=(0,2))
                    var = x.var(axis=(0,2))
                    norm_x = (x - mean[None,:,None]) / np.sqrt(var[None,:,None] + eps)
                    output = norm_x * gamma[None,:,None] + beta[None,:,None]

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
            if use_c:
                x_c = np.ascontiguousarray(x.reshape(B, C, S))
                out = np.empty_like(x_c)

                lib.batchnorm_forward_eval(
                    x_c.ctypes.data_as(_f32p),
                    out.ctypes.data_as(_f32p),
                    gamma.ctypes.data_as(_f32p),
                    beta.ctypes.data_as(_f32p),
                    running_mean.ctypes.data_as(_f32p),
                    running_var.ctypes.data_as(_f32p),
                    ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(S),
                    ctypes.c_float(eps),
                )
                output = out.reshape(x.shape)
                norm_x = None  # not needed in eval
            else:
                if ndim == 2:
                    norm_x = (x - running_mean[None,:]) / np.sqrt(running_var[None,:] + eps)
                    output = norm_x * gamma[None,:] + beta[None,:]
                else:
                    norm_x = (x - running_mean[None,:,None]) / np.sqrt(running_var[None,:,None] + eps)
                    output = norm_x * gamma[None,:,None] + beta[None,:,None]

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
        lib = _load_bn_c()
        ndim = grad_output.ndim

        if ndim not in (2, 3):
            raise RuntimeError("Invalid gradient for BatchNorm1d", grad_output.shape)

        B, C = grad_output.shape[:2]
        S = grad_output.shape[2] if ndim == 3 else 1

        if lib is not None and grad_output.dtype == np.float32:
            go = np.ascontiguousarray(grad_output.reshape(B, C, S))
            nx = np.ascontiguousarray(self.norm_x.reshape(B, C, S))
            grad_x = np.empty_like(go)
            grad_gamma = np.empty(C, dtype=np.float32)
            grad_beta = np.empty(C, dtype=np.float32)

            lib.batchnorm_backward(
                go.ctypes.data_as(_f32p),
                nx.ctypes.data_as(_f32p),
                self.gamma.ctypes.data_as(_f32p),
                self.var.ctypes.data_as(_f32p),
                grad_x.ctypes.data_as(_f32p),
                grad_gamma.ctypes.data_as(_f32p),
                grad_beta.ctypes.data_as(_f32p),
                ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(S),
                ctypes.c_float(self.eps),
            )
            grad_x = grad_x.reshape(grad_output.shape)
        else:
            if ndim == 2:
                grad_beta = grad_output.sum(axis=0)
                grad_gamma = (self.norm_x * grad_output).sum(axis=0)
                N = grad_output.shape[0]
                grad_x = (self.gamma[None,:] / (N * np.sqrt(self.var[None,:] + self.eps))) * (
                    N * grad_output - grad_output.sum(axis=0, keepdims=True)
                    - self.norm_x * (grad_output * self.norm_x).sum(axis=0, keepdims=True)
                )
            else:  # ndim == 3
                grad_beta = grad_output.sum(axis=(0,2))
                grad_gamma = (self.norm_x * grad_output).sum(axis=(0,2))
                N = grad_output.shape[0] * grad_output.shape[2]
                grad_x = (self.gamma[None,:,None] / (N * np.sqrt(self.var[None,:,None] + self.eps))) * (
                    N * grad_output - grad_output.sum(axis=(0,2), keepdims=True)
                    - self.norm_x * (grad_output * self.norm_x).sum(axis=(0,2), keepdims=True)
                )

        return grad_x, grad_gamma, grad_beta



class BatchNorm2d(Function):
    """
    BatchNorm2d with pre-allocated output buffers.

    Eliminates np.empty_like allocations for out, norm_x on every call.
    The C kernel writes directly into pre-existing buffers.
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
        if not x.ndim == 4:
            raise RuntimeError("Invalid input for BatchNorm2d", x.shape)

        B, C, H, W = x.shape
        S = H * W
        lib = _load_bn_c()
        use_c = lib is not None and x.dtype == np.float32

        if training:
            if use_c:
                x_c = np.ascontiguousarray(x)
                # Pre-allocated buffers
                out = _get_buf(self, '_out_buf', x_c.shape, x_c.dtype)
                norm_x = _get_buf(self, '_norm_x_buf', x_c.shape, x_c.dtype)
                mean = _get_buf(self, '_mean_buf', (C,), np.float32)
                var = _get_buf(self, '_var_buf', (C,), np.float32)

                lib.batchnorm_forward_train(
                    x_c.ctypes.data_as(_f32p),
                    out.ctypes.data_as(_f32p),
                    norm_x.ctypes.data_as(_f32p),
                    gamma.ctypes.data_as(_f32p),
                    beta.ctypes.data_as(_f32p),
                    mean.ctypes.data_as(_f32p),
                    var.ctypes.data_as(_f32p),
                    ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(S),
                    ctypes.c_float(eps),
                )
                output = out
            else:
                mean = x.mean(axis=(0,2,3))
                var = x.var(axis=(0,2,3))
                norm_x = (x - mean[None,:,None,None]) / np.sqrt(var[None,:,None,None] + eps)
                output = norm_x * gamma[None,:,None,None] + beta[None,:,None,None]

            if running_mean is not None:
                running_mean *= (1 - momentum)
                running_mean += momentum * mean
            else:
                running_mean = mean.copy()
            if running_var is not None:
                running_var *= (1 - momentum)
                n = B * H * W
                unbiased_var = var * n / (n - 1)
                running_var[:] = (1 - momentum) * running_var + momentum * unbiased_var
            else:
                running_var = var.copy()
        else:
            if use_c:
                x_c = np.ascontiguousarray(x)
                out = _get_buf(self, '_out_buf', x_c.shape, x_c.dtype)

                lib.batchnorm_forward_eval(
                    x_c.ctypes.data_as(_f32p),
                    out.ctypes.data_as(_f32p),
                    gamma.ctypes.data_as(_f32p),
                    beta.ctypes.data_as(_f32p),
                    running_mean.ctypes.data_as(_f32p),
                    running_var.ctypes.data_as(_f32p),
                    ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(S),
                    ctypes.c_float(eps),
                )
                output = out
                norm_x = None
            else:
                norm_x = (x - running_mean[None,:,None,None]) / np.sqrt(running_var[None,:,None,None] + eps)
                output = norm_x * gamma[None,:,None,None] + beta[None,:,None,None]

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
        if grad_output.ndim != 4:
            raise RuntimeError("Invalid gradient for BatchNorm2d", grad_output.shape)

        B, C, H, W = grad_output.shape
        S = H * W
        lib = _load_bn_c()

        if lib is not None and grad_output.dtype == np.float32:
            go = np.ascontiguousarray(grad_output)
            # Pre-allocated backward buffer
            grad_x = _get_buf(self, '_grad_x_buf', go.shape, go.dtype)
            grad_gamma = _get_buf(self, '_grad_gamma_buf', (C,), np.float32)
            grad_beta = _get_buf(self, '_grad_beta_buf', (C,), np.float32)

            lib.batchnorm_backward(
                go.ctypes.data_as(_f32p),
                self.norm_x.ctypes.data_as(_f32p),
                self.gamma.ctypes.data_as(_f32p),
                self.var.ctypes.data_as(_f32p),
                grad_x.ctypes.data_as(_f32p),
                grad_gamma.ctypes.data_as(_f32p),
                grad_beta.ctypes.data_as(_f32p),
                ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(S),
                ctypes.c_float(self.eps),
            )
        else:
            N = B * H * W
            grad_beta = grad_output.sum(axis=(0,2,3))
            grad_gamma = (self.norm_x * grad_output).sum(axis=(0,2,3))
            grad_x = (self.gamma[None,:,None,None] / (N * np.sqrt(self.var[None,:,None,None] + self.eps))) * (
                N * grad_output - grad_output.sum(axis=(0,2,3), keepdims=True)
                - self.norm_x * (grad_output * self.norm_x).sum(axis=(0,2,3), keepdims=True)
            )
            grad_gamma = grad_gamma
            grad_beta = grad_beta

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

