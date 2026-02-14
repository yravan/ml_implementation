"""
Pooling Functional Operations
=============================

This module provides functional operations for pooling layers.
Function classes handle the forward/backward computation with np.ndarray,
while Module classes in pooling.py wrap these for Tensor operations.

Function Classes:
    - MaxPool1d: 1D max pooling functional
    - MaxPool2d: 2D max pooling functional
    - AvgPool1d: 1D average pooling functional
    - AvgPool2d: 2D average pooling functional
    - AdaptiveMaxPool2d: Adaptive max pooling functional
    - AdaptiveAvgPool2d: Adaptive average pooling functional

Helper Functions:
    - max_pool1d, max_pool2d: Max pooling interfaces
    - avg_pool1d, avg_pool2d: Average pooling interfaces
    - adaptive_max_pool2d, adaptive_avg_pool2d: Adaptive pooling interfaces
    - global_avg_pool2d, global_max_pool2d: Global pooling interfaces
"""
import platform

import numpy as np
from typing import Tuple, Union, Optional

from python.foundations import Function, convert_to_function, _no_grad


# =============================================================================
# C Extension — auto-compile and load
# =============================================================================

import ctypes, subprocess, pathlib, os, warnings

_c_lib = None  # loaded on first use
_f32p = ctypes.POINTER(ctypes.c_float)
_i64p = ctypes.POINTER(ctypes.c_longlong)
_ci = ctypes.c_int
def _load_c_extension():
    """Compile (if needed) and load the C im2col/col2im shared library."""
    global _c_lib
    if _c_lib is not None:
        return _c_lib

    src = pathlib.Path(__file__).parent / "_pooling_c.c"
    so  = pathlib.Path(__file__).parent / "_pooling_c.so"

    if not src.exists():
        warnings.warn(
            f"C source {src} not found — using pure-numpy fallback. "
            "Place _conv_c.c next to conv_functional.py for 2-5× speedup.",
            RuntimeWarning, stacklevel=3,
        )
        return None

    needs_compile = (
        not so.exists()
        or os.path.getmtime(src) > os.path.getmtime(so)
    )

    needs_compile = (
        not so.exists()
        or os.path.getmtime(src) > os.path.getmtime(so)
    )

    if needs_compile:
        system = platform.system()

        if system == 'Linux':
            cmd = [
                "gcc", "-O3", "-march=native", "-ffast-math", "-fno-finite-math-only",
                "-fopenmp",
                "-shared", "-fPIC",
                "-o", str(so), str(src),
            ]
        elif system == 'Darwin':
            # macOS: clang needs a separately installed libomp
            omp_prefix = None
            for prefix in ["/opt/homebrew", "/usr/local"]:
                if os.path.exists(f"{prefix}/opt/libomp/lib/libomp.dylib"):
                    omp_prefix = f"{prefix}/opt/libomp"
                    break

            base = ["clang", "-O3", "-mcpu=native", "-ffast-math", "-fno-finite-math-only"]
            if omp_prefix:
                cmd = base + [
                    "-Xpreprocessor", "-fopenmp",
                    f"-I{omp_prefix}/include",
                    f"-L{omp_prefix}/lib", "-lomp",
                    "-shared", "-fPIC",
                    "-o", str(so), str(src),
                ]
            else:
                warnings.warn(
                    "libomp not found — compiling without OpenMP. "
                    "Run 'brew install libomp' for multi-threaded im2col/col2im.",
                    RuntimeWarning, stacklevel=3,
                )
                cmd = base + ["-shared", "-fPIC", "-o", str(so), str(src)]
        else:
            warnings.warn(
                f"Unsupported platform '{system}' — using pure-numpy fallback.",
                RuntimeWarning, stacklevel=3,
            )
            return None

        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            warnings.warn(
                "Failed to compile C extension — using pure-numpy fallback.",
                RuntimeWarning, stacklevel=3,
            )
            return None

    lib = ctypes.CDLL(str(so))
    # 1d
    lib.max_pool1d.argtypes = [
        _f32p, _f32p, _i64p, _ci, _ci, _ci, _ci, _ci, _ci, _ci,
    ]
    lib.max_pool1d.restype = None
    lib.max_pool1d_backward.argtypes = [
        _f32p, _f32p, _i64p, _ci, _ci, _ci, _ci,
    ]
    lib.max_pool1d_backward.restype = None

    # 2d
    lib.max_pool2d.argtypes = [
        _f32p, _f32p, _i64p,
        _ci, _ci, _ci, _ci,
        _ci, _ci, _ci, _ci, _ci, _ci,
        _ci, _ci,
    ]
    lib.max_pool2d.restype = None
    lib.max_pool2d_backward.argtypes = [
        _f32p, _f32p, _i64p,
        _ci, _ci, _ci, _ci,
        _ci, _ci,
    ]
    lib.max_pool2d_backward.restype = None


    # avg_pool1d
    lib.avg_pool1d.argtypes = [
        _f32p, _f32p,
        _ci, _ci, _ci,
        _ci, _ci,
        _ci,
        _ci, _ci, _ci,
    ]
    lib.avg_pool1d.restype = None
    lib.avg_pool1d_backward.argtypes = [
        _f32p, _f32p,
        _ci, _ci, _ci,
        _ci, _ci,
        _ci,
        _ci, _ci, _ci,
    ]
    lib.avg_pool1d_backward.restype = None

    # avg_pool2d
    lib.avg_pool2d.argtypes = [
        _f32p, _f32p,
        _ci, _ci, _ci, _ci,
        _ci, _ci, _ci, _ci,
        _ci, _ci,
        _ci, _ci, _ci, _ci, _ci,
    ]
    lib.avg_pool2d.restype = None
    lib.avg_pool2d_backward.argtypes = [
        _f32p, _f32p,
        _ci, _ci, _ci, _ci,
        _ci, _ci, _ci, _ci,
        _ci, _ci,
        _ci, _ci, _ci, _ci, _ci,
    ]
    lib.avg_pool2d_backward.restype = None

    lib.adaptive_avg_pool2d.argtypes = [
        _f32p, _f32p, _ci, _ci, _ci, _ci, _ci, _ci,
    ]
    lib.adaptive_avg_pool2d.restype = None
    lib.adaptive_avg_pool2d_backward.argtypes = [
        _f32p, _f32p, _ci, _ci, _ci, _ci, _ci, _ci,
    ]
    lib.adaptive_avg_pool2d_backward.restype = None


    _c_lib = lib
    return lib



# =============================================================================
# Helper Functions
# =============================================================================

def compute_pooling_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
) -> int:
    """
    Compute output size for pooling.

    Formula:
        output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
    """
    numerator = input_size + 2*padding - dilation*(kernel_size - 1) - 1
    if ceil_mode:
        output = int(np.ceil(numerator / stride)) + 1
    else:
        output = numerator // stride + 1
    return output


# =============================================================================
# Max Pooling Function Classes
# =============================================================================

class MaxPool1d(Function):
    """
    Max Pooling 1D functional operation.

    Applies max pooling over a 1D input signal.
    """

    def forward(
        self,
        x: np.ndarray,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        ceil_mode: bool = False,
    ) -> np.ndarray:
        """
        Compute 1D max pooling.

        Args:
            x: Input (batch_size, channels, length)
            kernel_size: Pooling window size
            stride: Step size (defaults to kernel_size)
            padding: Zero-padding
            dilation: Dilation factor
            ceil_mode: Use ceiling for output size

        Returns:
            Pooled output
        """
        assert x.ndim == 3
        lib = _load_c_extension()
        B, C, L = x.shape
        input_size = x.shape[-1]
        output_size = compute_pooling_output_size(input_size, kernel_size, stride, padding, dilation, ceil_mode)

        if padding > 0:
            padded_x = np.full((B, C, L + 2 * padding), np.finfo(np.float32).min, dtype=x.dtype)  # -inf not 0 for max
            padded_x[:, :, padding:padding + L] = x
            x = padded_x
            L = L + 2 * padding

        if lib is not None:
            out = np.zeros((*x.shape[:-1], output_size), dtype=x.dtype)
            indices = np.zeros_like(out, dtype=np.int64)
            lib.max_pool1d(
                x.ctypes.data_as(_f32p),
                out.ctypes.data_as(_f32p),
                indices.ctypes.data_as(_i64p),
                _ci(B), _ci(C), _ci(L),
                _ci(kernel_size), _ci(stride), _ci(dilation),
                _ci(output_size),
            )
        else:
            # numpy fallback
            out = np.empty((B, C, output_size), dtype=x.dtype)
            indices = np.empty((B, C, output_size), dtype=np.int32)
            for l in range(output_size):
                positions = np.arange(kernel_size) * dilation + l * stride
                window = x[:, :, positions]             # (B, C, K)
                idx = window.argmax(axis=-1)            # (B, C)
                indices[:, :, l] = positions[idx]
                out[:, :, l] = np.take_along_axis(window, idx[:, :, None], axis=-1).squeeze(-1)
        global _no_grad
        if not _no_grad:
            # store which indices the output came from
            self.shape = x.shape
            self.output_indices = indices # B, C, L_out
            self.padding = padding

        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient for 1D max pooling.

        Gradient flows only through the max elements.
        """
        lib = _load_c_extension()
        if lib is not None:
            grad_x = np.zeros(self.shape, dtype=grad_output.dtype)
            L_out = grad_output.shape[-1]
            B, C, L = self.shape
            lib.max_pool1d_backward(
                grad_output.ctypes.data_as(_f32p),
                grad_x.ctypes.data_as(_f32p),
                self.output_indices.ctypes.data_as(_i64p),
                _ci(B), _ci(C), _ci(L),
                _ci(L_out),
            )
        else:
            B, C, L_out = grad_output.shape
            grad_x = np.zeros(self.shape, dtype=grad_output.dtype)
            b_idx = np.arange(B)[:, None, None]
            c_idx = np.arange(C)[None, :, None]
            np.add.at(grad_x, (b_idx, c_idx, self.output_indices), grad_output)
        if self.padding > 0:
            grad_x = grad_x[:, :, self.padding:-self.padding]
        return (grad_x,)


class MaxPool2d(Function):
    def forward(
        self,
        x: np.ndarray,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert x.ndim == 4
        B, C, H, W = x.shape

        if isinstance(kernel_size, int):
            Kh = Kw = kernel_size
        else:
            Kh, Kw = kernel_size
        if stride is None:
            Sh, Sw = Kh, Kw
        elif isinstance(stride, int):
            Sh = Sw = stride
        else:
            Sh, Sw = stride
        if isinstance(padding, int):
            Ph = Pw = padding
        else:
            Ph, Pw = padding
        if isinstance(dilation, int):
            Dh = Dw = dilation
        else:
            Dh, Dw = dilation

        H_out = compute_pooling_output_size(H, Kh, Sh, Ph, Dh, ceil_mode)
        W_out = compute_pooling_output_size(W, Kw, Sw, Pw, Dw, ceil_mode)

        if Ph > 0 or Pw > 0:
            padded_x = np.full((B, C, H + 2 * Ph, W + 2 * Pw), np.finfo(np.float32).min, dtype=x.dtype)
            padded_x[:, :, Ph:Ph + H, Pw:Pw + W] = x
            x = padded_x
            H = H + 2 * Ph
            W = W + 2 * Pw

        lib = _load_c_extension()
        if lib is not None and x.dtype == np.float32:
            x = np.ascontiguousarray(x)
            out = np.empty((B, C, H_out, W_out), dtype=x.dtype)
            indices = np.empty((B, C, H_out, W_out), dtype=np.int64)
            lib.max_pool2d(
                x.ctypes.data_as(_f32p),
                out.ctypes.data_as(_f32p),
                indices.ctypes.data_as(_i64p),
                _ci(B), _ci(C), _ci(H), _ci(W),
                _ci(Kh), _ci(Kw), _ci(Sh), _ci(Sw), _ci(Dh), _ci(Dw),
                _ci(H_out), _ci(W_out),
            )
        else:
            # numpy fallback
            patch_h = np.arange(H_out)[:, None] * Sh + np.arange(Kh)[None, :] * Dh
            patch_w = np.arange(W_out)[:, None] * Sw + np.arange(Kw)[None, :] * Dw
            patches = x[:, :, patch_h[:, :, None, None], patch_w[None, None, :, :]]
            patches = patches.transpose((0, 1, 2, 4, 3, 5))
            flat_patches = patches.reshape(B, C, H_out, W_out, -1)
            flat_idx = flat_patches.argmax(axis=-1)
            out = flat_patches.max(axis=-1)

            kh_idx = flat_idx // Kw
            kw_idx = flat_idx % Kw
            h_coords = np.arange(H_out)[None, None, :, None] * Sh + kh_idx * Dh
            w_coords = np.arange(W_out)[None, None, None, :] * Sw + kw_idx * Dw
            # Flatten to 1D index into (H, W) for consistency with C path
            indices = h_coords * W + w_coords

        global _no_grad
        if not _no_grad:
            self.shape = x.shape
            self.indices = indices
            self.padding = (Ph, Pw)

        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        B, C, H, W = self.shape
        _, _, H_out, W_out = grad_output.shape
        Ph, Pw = self.padding

        lib = _load_c_extension()
        if lib is not None and grad_output.dtype == np.float32:
            if 'grad_x' not in self.__dict__ or self.grad_x.shape != grad_output.shape:
                grad_x = np.zeros(self.shape, dtype=grad_output.dtype)
            else:
                grad_x = self.grad_x
                grad_x.fill(0)
            lib.max_pool2d_backward(
                grad_output.ctypes.data_as(_f32p),
                grad_x.ctypes.data_as(_f32p),
                self.indices.ctypes.data_as(_i64p),
                _ci(B), _ci(C), _ci(H), _ci(W),
                _ci(H_out), _ci(W_out),
            )
        else:
            grad_x = np.zeros(self.shape, dtype=grad_output.dtype)
            b_idx = np.arange(B)[:, None, None, None]
            c_idx = np.arange(C)[None, :, None, None]
            # Convert flat index back to h, w
            h_idx = self.indices // W
            w_idx = self.indices % W
            np.add.at(grad_x, (b_idx, c_idx, h_idx, w_idx), grad_output)

        self.grad_x = grad_x

        if Ph > 0 or Pw > 0:
            grad_x = grad_x[:, :, Ph:H - Ph, Pw:W - Pw]

        return (grad_x,)

# =============================================================================
# Average Pooling
# =============================================================================

class AvgPool1d(Function):
    def forward(
        self,
        x: np.ndarray,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> np.ndarray:
        assert x.ndim == 3
        if stride is None:
            stride = kernel_size
        B, C, L = x.shape
        orig_L = L
        output_size = compute_pooling_output_size(L, kernel_size, stride, padding, 1, ceil_mode)

        if padding > 0:
            padded_x = np.zeros((B, C, L + 2 * padding), dtype=x.dtype)
            padded_x[:, :, padding:padding + L] = x
            x = padded_x
            L = L + 2 * padding

        lib = _load_c_extension()
        if lib is not None and x.dtype == np.float32:
            x = np.ascontiguousarray(x)
            out = np.empty((B, C, output_size), dtype=x.dtype)
            lib.avg_pool1d(
                x.ctypes.data_as(_f32p),
                out.ctypes.data_as(_f32p),
                _ci(B), _ci(C), _ci(L),
                _ci(kernel_size), _ci(stride),
                _ci(output_size),
                _ci(1 if count_include_pad else 0),
                _ci(padding), _ci(orig_L),
            )
        else:
            patch_indices = np.arange(output_size)[:, None] * stride + np.arange(kernel_size)[None, :]
            patches = x[:, :, patch_indices]
            if count_include_pad:
                out = patches.mean(axis=-1)
            else:
                mask = np.zeros_like(x)
                mask[:, :, padding:padding + orig_L] = 1.0
                mask_patches = mask[:, :, patch_indices]
                counts = mask_patches.sum(axis=-1)
                out = patches.sum(axis=-1) / counts

        global _no_grad
        if not _no_grad:
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.orig_L = orig_L
            self.shape = x.shape
            self.count_include_pad = count_include_pad

        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        B, C, L_out = grad_output.shape
        B, C, L = self.shape

        grad_x = np.zeros(self.shape, dtype=grad_output.dtype)
        lib = _load_c_extension()
        if lib is not None and grad_output.dtype == np.float32:
            lib.avg_pool1d_backward(
                grad_output.ctypes.data_as(_f32p),
                grad_x.ctypes.data_as(_f32p),
                _ci(B), _ci(C), _ci(L),
                _ci(self.kernel_size), _ci(self.stride),
                _ci(L_out),
                _ci(1 if self.count_include_pad else 0),
                _ci(self.padding), _ci(self.orig_L),
            )
        else:
            patch_indices = np.arange(L_out)[:, None] * self.stride + np.arange(self.kernel_size)[None, :]
            if self.count_include_pad:
                divisor = self.kernel_size
            else:
                mask = np.zeros_like(grad_x)
                mask[:, :, self.padding:self.padding + self.orig_L] = 1.0
                mask_patches = mask[:, :, patch_indices]
                divisor = mask_patches.sum(axis=-1, keepdims=True)
            scaled_grad = (grad_output / divisor)[:, :, :, None]
            np.add.at(grad_x, (slice(None), slice(None), patch_indices), scaled_grad)

        if self.padding > 0:
            grad_x = grad_x[:, :, self.padding:-self.padding]
        return (grad_x,)


class AvgPool2d(Function):
    def forward(
        self,
        x: np.ndarray,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> np.ndarray:
        assert x.ndim == 4
        B, C, H, W = x.shape

        if isinstance(kernel_size, int):
            Kh = Kw = kernel_size
        else:
            Kh, Kw = kernel_size
        if stride is None:
            Sh, Sw = Kh, Kw
        elif isinstance(stride, int):
            Sh = Sw = stride
        else:
            Sh, Sw = stride
        if isinstance(padding, int):
            Ph = Pw = padding
        else:
            Ph, Pw = padding

        orig_H, orig_W = H, W
        H_out = compute_pooling_output_size(H, Kh, Sh, Ph, 1, ceil_mode)
        W_out = compute_pooling_output_size(W, Kw, Sw, Pw, 1, ceil_mode)

        if Ph > 0 or Pw > 0:
            padded_x = np.zeros((B, C, H + 2 * Ph, W + 2 * Pw), dtype=x.dtype)
            padded_x[:, :, Ph:Ph + H, Pw:Pw + W] = x
            x = padded_x
            H = H + 2 * Ph
            W = W + 2 * Pw

        lib = _load_c_extension()
        if lib is not None and x.dtype == np.float32:
            x = np.ascontiguousarray(x)
            out = np.empty((B, C, H_out, W_out), dtype=x.dtype)
            lib.avg_pool2d(
                x.ctypes.data_as(_f32p),
                out.ctypes.data_as(_f32p),
                _ci(B), _ci(C), _ci(H), _ci(W),
                _ci(Kh), _ci(Kw), _ci(Sh), _ci(Sw),
                _ci(H_out), _ci(W_out),
                _ci(1 if count_include_pad else 0),
                _ci(Ph), _ci(Pw), _ci(orig_H), _ci(orig_W),
            )
        else:
            h_idx = np.arange(H_out)[:, None] * Sh + np.arange(Kh)[None, :]
            w_idx = np.arange(W_out)[:, None] * Sw + np.arange(Kw)[None, :]
            patches = x[:, :, h_idx[:, :, None, None], w_idx[None, None, :, :]]
            patches = patches.transpose(0, 1, 2, 4, 3, 5)
            if count_include_pad:
                out = patches.mean(axis=(-2, -1))
            else:
                mask = np.zeros((1, 1, H, W))
                mask[:, :, Ph:Ph + orig_H, Pw:Pw + orig_W] = 1.0
                mask_patches = mask[:, :, h_idx[:, :, None, None], w_idx[None, None, :, :]]
                mask_patches = mask_patches.transpose(0, 1, 2, 4, 3, 5)
                counts = mask_patches.sum(axis=(-2, -1))
                out = patches.sum(axis=(-2, -1)) / counts

        global _no_grad
        if not _no_grad:
            self.Kh, self.Kw = Kh, Kw
            self.Sh, self.Sw = Sh, Sw
            self.Ph, self.Pw = Ph, Pw
            self.orig_H, self.orig_W = orig_H, orig_W
            self.shape = x.shape
            self.count_include_pad = count_include_pad

        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        B, C, H_out, W_out = grad_output.shape
        B, C, H, W = self.shape

        grad_x = np.zeros(self.shape, dtype=grad_output.dtype)
        lib = _load_c_extension()
        if lib is not None and grad_output.dtype == np.float32:
            lib.avg_pool2d_backward(
                grad_output.ctypes.data_as(_f32p),
                grad_x.ctypes.data_as(_f32p),
                _ci(B), _ci(C), _ci(H), _ci(W),
                _ci(self.Kh), _ci(self.Kw), _ci(self.Sh), _ci(self.Sw),
                _ci(H_out), _ci(W_out),
                _ci(1 if self.count_include_pad else 0),
                _ci(self.Ph), _ci(self.Pw), _ci(self.orig_H), _ci(self.orig_W),
            )
        else:
            h_idx = np.arange(H_out)[:, None] * self.Sh + np.arange(self.Kh)[None, :]
            w_idx = np.arange(W_out)[:, None] * self.Sw + np.arange(self.Kw)[None, :]
            if self.count_include_pad:
                divisor = self.Kh * self.Kw
            else:
                mask = np.zeros((1, 1, H, W))
                mask[:, :, self.Ph:self.Ph + self.orig_H, self.Pw:self.Pw + self.orig_W] = 1.0
                mask_patches = mask[:, :, h_idx[:, :, None, None], w_idx[None, None, :, :]]
                mask_patches = mask_patches.transpose(0, 1, 2, 4, 3, 5)
                divisor = mask_patches.sum(axis=(-2, -1))

            scaled_grad = (grad_output / divisor)[:, :, :, :, None, None]
            h_full = h_idx[:, :, None, None]
            w_full = w_idx[None, None, :, :]
            scaled_grad = scaled_grad.transpose(0, 1, 2, 4, 3, 5)
            np.add.at(grad_x, (slice(None), slice(None), h_full, w_full), scaled_grad)

        if self.Ph > 0 or self.Pw > 0:
            grad_x = grad_x[:, :, self.Ph:H - self.Ph, self.Pw:W - self.Pw]
        return (grad_x,)


# =============================================================================
# Adaptive Pooling Function Classes
# =============================================================================

class AdaptiveMaxPool2d(Function):
    """
    Adaptive Max Pooling 2D functional operation.

    Automatically computes kernel size and stride to achieve target output size.
    """

    def forward(
        self,
        x: np.ndarray,
        output_size: Union[int, Tuple[int, int]],
    ) -> np.ndarray:
        """
        Compute adaptive max pooling.

        Args:
            x: Input (batch_size, channels, height, width)
            output_size: Target output size (h, w) or single int for square
            return_indices: Return indices of max values

        Returns:
            Pooled output with target size, and optionally indices
        """
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        output_h, output_w = output_size

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """Compute gradient for adaptive max pooling."""
        raise NotImplementedError("TODO: Implement AdaptiveMaxPool2d backward")

class AdaptiveAvgPool2d(Function):
    def forward(self, x: np.ndarray, output_size) -> np.ndarray:
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        B, C, H, W = x.shape
        H_out, W_out = output_size

        global _no_grad
        lib = _load_c_extension()

        if lib is not None and x.dtype == np.float32:
            x = np.ascontiguousarray(x)
            out = np.empty((B, C, H_out, W_out), dtype=x.dtype)
            lib.adaptive_avg_pool2d(
                x.ctypes.data_as(_f32p),
                out.ctypes.data_as(_f32p),
                _ci(B), _ci(C), _ci(H), _ci(W),
                _ci(H_out), _ci(W_out),
            )
        elif H_out == 1 and W_out == 1:
            out = x.mean(axis=(2, 3), keepdims=True)
        else:
            out = np.empty((B, C, H_out, W_out), dtype=x.dtype)
            for i in range(H_out):
                for j in range(W_out):
                    h_s, h_e = i * H // H_out, (i + 1) * H // H_out
                    w_s, w_e = j * W // W_out, (j + 1) * W // W_out
                    out[:, :, i, j] = x[:, :, h_s:h_e, w_s:w_e].mean(axis=(2, 3))

        if not _no_grad:
            self.input_shape = x.shape

        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        B, C, H, W = self.input_shape
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]

        grad_input = np.zeros(self.input_shape, dtype=grad_output.dtype)
        lib = _load_c_extension()

        if lib is not None and grad_output.dtype == np.float32:
            lib.adaptive_avg_pool2d_backward(
                grad_output.ctypes.data_as(_f32p),
                grad_input.ctypes.data_as(_f32p),
                _ci(B), _ci(C), _ci(H), _ci(W),
                _ci(H_out), _ci(W_out),
            )
        elif H_out == 1 and W_out == 1:
            grad_input[:] = grad_output / (H * W)
        else:
            for i in range(H_out):
                for j in range(W_out):
                    h_s, h_e = i * H // H_out, (i + 1) * H // H_out
                    w_s, w_e = j * W // W_out, (j + 1) * W // W_out
                    count = (h_e - h_s) * (w_e - w_s)
                    grad_input[:, :, h_s:h_e, w_s:w_e] += (
                        grad_output[:, :, i, j][:, :, None, None] / count
                    )

        return (grad_input,)

# =============================================================================
# Global Pooling Function Classes
# =============================================================================

class GlobalAvgPool1d(Function):
    """
    Global Average Pooling !D functional operation.

    Reduces spatial dimensions to 1x1 by taking mean.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3
        global _no_grad
        if not _no_grad:
            self.shape = x.shape
        return x.mean(axis=-1)  # B, C

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        L = self.shape[2]
        grad_x = grad_output[:, :, None] / (L)
        return (np.broadcast_to(grad_x, self.shape).copy(),)


class GlobalMaxPool1d(Function):
    """
    Global Max Pooling 2D functional operation.

    Reduces spatial dimensions to 1x1 by taking max.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3
        B, C, L = x.shape
        output = x.max(axis=2)  # B, C

        global _no_grad
        if not _no_grad:
            self.shape = x.shape
            self.indices = x.argmax(axis=2)

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        grad_x = np.zeros(self.shape, dtype=grad_output.dtype)
        b_idx = np.arange(grad_output.shape[0])[:, None]
        c_idx = np.arange(grad_output.shape[1])[None, :]
        grad_x[b_idx, c_idx, self.indices] = grad_output
        return (grad_x,)

class GlobalAvgPool2d(Function):
    """
    Global Average Pooling 2D functional operation.

    Reduces spatial dimensions to 1x1 by taking mean.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 4
        global _no_grad
        if not _no_grad:
            self.shape = x.shape
        return x.mean(axis=(2, 3))  # B, C

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        H, W = self.shape[2], self.shape[3]
        grad_x = grad_output[:, :, None, None] / (H * W)
        return (np.broadcast_to(grad_x, self.shape).copy(),)


class GlobalMaxPool2d(Function):
    """
    Global Max Pooling 2D functional operation.

    Reduces spatial dimensions to 1x1 by taking max.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 4
        B, C, H, W = x.shape
        output = x.max(axis=(2, 3))  # B, C

        global _no_grad
        if not _no_grad:
            self.shape = x.shape
            flat = x.reshape(B, C, -1)
            flat_idx = flat.argmax(axis=2)  # B, C
            self.h_indices = flat_idx // W
            self.w_indices = flat_idx % W

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        grad_x = np.zeros(self.shape, dtype=grad_output.dtype)
        b_idx = np.arange(grad_output.shape[0])[:, None]
        c_idx = np.arange(grad_output.shape[1])[None, :]
        grad_x[b_idx, c_idx, self.h_indices, self.w_indices] = grad_output
        return (grad_x,)

