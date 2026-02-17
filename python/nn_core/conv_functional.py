"""
Convolutional Functional Operations (Optimized — Zero-Alloc Hot Path)
=====================================================================

Performance optimizations:
    1. C-accelerated im2col / col2im (2-5× faster)
    2. 1×1 convolution fast path — skips im2col entirely
    3. Full im2col + single BLAS matmul (BLAS loves one big multiply)
    4. groups=1 fast path — no loop overhead for 99% of layers
    5. **PRE-ALLOCATED BUFFERS** — zero heap allocation on hot path:
       - Padding buffer: allocated once, interior-copy only per forward
       - im2col buffer: allocated once, C kernel writes into it
       - matmul output buffer: allocated once, np.matmul writes into it
       - col2im buffer: allocated once per backward
       Eliminates ~4GB of malloc/mmap per iteration for AlexNet.
    6. Auto-compiles C extension on first import; pure-numpy fallback

Function Classes:
    - Conv1d: 1D convolution functional
    - Conv2d: 2D convolution functional (optimized)
    - ConvTranspose2d: 2D transposed convolution functional
    - DepthwiseConv2d: Depthwise 2D convolution functional

Helper Functions:
    - im2col_2d: Image to column transformation (C-accelerated)
    - col2im_2d: Column to image transformation (C-accelerated)
"""
import numpy as np
from typing import List, Tuple, Union, Optional

from python.foundations import Function, convert_to_function, _no_grad

# =============================================================================
# C Extension — loaded from central _c_lib module
# =============================================================================

from ._c_lib import get_c_lib as _load_c_extension, F32P as _f32p, F64P as _f64p, CI as _ci
# =============================================================================
# Buffer Management — the core of the zero-alloc strategy
# =============================================================================

def _get_buf(obj, name, shape, dtype):
    """
    Lazy buffer: return existing if shape/dtype match, else allocate + cache.
    On steady-state iterations this is a dict lookup + shape compare = ~0 cost.
    """
    buf = getattr(obj, name, None)
    if buf is None or buf.shape != shape or buf.dtype != dtype:
        buf = np.empty(shape, dtype=dtype)
        setattr(obj, name, buf)
    return buf


def _get_pad_buf(obj, name, shape, dtype):
    """
    Lazy padded-input buffer. Zeroed on first allocation only.
    The padding region (border) stays zero forever because only the
    interior is overwritten each forward call.
    """
    buf = getattr(obj, name, None)
    if buf is None or buf.shape != shape or buf.dtype != dtype:
        buf = np.zeros(shape, dtype=dtype)
        setattr(obj, name, buf)
    return buf


# =============================================================================
# Helper Functions
# =============================================================================


def im2col_1d(
    x: np.ndarray,
    kernel: int,
    stride: int,
    dilation: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Image to column transformation.

    If `out` is provided, writes directly into it (zero allocation on hot path).
    Otherwise allocates a new array.

    Accepts pre padded input

    Returns
    -------
    cols : (B, C*K, L_out) — always C-contiguous
    """
    B, C, L = x.shape
    L_out = (L - dilation * (kernel - 1) - 1) // stride + 1
    CK = C * kernel

    lib = _load_c_extension()
    if lib is not None:
        x = np.ascontiguousarray(x)
        cols = out if out is not None else np.empty((B, CK, L_out), dtype=x.dtype)
        if x.dtype == np.float32:
            lib.im2col1d_f32(
                x.ctypes.data_as(_f32p), cols.ctypes.data_as(_f32p),
                B, C, L, kernel, stride, dilation, L_out,
            )
        else:
            lib.im2col1d_f64(
                x.ctypes.data_as(_f64p), cols.ctypes.data_as(_f64p),
                B, C, L, kernel, stride, dilation, L_out,
            )
        return cols

    # pure numpy fallback
    sB, sC, sL = x.strides
    cols = np.lib.stride_tricks.as_strided(
        x, shape=(B, C, L_out, kernel), strides = (sB, sC, sL * stride, sL * dilation), writeable=False)

    cols = cols.transpose((0, 1, 3, 2))
    cols = cols.reshape((B, -1, L_out))
    if out is not None:
        np.copyto(out, cols)
        return out

    return cols


def im2col_2d(
    x: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Image to column transformation.

    If `out` is provided, writes directly into it (zero allocation on hot path).
    Otherwise allocates a new array.

    Accepts pre-padded input

    Returns
    -------
    cols : (B, C*kh*kw, H_out*W_out) — always C-contiguous
    """
    B, C, H, W = x.shape
    sh, sw = stride
    dh, dw = dilation
    H_out, W_out = calculate_output_shape(
        (H, W), (kernel_h, kernel_w), stride, (0,0), dilation
    )
    N = H_out * W_out
    CKK = C * kernel_h * kernel_w

    lib = _load_c_extension()
    if lib is not None:
        x = np.ascontiguousarray(x)
        cols = out if out is not None else np.empty((B, CKK, N), dtype=x.dtype)
        if x.dtype == np.float32:
            lib.im2col_f32(
                x.ctypes.data_as(_f32p), cols.ctypes.data_as(_f32p),
                B, C, H, W, kernel_h, kernel_w, sh, sw, dh, dw, H_out, W_out,
            )
        else:
            lib.im2col_f64(
                x.ctypes.data_as(_f64p), cols.ctypes.data_as(_f64p),
                B, C, H, W, kernel_h, kernel_w, sh, sw, dh, dw, H_out, W_out,
            )
        return cols

    # Pure-numpy fallback
    sB, sC, sH, sW = x.strides
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(B, C, kernel_h, kernel_w, H_out, W_out),
        strides=(sB, sC, dh * sH, dw * sW, sh * sH, sw * sW),
        writeable=False,
    )
    if out is not None:
        np.copyto(out, patches.reshape(B, CKK, N))
        return out
    return patches.reshape(B, CKK, N)


def col2im_1d(
    cols: np.ndarray,
    x_shape: Tuple[int, ...], # padded shape
    kernel: int,
    stride: int,
    dilation: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Scatter columns back to an image (inverse of im2col).

    If `out` is provided, zeros it and accumulates into it (zero allocation).
    Otherwise allocates a new zeroed array.

    outputs padded input

    """
    B, C, L = x_shape
    B_cols, CK, L_out = cols.shape

    lib = _load_c_extension()
    if lib is not None:
        cols = np.ascontiguousarray(cols)
        if out is not None:
            x = out
            x[:] = 0
        else:
            x = np.zeros((B, C, L), dtype=cols.dtype)
        if cols.dtype == np.float32:
            lib.col2im1d_f32(
                cols.ctypes.data_as(_f32p), x.ctypes.data_as(_f32p),
                B, C, L, kernel, stride, dilation, L_out,
            )
        else:
            lib.col2im1d_f64(
                cols.ctypes.data_as(_f64p), x.ctypes.data_as(_f64p),
                B, C, L, kernel, stride, dilation, L_out,
            )
        return x

    # pure numpy fallback
    if out is not None:
        x_padded = out
        x_padded[:] = 0
    else:
        x_padded = np.zeros((B, C, L), dtype=cols.dtype)

    cols_reshaped = cols.reshape(B, C, kernel, L_out)
    for k in range(kernel):
        k_offset = k * dilation
        x_padded[:, :, k_offset : k_offset + stride * L_out : stride] += cols_reshaped[:, :, k, :]

    return x_padded

def col2im_2d(
    cols: np.ndarray,
    x_shape: Tuple[int, ...],
    kernel_h: int,
    kernel_w: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Scatter columns back to an image (inverse of im2col).

    If `out` is provided, zeros it and accumulates into it (zero allocation).
    Otherwise allocates a new zeroed array.

    Returns padded output!
    """
    B, C, H, W = x_shape
    sh, sw = stride
    dh, dw = dilation
    H_out, W_out = calculate_output_shape(
        (H, W), (kernel_h, kernel_w), stride, (0, 0), dilation
    )

    H_p = H
    W_p = W

    lib = _load_c_extension()
    if lib is not None:
        cols = np.ascontiguousarray(cols)
        if out is not None:
            x = out
            x[:] = 0
        else:
            x = np.zeros((B, C, H_p, W_p), dtype=cols.dtype)
        if cols.dtype == np.float32:
            lib.col2im_f32(
                cols.ctypes.data_as(_f32p), x.ctypes.data_as(_f32p),
                B, C, H_p, W_p, kernel_h, kernel_w, sh, sw, dh, dw, H_out, W_out,
            )
        else:
            lib.col2im_f64(
                cols.ctypes.data_as(_f64p), x.ctypes.data_as(_f64p),
                B, C, H_p, W_p, kernel_h, kernel_w, sh, sw, dh, dw, H_out, W_out,
            )
        return x

    # Pure-numpy fallback
    if out is not None:
        x = out
        x[:] = 0
    else:
        x = np.zeros((B, C, H_p, W_p), dtype=cols.dtype)
    cols = cols.reshape(B, C, kernel_h, kernel_w, H_out, W_out)
    for kh_i in range(kernel_h):
        h_start = kh_i * dh
        h_end   = h_start + sh * H_out
        for kw_i in range(kernel_w):
            w_start = kw_i * dw
            w_end   = w_start + sw * W_out
            x[:, :, h_start:h_end:sh, w_start:w_end:sw] += cols[:, :, kh_i, kw_i, :, :]

    return x



def calculate_output_shape(
    input_shape: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1)
) -> Tuple[int, int]:
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
    k_h, k_w = kernel_size
    standard = out_channels * in_channels * k_h * k_w
    if bias:
        standard += out_channels
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
    1D Convolution — optimized autograd Function with zero-alloc hot path.

    Mirrors all Conv2d optimizations (1D equivalents):
      - C-accelerated im2col_1d / col2im_1d
      - 1×1 (K=1) fast path — skips im2col entirely
      - Pre-allocated buffers (zero heap allocation on steady-state)
      - Deep vs early layer GEMM strategy for grad_weight
      - Grouped convolution support

    Pre-allocated buffers:
      _x_pad_buf   — padded input (zeroed once, interior overwritten)
      _cols_buf    — im2col output
      _fwd_out_buf — matmul output
      _Wmat_T      — transposed weight for backward
      _gw_buf      — intermediate grad_weight (batched)
      _gw_sum_buf  — summed grad_weight
      _grad_x_buf  — col2im output in backward
    """

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert x.ndim == 3
        global _no_grad

        B, C, L = x.shape
        C_out, _, K = weight.shape
        s = stride
        p = padding
        d = dilation

        # --- Pad into pre-allocated buffer ---
        if p > 0:
            Lp = L + 2 * p
            x_padded = _get_pad_buf(self, '_x_pad_buf', (B, C, Lp), x.dtype)
            x_padded[:, :, p:p+L] = x
            x = x_padded
        else:
            Lp = L

        L_out = (Lp - d * (K - 1) - 1) // s + 1

        # =====================================================================
        # FAST PATH: groups == 1 (covers ~99% of conv layers)
        # =====================================================================
        if groups == 1:
            Wmat = weight.reshape(C_out, -1)  # (C_out, C*K)
            CK = C * K

            # --- 1×1 fast path: skip im2col entirely ---
            if K == 1 and d == 1:
                if s == 1:
                    cols = x.reshape(B, C, L_out)
                else:
                    cols = x[:, :, ::s].reshape(B, C, L_out)
                out_buf = _get_buf(self, '_fwd_out_buf', (B, C_out, L_out), x.dtype)
                np.matmul(Wmat, cols, out=out_buf)
                out = out_buf
            else:
                # --- General path: im2col into pre-allocated buffer ---
                cols_buf = _get_buf(self, '_cols_buf', (B, CK, L_out), x.dtype)
                cols = im2col_1d(x, K, s, d, out=cols_buf)

                out_buf = _get_buf(self, '_fwd_out_buf', (B, C_out, L_out), x.dtype)
                np.matmul(Wmat, cols, out=out_buf)
                out = out_buf

            if bias is not None:
                out += bias[None, :, None]

            if not _no_grad:
                self.cols = cols
                self.weight = weight
                self._cols_owned = (K != 1)

                # Pre-allocate backward buffers
                wt_shape = (CK, C_out)
                if not hasattr(self, '_Wmat_T') or self._Wmat_T.shape != wt_shape:
                    self._Wmat_T = np.empty(wt_shape, dtype=x.dtype)
                np.copyto(self._Wmat_T, Wmat.T)

                gw_shape = (B, C_out, CK)
                if not hasattr(self, '_gw_buf') or self._gw_buf.shape != gw_shape:
                    self._gw_buf = np.empty(gw_shape, dtype=x.dtype)
                if not hasattr(self, '_gw_sum_buf') or self._gw_sum_buf.shape != (C_out, CK):
                    self._gw_sum_buf = np.empty((C_out, CK), dtype=x.dtype)

                # Deep layer: single big GEMM via batch flattening
                if L_out <= 64 and L_out < CK:
                    go_shape = (C_out, B * L_out)
                    cf_shape = (CK, B * L_out)
                    if not hasattr(self, '_go_flat_buf') or self._go_flat_buf.shape != go_shape:
                        self._go_flat_buf = np.empty(go_shape, dtype=x.dtype)
                    if not hasattr(self, '_cols_flat_buf') or self._cols_flat_buf.shape != cf_shape:
                        self._cols_flat_buf = np.empty(cf_shape, dtype=x.dtype)

                self.x_padded_shape = (B, C, Lp)
                self.stride = s
                self.padding = p
                self.dilation = d
                self.groups = 1
                self.has_bias = bias is not None
                self.K = K

            return out

        # =====================================================================
        # GROUPED CONVOLUTION (depthwise, etc.)
        # =====================================================================
        Cg = C // groups
        Cog = C_out // groups

        outputs = []
        cols_list = []

        for g in range(groups):
            xg = x[:, g * Cg : (g + 1) * Cg]
            wg = weight[g * Cog : (g + 1) * Cog]
            Wmat = wg.reshape(Cog, -1)

            if K == 1 and d == 1:
                if s == 1:
                    cols = xg.reshape(B, Cg, L_out)
                else:
                    cols = xg[:, :, ::s].reshape(B, Cg, L_out)
            else:
                cols = im2col_1d(xg, K, s, d)

            outg = np.matmul(Wmat, cols)
            outputs.append(outg)
            cols_list.append(cols)

        out = np.concatenate(outputs, axis=1)

        if bias is not None:
            out += bias[None, :, None]

        if not _no_grad:
            self.cols_list = cols_list
            self.weight = weight
            self.x_padded_shape = (B, C, Lp)
            self.stride = s
            self.padding = p
            self.dilation = d
            self.groups = groups
            self.has_bias = bias is not None
            self.K = K

        return out

    def backward(self, grad_output):
        B, C_out, L_out = grad_output.shape
        groups = self.groups
        K = self.K
        B, C, Lp = self.x_padded_shape

        # =================================================================
        # FAST PATH: groups == 1
        # =================================================================
        if groups == 1:
            Wmat = self.weight.reshape(C_out, -1)
            cols = self.cols
            CK = cols.shape[1]
            go = grad_output  # already (B, C_out, L_out)

            if L_out <= 64 and L_out < CK:
                # Deep layers: single big GEMM
                go_flat = self._go_flat_buf
                cols_flat = self._cols_flat_buf
                np.copyto(go_flat.reshape(C_out, B, L_out), go.transpose(1, 0, 2))
                np.copyto(cols_flat.reshape(CK, B, L_out), cols.transpose(1, 0, 2))
                grad_weight = (go_flat @ cols_flat.T).reshape(self.weight.shape)
                np.matmul(self._Wmat_T, go_flat, out=cols_flat)
                np.copyto(cols, cols_flat.reshape(CK, B, L_out).transpose(1, 0, 2))
                grad_cols = cols
            else:
                # Early layers: batched matmul + sum
                np.matmul(go, cols.transpose(0, 2, 1), out=self._gw_buf)
                np.sum(self._gw_buf, axis=0, out=self._gw_sum_buf)
                grad_weight = self._gw_sum_buf.reshape(self.weight.shape)
                if self._cols_owned:
                    np.matmul(self._Wmat_T, go, out=cols)
                    grad_cols = cols
                else:
                    grad_cols = np.matmul(self._Wmat_T, go)

            # col2im: grad_cols → grad_x_padded
            if K == 1:
                s = self.stride
                if s == 1:
                    grad_x = grad_cols.reshape(B, C, Lp)
                else:
                    grad_x = _get_buf(self, '_grad_x_buf', (B, C, Lp), grad_cols.dtype)
                    grad_x[:] = 0
                    grad_x[:, :, ::s] = grad_cols.reshape(B, C, L_out)
            else:
                grad_x_buf = _get_buf(self, '_grad_x_buf', (B, C, Lp), grad_cols.dtype)
                grad_x = col2im_1d(
                    grad_cols,
                    (B, C, Lp),
                    K,
                    self.stride,
                    self.dilation,
                    out=grad_x_buf,
                )

            # Strip padding (view, no alloc)
            p = self.padding
            if p > 0:
                grad_x = grad_x[:, :, p : Lp - p]

            grad_bias = grad_output.sum(axis=(0, 2)) if self.has_bias else None
            return grad_x, grad_weight, grad_bias

        # =================================================================
        # GROUPED CONVOLUTION backward
        # =================================================================
        Cg = C // groups
        Cog = C_out // groups

        grad_weight = np.zeros_like(self.weight, dtype=grad_output.dtype)
        grad_x = np.zeros((B, C, Lp), dtype=grad_output.dtype)

        for g in range(groups):
            go = grad_output[:, g * Cog : (g + 1) * Cog]
            wg = self.weight[g * Cog : (g + 1) * Cog]
            Wmat = wg.reshape(Cog, -1)
            cols = self.cols_list[g]
            CKg = cols.shape[1]

            if L_out <= 64 and L_out < CKg:
                # Deep layers: single big GEMM
                go_flat = np.ascontiguousarray(go.transpose(1, 0, 2)).reshape(Cog, B * L_out)
                cols_flat = np.ascontiguousarray(cols.transpose(1, 0, 2)).reshape(CKg, B * L_out)
                gW = go_flat @ cols_flat.T
                grad_cols = np.ascontiguousarray(
                    (Wmat.T @ go_flat).reshape(CKg, B, L_out).transpose(1, 0, 2)
                )
            else:
                gW = np.matmul(go, cols.transpose(0, 2, 1)).sum(axis=0)
                grad_cols = np.matmul(Wmat.T, go)
            grad_weight[g * Cog : (g + 1) * Cog] = gW.reshape(wg.shape)

            if K == 1:
                s = self.stride
                if s == 1:
                    grad_x[:, g * Cg : (g + 1) * Cg] = grad_cols.reshape(B, Cg, Lp)
                else:
                    grad_x[:, g * Cg : (g + 1) * Cg, ::s] = grad_cols.reshape(B, Cg, L_out)
            else:
                gx = col2im_1d(
                    grad_cols,
                    (B, Cg, Lp),
                    K,
                    self.stride,
                    self.dilation,
                )
                grad_x[:, g * Cg : (g + 1) * Cg] = gx

        p = self.padding
        if p > 0:
            grad_x = grad_x[:, :, p : Lp - p]

        grad_bias = grad_output.sum(axis=(0, 2)) if self.has_bias else None

        return grad_x, grad_weight, grad_bias


class ConvTranspose1d(Function):
    """
    1D Transposed Convolution — optimized autograd Function.

    Exploits the duality between Conv1d and ConvTranspose1d:
      Forward:   GEMM (W^T @ x_flat)  →  col2im_1d  →  crop  →  output
      Backward:  im2col_1d(grad)  →  GEMM (W @ grad_cols → grad_x)
                                      GEMM (x @ grad_cols^T → grad_W)

    Reuses the same C-accelerated im2col_1d / col2im_1d kernels as Conv1d.

    Weight shape: (C_in, C_out_per_group, K) — matches PyTorch convention.

    Pre-allocated buffers (reused across calls, zero alloc on steady state):
      _cols_buf      — GEMM output (W^T @ x) before col2im
      _col2im_buf    — col2im scatter target (full output before crop)
      _grad_full_buf — unpadded grad for backward's im2col
      _im2col_buf    — im2col output in backward
      _gw_buf        — intermediate for grad_weight (batched)
      _gw_sum_buf    — summed grad_weight
      _grad_x_buf    — grad_input from GEMM
    """

    def forward(self, x, weight, bias=None, stride=1, padding=0,
                output_padding=0, dilation=1, groups=1):
        assert x.ndim == 3
        global _no_grad

        B, C_in, L_in = x.shape
        # Weight shape: (C_in, C_out_per_group, K)
        _, C_out_g, K = weight.shape
        s = stride
        p = padding
        op = output_padding
        d = dilation

        C_out = C_out_g * groups

        # Output length: L_out = (L_in - 1)*s - 2*p + d*(K-1) + op + 1
        L_out = (L_in - 1) * s - 2 * p + d * (K - 1) + op + 1

        # Full col2im target size (before padding crop):
        #   L_col = (L_in - 1)*s + d*(K-1) + 1
        L_col = (L_in - 1) * s + d * (K - 1) + 1

        # =================================================================
        # FAST PATH: groups == 1
        # =================================================================
        if groups == 1:
            CK = C_out * K
            W_mat = weight.reshape(C_in, CK)       # (C_in, C_out*K)
            x_flat = x                              # already (B, C_in, L_in)

            # --- Step 1: GEMM   cols = W^T @ x_flat  →  (B, CK, L_in) ---
            cols_buf = _get_buf(self, '_cols_buf', (B, CK, L_in), x.dtype)
            np.matmul(W_mat.T, x_flat, out=cols_buf)

            # --- Step 2: col2im_1d  cols → spatial output ---
            if K == 1 and d == 1:
                # K=1 fast path: col2im is just scatter at stride intervals
                if s == 1:
                    out = cols_buf.reshape(B, C_out, L_in)
                else:
                    out = _get_buf(self, '_col2im_buf', (B, C_out, L_col), x.dtype)
                    out[:] = 0
                    out[:, :, ::s] = cols_buf.reshape(B, C_out, L_in)
            else:
                # General path: C-accelerated col2im_1d
                col2im_buf = _get_buf(self, '_col2im_buf', (B, C_out, L_col), x.dtype)
                col2im_1d(
                    cols_buf, (B, C_out, L_col),
                    K, s, d, out=col2im_buf,
                )
                out = col2im_buf

            # --- Step 3: Crop padding ---
            if p > 0:
                out = out[:, :, p : L_col - p]

            # --- Step 4: Handle output_padding (rare, usually 0) ---
            if op > 0:
                out_final = np.zeros((B, C_out, L_out), dtype=x.dtype)
                out_final[:, :, :out.shape[2]] = out
                out = out_final

            # --- Step 5: Bias ---
            if bias is not None:
                out = np.ascontiguousarray(out) if not out.flags['C_CONTIGUOUS'] else out
                out += bias[None, :, None]

            # --- Save for backward ---
            if not _no_grad:
                self.x_flat = x_flat
                self.weight = weight
                self._W_mat = W_mat
                self.stride = s
                self.padding = p
                self.output_padding = op
                self.dilation = d
                self.groups = 1
                self.has_bias = bias is not None
                self.K = K
                self.L_in = L_in
                self.L_col = L_col
                self.L_out = L_out
                self.B, self.C_in, self.C_out = B, C_in, C_out

                # Pre-allocate backward buffers
                gw_shape = (B, C_in, CK)
                if not hasattr(self, '_gw_buf') or self._gw_buf.shape != gw_shape:
                    self._gw_buf = np.empty(gw_shape, dtype=x.dtype)
                gw_sum_shape = (C_in, CK)
                if not hasattr(self, '_gw_sum_buf') or self._gw_sum_buf.shape != gw_sum_shape:
                    self._gw_sum_buf = np.empty(gw_sum_shape, dtype=x.dtype)

                # Deep layer optimization buffers
                if L_in <= 64 and L_in < C_in:
                    cin_bl = (C_in, B * L_in)
                    ck_bl = (CK, B * L_in)
                    if not hasattr(self, '_xf_flat_buf') or self._xf_flat_buf.shape != cin_bl:
                        self._xf_flat_buf = np.empty(cin_bl, dtype=x.dtype)
                    if not hasattr(self, '_gc_flat_buf') or self._gc_flat_buf.shape != ck_bl:
                        self._gc_flat_buf = np.empty(ck_bl, dtype=x.dtype)

            return out

        # =================================================================
        # GROUPED CONVOLUTION
        # =================================================================
        C_in_g = C_in // groups
        outputs = []
        x_flat_list = []

        for g in range(groups):
            xg = x[:, g * C_in_g : (g + 1) * C_in_g]
            wg = weight[g * C_in_g : (g + 1) * C_in_g]
            W_mat_g = wg.reshape(C_in_g, C_out_g * K)
            xg_flat = xg.reshape(B, C_in_g, L_in)

            cols_g = np.matmul(W_mat_g.T, xg_flat)  # (B, C_out_g*K, L_in)

            if K == 1 and d == 1:
                if s == 1:
                    out_g = cols_g.reshape(B, C_out_g, L_in)
                else:
                    out_g = np.zeros((B, C_out_g, L_col), dtype=x.dtype)
                    out_g[:, :, ::s] = cols_g.reshape(B, C_out_g, L_in)
            else:
                out_g = col2im_1d(
                    cols_g, (B, C_out_g, L_col),
                    K, s, d,
                )

            if p > 0:
                out_g = out_g[:, :, p : L_col - p]

            outputs.append(out_g)
            x_flat_list.append(xg_flat)

        out = np.concatenate(outputs, axis=1)

        if op > 0:
            out_final = np.zeros((B, C_out, L_out), dtype=x.dtype)
            out_final[:, :, :out.shape[2]] = out
            out = out_final

        if bias is not None:
            out = np.ascontiguousarray(out) if not out.flags['C_CONTIGUOUS'] else out
            out += bias[None, :, None]

        if not _no_grad:
            self.x_flat_list = x_flat_list
            self.weight = weight
            self.stride = s
            self.padding = p
            self.output_padding = op
            self.dilation = d
            self.groups = groups
            self.has_bias = bias is not None
            self.K = K
            self.L_in = L_in
            self.L_col = L_col
            self.L_out = L_out
            self.B, self.C_in, self.C_out = B, C_in, C_out

        return out

    def backward(self, grad_output):
        B = self.B
        C_in, C_out = self.C_in, self.C_out
        _, C_out_g, K = self.weight.shape
        groups = self.groups
        s = self.stride
        p = self.padding
        op = self.output_padding
        d = self.dilation
        L_in = self.L_in
        L_col = self.L_col
        L_out = self.L_out

        # =================================================================
        # Step 1: Reverse the crop + output_padding to get grad_full
        #         (gradient w.r.t. the full col2im output)
        # =================================================================
        if p > 0 or op > 0:
            grad_full = _get_pad_buf(self, '_grad_full_buf',
                                     (B, C_out, L_col), grad_output.dtype)
            grad_full[:] = 0
            l_copy = L_col - 2 * p     # = L_out - op
            grad_full[:, :, p : L_col - p] = \
                grad_output[:, :, :l_copy]
        else:
            grad_full = grad_output

        # =================================================================
        # FAST PATH: groups == 1
        # =================================================================
        if groups == 1:
            CK = C_out * K
            W_mat = self._W_mat       # (C_in, CK)
            x_flat = self.x_flat      # (B, C_in, L_in)

            # --- Backward of col2im = im2col ---
            if K == 1 and d == 1:
                if s == 1:
                    grad_cols = grad_full.reshape(B, C_out, L_in)
                else:
                    grad_cols = grad_full[:, :, ::s].reshape(B, C_out, L_in)
            else:
                im2col_buf = _get_buf(self, '_im2col_buf', (B, CK, L_in), grad_output.dtype)
                grad_cols = im2col_1d(
                    grad_full, K, s, d, out=im2col_buf,
                )

            # --- grad_weight = x_flat @ grad_cols^T ---
            if L_in <= 64 and L_in < C_in:
                # Deep layers: single big GEMM
                xf = self._xf_flat_buf       # (C_in, B*L_in)
                gcf = self._gc_flat_buf      # (CK, B*L_in)
                np.copyto(xf.reshape(C_in, B, L_in), x_flat.transpose(1, 0, 2))
                np.copyto(gcf.reshape(CK, B, L_in), grad_cols.transpose(1, 0, 2))
                grad_weight = (xf @ gcf.T).reshape(self.weight.shape)
            else:
                # Early layers: batched matmul + sum
                np.matmul(x_flat, grad_cols.transpose(0, 2, 1), out=self._gw_buf)
                np.sum(self._gw_buf, axis=0, out=self._gw_sum_buf)
                grad_weight = self._gw_sum_buf.reshape(self.weight.shape)

            # --- grad_input = W_mat @ grad_cols ---
            grad_x_buf = _get_buf(self, '_grad_x_buf', (B, C_in, L_in), grad_output.dtype)
            np.matmul(W_mat, grad_cols, out=grad_x_buf)
            grad_x = grad_x_buf.reshape(B, C_in, L_in)

            # --- grad_bias ---
            grad_bias = grad_output.sum(axis=(0, 2)) if self.has_bias else None

            return grad_x, grad_weight, grad_bias

        # =================================================================
        # GROUPED CONVOLUTION backward
        # =================================================================
        C_in_g = C_in // groups
        grad_weight = np.zeros_like(self.weight)
        grad_x = np.zeros((B, C_in, L_in), dtype=grad_output.dtype)

        for g in range(groups):
            grad_full_g = grad_full[:, g * C_out_g : (g + 1) * C_out_g]
            wg = self.weight[g * C_in_g : (g + 1) * C_in_g]
            W_mat_g = wg.reshape(C_in_g, C_out_g * K)
            xg_flat = self.x_flat_list[g]
            CK_g = C_out_g * K

            # im2col on this group's grad
            if K == 1 and d == 1:
                if s == 1:
                    grad_cols_g = grad_full_g.reshape(B, C_out_g, L_in)
                else:
                    grad_cols_g = grad_full_g[:, :, ::s].reshape(B, C_out_g, L_in)
            else:
                grad_cols_g = im2col_1d(
                    grad_full_g, K, s, d,
                )

            # grad_weight for this group
            if L_in <= 64 and L_in < C_in_g:
                xf = np.ascontiguousarray(xg_flat.transpose(1, 0, 2)).reshape(C_in_g, B * L_in)
                gcf = np.ascontiguousarray(grad_cols_g.transpose(1, 0, 2)).reshape(CK_g, B * L_in)
                gW = xf @ gcf.T
            else:
                gW = np.matmul(xg_flat, grad_cols_g.transpose(0, 2, 1)).sum(axis=0)
            grad_weight[g * C_in_g : (g + 1) * C_in_g] = gW.reshape(wg.shape)

            # grad_input for this group
            grad_xg = np.matmul(W_mat_g, grad_cols_g)  # (B, C_in_g, L_in)
            grad_x[:, g * C_in_g : (g + 1) * C_in_g] = grad_xg.reshape(B, C_in_g, L_in)

        grad_bias = grad_output.sum(axis=(0, 2)) if self.has_bias else None

        return grad_x, grad_weight, grad_bias


class Conv2d(Function):
    """
    2D Convolution — optimized autograd Function with zero-alloc hot path.

    All large buffers are pre-allocated on first call and reused:
      _x_pad_buf   — padded input (zeroed once, interior overwritten)
      _cols_buf    — im2col output (C kernel writes into it)
      _fwd_out_buf — matmul output (np.matmul writes into it)
      _grad_x_buf  — col2im output in backward
      _Wmat_T      — transposed weight matrix for backward
      _gw_buf      — intermediate for grad_weight computation
      _gw_sum_buf  — summed grad_weight

    On steady-state iterations: zero new heap allocations in forward+backward.
    """

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert x.ndim == 4
        global _no_grad

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        B, C, H, W = x.shape
        C_out, _, kh, kw = weight.shape
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation
        H_out, W_out = calculate_output_shape(
            (H, W), (kh, kw), stride, padding, dilation
        )
        N = H_out * W_out

        # --- Pad into pre-allocated buffer ---
        # _x_pad_buf is zeroed on first allocation only.
        # The padding border stays zero because we only overwrite the interior.
        if padding != (0, 0):
            Hp, Wp = H + 2 * ph, W + 2 * pw
            x_padded = _get_pad_buf(self, '_x_pad_buf', (B, C, Hp, Wp), x.dtype)
            x_padded[:, :, ph:ph+H, pw:pw+W] = x  # only copy interior
            x = x_padded
        else:
            Hp, Wp = H, W

        # =====================================================================
        # FAST PATH: groups == 1 (covers ~99% of conv layers)
        # =====================================================================
        if groups == 1:
            Wmat = weight.reshape(C_out, -1)  # (C_out, C*kh*kw)
            CKK = C * kh * kw

            # --- 1×1 fast path: skip im2col entirely ---
            if kh == 1 and kw == 1 and dh == 1 and dw == 1:
                if sh == 1 and sw == 1:
                    cols = x.reshape(B, C, N)
                else:
                    cols = x[:, :, ::sh, ::sw].reshape(B, C, N)
                # Pre-allocated matmul output
                out_buf = _get_buf(self, '_fwd_out_buf', (B, C_out, N), x.dtype)
                np.matmul(Wmat, cols, out=out_buf)
                out = out_buf
            else:
                # --- General path: im2col into pre-allocated buffer ---
                cols_buf = _get_buf(self, '_cols_buf', (B, CKK, N), x.dtype)
                cols = im2col_2d(x, kh, kw, stride,  dilation, out=cols_buf)

                # Pre-allocated matmul output
                out_buf = _get_buf(self, '_fwd_out_buf', (B, C_out, N), x.dtype)
                np.matmul(Wmat, cols, out=out_buf)
                out = out_buf

            out = out.reshape(B, C_out, H_out, W_out)

            if bias is not None:
                out += bias[None, :, None, None]

            if not _no_grad:
                self.cols = cols
                self.weight = weight
                self._cols_owned = not (kh == 1 and kw == 1)

                # Lazy-allocate backward buffers
                wt_shape = (CKK, C_out)
                if not hasattr(self, '_Wmat_T') or self._Wmat_T.shape != wt_shape:
                    self._Wmat_T = np.empty(wt_shape, dtype=x.dtype)
                np.copyto(self._Wmat_T, Wmat.T)

                gw_shape = (B, C_out, CKK)
                if not hasattr(self, '_gw_buf') or self._gw_buf.shape != gw_shape:
                    self._gw_buf = np.empty(gw_shape, dtype=x.dtype)
                if not hasattr(self, "_gw_sum_buf") or self._gw_sum_buf.shape != (C_out, CKK):
                    self._gw_sum_buf = np.empty((C_out, CKK), dtype=x.dtype)

                if N <= 64 and N < CKK:
                    go_shape = (C_out, B * N)
                    cf_shape = (CKK, B * N)
                    if not hasattr(self, '_go_flat_buf') or self._go_flat_buf.shape != go_shape:
                        self._go_flat_buf = np.empty(go_shape, dtype=x.dtype)
                    if not hasattr(self, '_cols_flat_buf') or self._cols_flat_buf.shape != cf_shape:
                        self._cols_flat_buf = np.empty(cf_shape, dtype=x.dtype)

                self.x_padded_shape = (B, C, Hp, Wp)
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.groups = 1
                self.has_bias = bias is not None
                self.kh, self.kw = kh, kw

            return out

        # =====================================================================
        # GROUPED CONVOLUTION (depthwise, etc.)
        # =====================================================================
        Cg = C // groups
        Cog = C_out // groups

        outputs = []
        cols_list = []

        for g in range(groups):
            xg = x[:, g * Cg : (g + 1) * Cg]
            wg = weight[g * Cog : (g + 1) * Cog]
            Wmat = wg.reshape(Cog, -1)

            if kh == 1 and kw == 1 and dh == 1 and dw == 1:
                if sh == 1 and sw == 1:
                    cols = xg.reshape(B, Cg, N)
                else:
                    cols = xg[:, :, ::sh, ::sw].reshape(B, Cg, N)
            else:
                cols = im2col_2d(xg, kh, kw, stride,  dilation)

            outg = np.matmul(Wmat, cols)
            outputs.append(outg)
            cols_list.append(cols)

        out = np.concatenate(outputs, axis=1).reshape(B, C_out, H_out, W_out)

        if bias is not None:
            out += bias[None, :, None, None]

        if not _no_grad:
            self.cols_list = cols_list
            self.weight = weight
            self.x_padded_shape = (B, C, Hp, Wp)
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.has_bias = bias is not None
            self.kh, self.kw = kh, kw

        return out

    def backward(self, grad_output):
        B, C_out, H_out, W_out = grad_output.shape
        groups = self.groups
        N = H_out * W_out
        kh, kw = self.kh, self.kw
        B, C, Hp, Wp = self.x_padded_shape

        # =================================================================
        # FAST PATH: groups == 1
        # =================================================================
        if groups == 1:
            Wmat = self.weight.reshape(C_out, -1)
            cols = self.cols
            CKK = cols.shape[1]
            go = grad_output.reshape(B, C_out, N)

            if N <= 64 and N < CKK:
                # Deep layers: single big GEMM
                go_flat = self._go_flat_buf
                cols_flat = self._cols_flat_buf
                np.copyto(go_flat.reshape(C_out, B, N), go.transpose(1, 0, 2))
                np.copyto(cols_flat.reshape(CKK, B, N), cols.transpose(1, 0, 2))
                grad_weight = (go_flat @ cols_flat.T).reshape(self.weight.shape)
                np.matmul(self._Wmat_T, go_flat, out=cols_flat)
                np.copyto(cols, cols_flat.reshape(CKK, B, N).transpose(1, 0, 2))
                grad_cols = cols
            else:
                # Early layers: batched matmul, reuse cols buffer
                np.matmul(go, cols.transpose(0, 2, 1), out=self._gw_buf)
                np.sum(self._gw_buf, axis=0, out=self._gw_sum_buf)
                grad_weight = self._gw_sum_buf.reshape(self.weight.shape)
                if self._cols_owned:
                    np.matmul(self._Wmat_T, go, out=cols)
                    grad_cols = cols
                else:
                    grad_cols = np.matmul(self._Wmat_T, go)

            # col2im: grad_cols → grad_x_padded (pre-allocated)
            if kh == 1 and kw == 1:
                sh, sw = self.stride
                if sh == 1 and sw == 1:
                    grad_x = grad_cols.reshape(B, C, Hp, Wp)
                else:
                    grad_x = _get_buf(self, '_grad_x_buf', (B, C, Hp, Wp), grad_cols.dtype)
                    grad_x[:] = 0
                    grad_x[:, :, ::sh, ::sw] = grad_cols.reshape(B, C, H_out, W_out)
            else:
                # Pre-allocated col2im output
                grad_x_buf = _get_buf(self, '_grad_x_buf', (B, C, Hp, Wp), grad_cols.dtype)
                grad_x = col2im_2d(
                    grad_cols,
                    (B, C, Hp, Wp),
                    kh, kw,
                    self.stride,
                    self.dilation,
                    out=grad_x_buf,
                )

            # Strip padding (view, no alloc)
            ph, pw = self.padding
            if ph > 0 or pw > 0:
                grad_x = grad_x[
                    :, :,
                    ph : Hp - ph if ph > 0 else Hp,
                    pw : Wp - pw if pw > 0 else Wp,
                ]

            grad_bias = grad_output.sum(axis=(0, 2, 3)) if self.has_bias else None
            return grad_x, grad_weight, grad_bias

        # =================================================================
        # GROUPED CONVOLUTION backward
        # =================================================================
        Cg = C // groups
        Cog = C_out // groups

        grad_weight = np.zeros_like(self.weight, dtype=grad_output.dtype)
        grad_x = np.zeros((B, C, Hp, Wp), dtype=grad_output.dtype)

        for g in range(groups):
            go = grad_output[:, g * Cog : (g + 1) * Cog].reshape(B, Cog, N)
            wg = self.weight[g * Cog : (g + 1) * Cog]
            Wmat = wg.reshape(Cog, -1)
            cols = self.cols_list[g]
            CKKg = cols.shape[1]

            if N <= 64 and N < CKKg:
                go_flat = np.ascontiguousarray(go.transpose(1, 0, 2)).reshape(Cog, B * N)
                cols_flat = np.ascontiguousarray(cols.transpose(1, 0, 2)).reshape(CKKg, B * N)
                gW = go_flat @ cols_flat.T
                grad_cols = np.ascontiguousarray(
                    (Wmat.T @ go_flat).reshape(CKKg, B, N).transpose(1, 0, 2)
                )
            else:
                gW = np.matmul(go, cols.transpose(0, 2, 1)).sum(axis=0)
                grad_cols = np.matmul(Wmat.T, go)
            grad_weight[g * Cog : (g + 1) * Cog] = gW.reshape(wg.shape)

            if kh == 1 and kw == 1:
                sh, sw = self.stride
                if sh == 1 and sw == 1:
                    grad_x[:, g * Cg : (g + 1) * Cg] = grad_cols.reshape(B, Cg, Hp, Wp)
                else:
                    grad_x[:, g * Cg : (g + 1) * Cg, ::sh, ::sw] = grad_cols.reshape(B, Cg, H_out, W_out)
            else:
                gx = col2im_2d(
                    grad_cols,
                    (B, Cg, Hp, Wp),
                    kh, kw,
                    self.stride,
                    self.dilation,
                )
                grad_x[:, g * Cg : (g + 1) * Cg] = gx

        ph, pw = self.padding
        if ph > 0 or pw > 0:
            grad_x = grad_x[
                :, :,
                ph : Hp - ph if ph > 0 else Hp,
                pw : Wp - pw if pw > 0 else Wp,
            ]

        grad_bias = grad_output.sum(axis=(0, 2, 3)) if self.has_bias else None

        return grad_x, grad_weight, grad_bias


class ConvTranspose2d(Function):
    """
    2D Transposed Convolution — optimized autograd Function.

    Exploits the duality between Conv2d and ConvTranspose2d:
      Forward:   GEMM (W^T @ x_flat)  →  col2im   →  crop   →  output
      Backward:  im2col(grad)  →  GEMM (W @ grad_cols → grad_x)
                                   GEMM (x @ grad_cols^T → grad_W)

    Reuses the same C-accelerated im2col / col2im kernels as Conv2d.

    Pre-allocated buffers (reused across calls, zero alloc on steady state):
      _cols_buf      — GEMM output (W^T @ x) before col2im
      _col2im_buf    — col2im scatter target (full output before crop)
      _grad_full_buf — unpadded grad for backward's im2col
      _im2col_buf    — im2col output in backward
      _gw_buf        — intermediate for grad_weight (batched)
      _gw_sum_buf    — summed grad_weight
      _grad_x_buf    — grad_input from GEMM
    """

    def forward(self, x, weight, bias=None, stride=(1, 1), padding=(0, 0),
                output_padding=(0, 0), dilation=(1, 1), groups=1):
        assert x.ndim == 4
        global _no_grad

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        B, C_in, H_in, W_in = x.shape
        # Weight shape: (C_in, C_out_per_group, kh, kw)
        _, C_out_g, kh, kw = weight.shape
        sh, sw = stride
        ph, pw = padding
        op_h, op_w = output_padding
        dh, dw = dilation

        C_out = C_out_g * groups
        H_out, W_out = calculate_transposed_output_shape(
            (H_in, W_in), (kh, kw), stride, padding, output_padding, dilation
        )
        N_in = H_in * W_in

        # Full col2im target size (before padding crop):
        #   H_col = (H_in - 1)*sh + dh*(kh-1) + 1
        # This is the size such that calculate_output_shape(H_col, kh, sh, 0, dh) == H_in
        H_col = (H_in - 1) * sh + dh * (kh - 1) + 1
        W_col = (W_in - 1) * sw + dw * (kw - 1) + 1

        # =================================================================
        # FAST PATH: groups == 1
        # =================================================================
        if groups == 1:
            CKK = C_out * kh * kw
            W_mat = weight.reshape(C_in, CKK)      # (C_in, C_out*kh*kw)
            x_flat = x.reshape(B, C_in, N_in)       # (B, C_in, N_in)

            # --- Step 1: GEMM   cols = W^T @ x_flat  →  (B, CKK, N_in) ---
            cols_buf = _get_buf(self, '_cols_buf', (B, CKK, N_in), x.dtype)
            np.matmul(W_mat.T, x_flat, out=cols_buf)

            # --- Step 2: col2im  cols → spatial output ---
            if kh == 1 and kw == 1 and dh == 1 and dw == 1:
                # 1×1 fast path: col2im is just scatter at stride intervals
                if sh == 1 and sw == 1:
                    out = cols_buf.reshape(B, C_out, H_in, W_in)
                else:
                    out = _get_buf(self, '_col2im_buf', (B, C_out, H_col, W_col), x.dtype)
                    out[:] = 0
                    out[:, :, ::sh, ::sw] = cols_buf.reshape(B, C_out, H_in, W_in)
            else:
                # General path: C-accelerated col2im
                col2im_buf = _get_buf(self, '_col2im_buf', (B, C_out, H_col, W_col), x.dtype)
                col2im_2d(
                    cols_buf, (B, C_out, H_col, W_col),
                    kh, kw, stride,  dilation, out=col2im_buf,
                )
                out = col2im_buf

            # --- Step 3: Crop padding ---
            if ph > 0 or pw > 0:
                out = out[:, :, ph : H_col - ph, pw : W_col - pw]

            # --- Step 4: Handle output_padding (rare, usually 0) ---
            if op_h > 0 or op_w > 0:
                out_final = np.zeros((B, C_out, H_out, W_out), dtype=x.dtype)
                out_final[:, :, :out.shape[2], :out.shape[3]] = out
                out = out_final

            # --- Step 5: Bias ---
            if bias is not None:
                out = np.ascontiguousarray(out) if not out.flags['C_CONTIGUOUS'] else out
                out += bias[None, :, None, None]

            # --- Save for backward ---
            if not _no_grad:
                self.x_flat = x_flat
                self.weight = weight
                self._W_mat = W_mat
                self.stride = stride
                self.padding = padding
                self.output_padding = output_padding
                self.dilation = dilation
                self.groups = 1
                self.has_bias = bias is not None
                self.kh, self.kw = kh, kw
                self.H_in, self.W_in = H_in, W_in
                self.H_col, self.W_col = H_col, W_col
                self.H_out, self.W_out = H_out, W_out
                self.B, self.C_in, self.C_out = B, C_in, C_out

                # Pre-allocate backward buffers
                gw_shape = (B, C_in, CKK)
                if not hasattr(self, '_gw_buf') or self._gw_buf.shape != gw_shape:
                    self._gw_buf = np.empty(gw_shape, dtype=x.dtype)
                gw_sum_shape = (C_in, CKK)
                if not hasattr(self, '_gw_sum_buf') or self._gw_sum_buf.shape != gw_sum_shape:
                    self._gw_sum_buf = np.empty(gw_sum_shape, dtype=x.dtype)

                # Deep layer optimization buffers
                if N_in <= 64 and N_in < C_in:
                    cin_bn = (C_in, B * N_in)
                    ckk_bn = (CKK, B * N_in)
                    if not hasattr(self, '_xf_flat_buf') or self._xf_flat_buf.shape != cin_bn:
                        self._xf_flat_buf = np.empty(cin_bn, dtype=x.dtype)
                    if not hasattr(self, '_gc_flat_buf') or self._gc_flat_buf.shape != ckk_bn:
                        self._gc_flat_buf = np.empty(ckk_bn, dtype=x.dtype)

            return out

        # =================================================================
        # GROUPED CONVOLUTION
        # =================================================================
        C_in_g = C_in // groups
        outputs = []
        x_flat_list = []

        for g in range(groups):
            xg = x[:, g * C_in_g : (g + 1) * C_in_g]
            wg = weight[g * C_in_g : (g + 1) * C_in_g]
            W_mat_g = wg.reshape(C_in_g, C_out_g * kh * kw)
            xg_flat = xg.reshape(B, C_in_g, N_in)

            cols_g = np.matmul(W_mat_g.T, xg_flat)  # (B, C_out_g*kh*kw, N_in)

            if kh == 1 and kw == 1 and dh == 1 and dw == 1:
                if sh == 1 and sw == 1:
                    out_g = cols_g.reshape(B, C_out_g, H_in, W_in)
                else:
                    out_g = np.zeros((B, C_out_g, H_col, W_col), dtype=x.dtype)
                    out_g[:, :, ::sh, ::sw] = cols_g.reshape(B, C_out_g, H_in, W_in)
            else:
                out_g = col2im_2d(
                    cols_g, (B, C_out_g, H_col, W_col),
                    kh, kw, stride,  dilation,
                )

            if ph > 0 or pw > 0:
                out_g = out_g[:, :, ph : H_col - ph, pw : W_col - pw]

            outputs.append(out_g)
            x_flat_list.append(xg_flat)

        out = np.concatenate(outputs, axis=1)

        if op_h > 0 or op_w > 0:
            out_final = np.zeros((B, C_out, H_out, W_out), dtype=x.dtype)
            out_final[:, :, :out.shape[2], :out.shape[3]] = out
            out = out_final

        if bias is not None:
            out = np.ascontiguousarray(out) if not out.flags['C_CONTIGUOUS'] else out
            out += bias[None, :, None, None]

        if not _no_grad:
            self.x_flat_list = x_flat_list
            self.weight = weight
            self.stride = stride
            self.padding = padding
            self.output_padding = output_padding
            self.dilation = dilation
            self.groups = groups
            self.has_bias = bias is not None
            self.kh, self.kw = kh, kw
            self.H_in, self.W_in = H_in, W_in
            self.H_col, self.W_col = H_col, W_col
            self.H_out, self.W_out = H_out, W_out
            self.B, self.C_in, self.C_out = B, C_in, C_out

        return out

    def backward(self, grad_output):
        B = self.B
        C_in, C_out = self.C_in, self.C_out
        _, C_out_g, kh, kw = self.weight.shape
        groups = self.groups
        sh, sw = self.stride
        ph, pw = self.padding
        op_h, op_w = self.output_padding
        dh, dw = self.dilation
        H_in, W_in = self.H_in, self.W_in
        H_col, W_col = self.H_col, self.W_col
        H_out, W_out = self.H_out, self.W_out
        N_in = H_in * W_in

        # =================================================================
        # Step 1: Reverse the crop + output_padding to get grad_full
        #         (gradient w.r.t. the full col2im output)
        # =================================================================
        if ph > 0 or pw > 0 or op_h > 0 or op_w > 0:
            grad_full = _get_pad_buf(self, '_grad_full_buf',
                                     (B, C_out, H_col, W_col), grad_output.dtype)
            grad_full[:] = 0
            h_copy = H_col - 2 * ph   # = H_out - op_h
            w_copy = W_col - 2 * pw
            grad_full[:, :, ph : H_col - ph, pw : W_col - pw] = \
                grad_output[:, :, :h_copy, :w_copy]
        else:
            grad_full = grad_output

        # =================================================================
        # FAST PATH: groups == 1
        # =================================================================
        if groups == 1:
            CKK = C_out * kh * kw
            W_mat = self._W_mat       # (C_in, CKK)
            x_flat = self.x_flat      # (B, C_in, N_in)

            # --- Backward of col2im = im2col ---
            if kh == 1 and kw == 1 and dh == 1 and dw == 1:
                if sh == 1 and sw == 1:
                    grad_cols = grad_full.reshape(B, C_out, N_in)
                else:
                    grad_cols = grad_full[:, :, ::sh, ::sw].reshape(B, C_out, N_in)
            else:
                im2col_buf = _get_buf(self, '_im2col_buf', (B, CKK, N_in), grad_output.dtype)
                grad_cols = im2col_2d(
                    grad_full, kh, kw, (sh, sw),  (dh, dw), out=im2col_buf,
                )

            # --- grad_weight = x_flat @ grad_cols^T ---
            if N_in <= 64 and N_in < C_in:
                # Deep layers: single big GEMM
                xf = self._xf_flat_buf       # (C_in, B*N_in)
                gcf = self._gc_flat_buf      # (CKK, B*N_in)
                np.copyto(xf.reshape(C_in, B, N_in), x_flat.transpose(1, 0, 2))
                np.copyto(gcf.reshape(CKK, B, N_in), grad_cols.transpose(1, 0, 2))
                grad_weight = (xf @ gcf.T).reshape(self.weight.shape)
            else:
                # Early layers: batched matmul + sum
                np.matmul(x_flat, grad_cols.transpose(0, 2, 1), out=self._gw_buf)
                np.sum(self._gw_buf, axis=0, out=self._gw_sum_buf)
                grad_weight = self._gw_sum_buf.reshape(self.weight.shape)

            # --- grad_input = W_mat @ grad_cols ---
            grad_x_buf = _get_buf(self, '_grad_x_buf', (B, C_in, N_in), grad_output.dtype)
            np.matmul(W_mat, grad_cols, out=grad_x_buf)
            grad_x = grad_x_buf.reshape(B, C_in, H_in, W_in)

            # --- grad_bias ---
            grad_bias = grad_output.sum(axis=(0, 2, 3)) if self.has_bias else None

            return grad_x, grad_weight, grad_bias

        # =================================================================
        # GROUPED CONVOLUTION backward
        # =================================================================
        C_in_g = C_in // groups
        grad_weight = np.zeros_like(self.weight)
        grad_x = np.zeros((B, C_in, H_in, W_in), dtype=grad_output.dtype)

        for g in range(groups):
            grad_full_g = grad_full[:, g * C_out_g : (g + 1) * C_out_g]
            wg = self.weight[g * C_in_g : (g + 1) * C_in_g]
            W_mat_g = wg.reshape(C_in_g, C_out_g * kh * kw)
            xg_flat = self.x_flat_list[g]
            CKK_g = C_out_g * kh * kw

            # im2col on this group's grad
            if kh == 1 and kw == 1 and dh == 1 and dw == 1:
                if sh == 1 and sw == 1:
                    grad_cols_g = grad_full_g.reshape(B, C_out_g, N_in)
                else:
                    grad_cols_g = grad_full_g[:, :, ::sh, ::sw].reshape(B, C_out_g, N_in)
            else:
                grad_cols_g = im2col_2d(
                    grad_full_g, kh, kw, (sh, sw),  (dh, dw),
                )

            # grad_weight for this group
            if N_in <= 64 and N_in < C_in_g:
                xf = np.ascontiguousarray(xg_flat.transpose(1, 0, 2)).reshape(C_in_g, B * N_in)
                gcf = np.ascontiguousarray(grad_cols_g.transpose(1, 0, 2)).reshape(CKK_g, B * N_in)
                gW = xf @ gcf.T
            else:
                gW = np.matmul(xg_flat, grad_cols_g.transpose(0, 2, 1)).sum(axis=0)
            grad_weight[g * C_in_g : (g + 1) * C_in_g] = gW.reshape(wg.shape)

            # grad_input for this group
            grad_xg = np.matmul(W_mat_g, grad_cols_g)  # (B, C_in_g, N_in)
            grad_x[:, g * C_in_g : (g + 1) * C_in_g] = \
                grad_xg.reshape(B, C_in_g, H_in, W_in)

        grad_bias = grad_output.sum(axis=(0, 2, 3)) if self.has_bias else None

        return grad_x, grad_weight, grad_bias


class DepthwiseConv2d(Function):
    """Depthwise 2D Convolution functional operation."""

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1):
        raise NotImplementedError("TODO: Implement DepthwiseConv2d forward")

    def backward(self, grad_output):
        raise NotImplementedError("TODO: Implement DepthwiseConv2d backward")


class Conv3d(Function):
    """3D Convolution functional operation."""

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        raise NotImplementedError("TODO: Implement Conv3d forward")

    def backward(self, grad_output):
        raise NotImplementedError("TODO: Implement Conv3d backward")


class ConvTranspose3d(Function):
    """3D Transposed Convolution functional operation."""

    def forward(self, x, weight, bias=None, stride=1, padding=0,
                output_padding=0, dilation=1, groups=1):
        raise NotImplementedError("TODO: Implement ConvTranspose3d forward")

    def backward(self, grad_output):
        raise NotImplementedError("TODO: Implement ConvTranspose3d backward")


class DepthwiseSeparableConv2d(Function):
    """Depthwise Separable 2D Convolution functional operation."""

    def forward(self, x, depthwise_weight, pointwise_weight,
                depthwise_bias=None, pointwise_bias=None,
                stride=1, padding=0, dilation=1):
        raise NotImplementedError("TODO: Implement DepthwiseSeparableConv2d forward")

    def backward(self, grad_output):
        raise NotImplementedError("TODO: Implement DepthwiseSeparableConv2d backward")


class DilatedConv2d(Function):
    """Dilated (Atrous) 2D Convolution functional operation."""

    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=2):
        raise NotImplementedError("TODO: Implement DilatedConv2d forward")

    def backward(self, grad_output):
        raise NotImplementedError("TODO: Implement DilatedConv2d backward")