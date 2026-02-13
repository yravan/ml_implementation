"""
Benchmark: C direct backward kernels vs matmul-based backward.

The C file has conv2d_backward_weight_f32 and conv2d_backward_input_f32
that fuse matmul + col2im into a single pass. Let's see if they're faster.
"""
import numpy as np
import ctypes, pathlib, subprocess, os, time

# ── Compile & load ──────────────────────────────────────────────
src = pathlib.Path("./nn_core/_conv_c.c")
so  = pathlib.Path("./nn_core/_conv_c.so")

# Find OpenMP
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

subprocess.check_call(cmd, stderr=subprocess.PIPE)
lib = ctypes.CDLL(str(so))

_f32p = ctypes.POINTER(ctypes.c_float)
_ci = ctypes.c_int

# Register im2col
lib.im2col_f32.argtypes = [_f32p, _f32p] + [_ci]*12
lib.im2col_f32.restype = None

lib.col2im_f32.argtypes = [_f32p, _f32p] + [_ci]*12
lib.col2im_f32.restype = None

# Register direct backward kernels
lib.conv2d_backward_weight_f32.argtypes = [
    _f32p, _f32p, _f32p,  # x, grad_out, grad_weight
    _ci, _ci, _ci, _ci,   # B, C, H, W
    _ci, _ci, _ci,         # C_out, kh, kw
    _ci, _ci, _ci, _ci,   # sh, sw, dh, dw
    _ci, _ci, _ci,         # H_out, W_out, groups
]
lib.conv2d_backward_weight_f32.restype = None

lib.conv2d_backward_input_f32.argtypes = [
    _f32p, _f32p, _f32p,  # weight, grad_out, grad_x
    _ci, _ci, _ci, _ci,   # B, C, H, W
    _ci, _ci, _ci,         # C_out, kh, kw
    _ci, _ci, _ci, _ci,   # sh, sw, dh, dw
    _ci, _ci, _ci,         # H_out, W_out, groups
]
lib.conv2d_backward_input_f32.restype = None

# Register direct forward
lib.conv2d_forward_f32.argtypes = [
    _f32p, _f32p, _f32p, _f32p,  # x, weight, bias, out
    _ci, _ci, _ci, _ci,           # B, C, H, W
    _ci, _ci, _ci,                 # C_out, kh, kw
    _ci, _ci, _ci, _ci,           # sh, sw, dh, dw
    _ci, _ci, _ci,                 # H_out, W_out, groups
]
lib.conv2d_forward_f32.restype = None


def im2col_c(x, kh, kw, sh, sw, dh, dw, H_out, W_out):
    B, C, H, W = x.shape
    CKK = C * kh * kw
    N = H_out * W_out
    cols = np.empty((B, CKK, N), dtype=np.float32)
    lib.im2col_f32(
        x.ctypes.data_as(_f32p), cols.ctypes.data_as(_f32p),
        B, C, H, W, kh, kw, sh, sw, dh, dw, H_out, W_out
    )
    return cols

def col2im_c(cols, B, C, H, W, kh, kw, sh, sw, dh, dw, H_out, W_out):
    grad_x = np.zeros((B, C, H, W), dtype=np.float32)
    lib.col2im_f32(
        cols.ctypes.data_as(_f32p), grad_x.ctypes.data_as(_f32p),
        B, C, H, W, kh, kw, sh, sw, dh, dw, H_out, W_out
    )
    return grad_x


def bench_layer(name, B, C_in, H_in, W_in, C_out, kh, kw, sh, sw, pad, n_iter=5):
    """Compare matmul-based backward vs C direct backward for one layer."""
    dh, dw = 1, 1
    H_pad = H_in + 2*pad
    W_pad = W_in + 2*pad
    H_out = (H_pad - kh) // sh + 1
    W_out = (W_pad - kw) // sw + 1
    N = H_out * W_out
    CKK = C_in * kh * kw

    # Allocate inputs
    x_pad = np.random.randn(B, C_in, H_pad, W_pad).astype(np.float32)
    weight = np.random.randn(C_out, C_in, kh, kw).astype(np.float32)
    grad_out = np.random.randn(B, C_out, H_out, W_out).astype(np.float32)

    # im2col for matmul path
    cols = im2col_c(x_pad, kh, kw, sh, sw, dh, dw, H_out, W_out)
    Wmat = weight.reshape(C_out, -1)
    Wmat_T = np.ascontiguousarray(Wmat.T)
    go = grad_out.reshape(B, C_out, N)

    # Pre-allocated buffers for matmul path
    gw_buf = np.empty((B, C_out, CKK), dtype=np.float32)
    gw_sum_buf = np.empty((C_out, CKK), dtype=np.float32)

    print(f"\n{'='*70}")
    print(f"  {name}: ({B},{C_in},{H_in},{W_in}) → ({B},{C_out},{H_out},{W_out})")
    print(f"  kernel={kh}×{kw}, stride={sh}, pad={pad}")
    cols_mb = cols.nbytes / 1e6
    xpad_mb = x_pad.nbytes / 1e6
    print(f"  cols: {cols_mb:.1f} MB  |  x_padded: {xpad_mb:.1f} MB")
    print(f"{'='*70}")

    # ── Warmup ──
    for _ in range(2):
        np.matmul(go, cols.transpose(0, 2, 1), out=gw_buf)
        np.sum(gw_buf, axis=0, out=gw_sum_buf)
        np.matmul(Wmat_T, go, out=cols)
        cols = im2col_c(x_pad, kh, kw, sh, sw, dh, dw, H_out, W_out)

    # ── METHOD A: Current matmul approach ──
    times_a = []
    for _ in range(n_iter):
        cols = im2col_c(x_pad, kh, kw, sh, sw, dh, dw, H_out, W_out)
        go = grad_out.reshape(B, C_out, N)

        t0 = time.perf_counter()

        # grad_weight
        np.matmul(go, cols.transpose(0, 2, 1), out=gw_buf)
        np.sum(gw_buf, axis=0, out=gw_sum_buf)
        gw_a = gw_sum_buf.reshape(weight.shape).copy()

        # grad_input (overwrite cols)
        grad_cols = np.matmul(Wmat_T, go)

        # col2im
        gx_a = col2im_c(grad_cols, B, C_in, H_pad, W_pad,
                         kh, kw, sh, sw, dh, dw, H_out, W_out)

        t1 = time.perf_counter()
        times_a.append((t1 - t0) * 1000)

    # ── METHOD B: C direct backward ──
    times_b_w = []
    times_b_x = []
    times_b = []
    for _ in range(n_iter):
        t0 = time.perf_counter()

        gw_b = np.empty_like(weight)
        lib.conv2d_backward_weight_f32(
            x_pad.ctypes.data_as(_f32p),
            grad_out.ctypes.data_as(_f32p),
            gw_b.ctypes.data_as(_f32p),
            B, C_in, H_pad, W_pad,
            C_out, kh, kw,
            sh, sw, dh, dw,
            H_out, W_out, 1,
        )
        t1 = time.perf_counter()

        gx_b = np.zeros((B, C_in, H_pad, W_pad), dtype=np.float32)
        lib.conv2d_backward_input_f32(
            weight.ctypes.data_as(_f32p),
            grad_out.ctypes.data_as(_f32p),
            gx_b.ctypes.data_as(_f32p),
            B, C_in, H_pad, W_pad,
            C_out, kh, kw,
            sh, sw, dh, dw,
            H_out, W_out, 1,
        )
        t2 = time.perf_counter()

        times_b_w.append((t1 - t0) * 1000)
        times_b_x.append((t2 - t1) * 1000)
        times_b.append((t2 - t0) * 1000)

    # ── Correctness check ──
    gw_ok = np.allclose(gw_a, gw_b, rtol=1e-3, atol=1e-4)
    gx_ok = np.allclose(gx_a, gx_b, rtol=1e-3, atol=1e-4)

    avg_a = np.median(times_a)
    avg_b = np.median(times_b)
    avg_bw = np.median(times_b_w)
    avg_bx = np.median(times_b_x)

    print(f"  Matmul+col2im:   {avg_a:8.2f} ms")
    print(f"  C direct total:  {avg_b:8.2f} ms  (gw={avg_bw:.2f}, gx={avg_bx:.2f})")
    print(f"  Speedup:         {avg_a/avg_b:8.2f}×")
    print(f"  Correct: gw={gw_ok}, gx={gx_ok}")
    if not gw_ok:
        diff = np.abs(gw_a - gw_b)
        print(f"    gw max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
    if not gx_ok:
        diff = np.abs(gx_a - gx_b)
        print(f"    gx max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")

    return avg_a, avg_b


# ── AlexNet layers (B=128) ──
B = 128
results = []
results.append(bench_layer("Conv1", B, 3,   227, 227, 64,  11, 11, 4, 4, 2))
results.append(bench_layer("Conv2", B, 64,  27,  27,  192, 5,  5,  1, 1, 2))
results.append(bench_layer("Conv3", B, 192, 13,  13,  384, 3,  3,  1, 1, 1))
results.append(bench_layer("Conv4", B, 384, 13,  13,  256, 3,  3,  1, 1, 1))
results.append(bench_layer("Conv5", B, 256, 13,  13,  256, 3,  3,  1, 1, 1))

print(f"\n{'='*70}")
print(f"  TOTALS")
print(f"{'='*70}")
total_a = sum(r[0] for r in results)
total_b = sum(r[1] for r in results)
print(f"  Matmul+col2im: {total_a:.1f} ms")
print(f"  C direct:      {total_b:.1f} ms")
print(f"  Speedup:       {total_a/total_b:.2f}×")