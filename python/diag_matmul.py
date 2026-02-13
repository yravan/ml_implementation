"""
Diagnose why MatMul backward is slow and benchmark fixes.
Run this on your machine: python diag_matmul.py
"""

import numpy as np
import time

print("=" * 70)
print("BLAS / NUMPY DIAGNOSTICS")
print("=" * 70)

# Check numpy BLAS linkage
print(f"\nNumPy version: {np.__version__}")
print(f"NumPy location: {np.__file__}")

try:
    config = np.show_config(mode="dicts")
    if isinstance(config, dict):
        blas = config.get("Build Dependencies", {}).get("blas", {})
        print(f"BLAS: {blas}")
    else:
        np.show_config()
except:
    np.show_config()

# Quick BLAS benchmark — if these are slow, it's a BLAS problem
print("\n" + "=" * 70)
print("BLAS GEMM BENCHMARKS (float32)")
print("=" * 70)


def bench(fn, label, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:45s} {ms:8.2f} ms")
    return ms


np.random.seed(42)

# The actual FC1 backward operations
x = np.random.randn(32, 9216).astype(np.float32)
g = np.random.randn(32, 4096).astype(np.float32)
w = np.random.randn(9216, 4096).astype(np.float32)

print("\nFC1 backward shapes: x=(32,9216) g=(32,4096) w=(9216,4096)")
print(f"  Weight size: {w.nbytes / 1e6:.0f} MB\n")

ms_dx = bench(lambda: g @ w.T, "dx = grad @ weight.T  (32,4096)@(4096,9216)")
ms_dy = bench(lambda: x.T @ g, "dy = x.T @ grad       (9216,32)@(32,4096)")
ms_both = bench(lambda: (g @ w.T, x.T @ g), "Both dx + dy")

print(f"\n  Total FC1 backward: {ms_both:.2f} ms")
print("  Your profiler showed: ~120 ms avg (includes FC1 + FC2)")

# FC2 backward
x2 = np.random.randn(32, 4096).astype(np.float32)
g2 = np.random.randn(32, 4096).astype(np.float32)
w2 = np.random.randn(4096, 4096).astype(np.float32)

print("\nFC2 backward shapes: x=(32,4096) g=(32,4096) w=(4096,4096)")
ms_fc2 = bench(lambda: (g2 @ w2.T, x2.T @ g2), "Both dx + dy")

print(f"\n  Expected combined FC backward per iter: {ms_both + ms_fc2:.2f} ms")
print("  Your profiler showed: ~240 ms per iter")

# Check if transposed inputs are slow
print("\n--- Transpose handling test ---")
xT = np.ascontiguousarray(x.T)  # (9216, 32) contiguous
bench(lambda: x.T @ g, "x.T @ g          (transposed view)")
bench(lambda: xT @ g, "x_contig.T @ g   (pre-copied)")

# Test alternative formulations
print("\n--- Alternative weight gradient formulations ---")
bench(lambda: x.T @ g, "x.T @ grad               (standard)")
bench(lambda: np.dot(x.T, g), "np.dot(x.T, grad)")
bench(lambda: np.matmul(x.T, g), "np.matmul(x.T, grad)")
bench(lambda: np.einsum("bi,bj->ij", x, g), "einsum('bi,bj->ij')")

# Full Linear backward timing
print("\n--- Full Linear backward (dx + dy + db) ---")


def linear_bwd_current():
    dx = g @ w.T
    dy = x.T @ g
    db = g.sum(axis=0)
    return dx, dy, db


bench(linear_bwd_current, "Current (3 separate ops)")

# Check: is memory allocation the bottleneck?
print("\n--- Memory allocation test ---")
out_buf = np.empty((9216, 4096), dtype=np.float32)
bench(lambda: np.matmul(x.T, g, out=out_buf), "matmul with pre-allocated out")
bench(lambda: x.T @ g, "matmul with new allocation")
bench(lambda: np.empty((9216, 4096), dtype=np.float32), "Just np.empty((9216,4096))")

print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
if ms_both > 50:
    print(f"""
  FC1 backward takes {ms_both:.0f} ms — this is VERY SLOW.
  Expected: 3-10 ms on Apple Silicon with Accelerate.

  Your numpy is likely using OpenBLAS (from conda) instead of 
  Apple Accelerate (which uses AMX for fast float32 GEMM).

  FIX: Install numpy with Accelerate:
    pip install --force-reinstall numpy

  Or create a new env with pip-installed numpy:
    conda create -n ml-fast python=3.12
    conda activate ml-fast
    pip install numpy  # This links to Accelerate on macOS
""")
elif ms_both > 15:
    print(f"""
  FC1 backward takes {ms_both:.0f} ms — somewhat slow.
  May benefit from pre-allocated buffers or BLAS tuning.
""")
else:
    print(f"""
  FC1 backward takes {ms_both:.0f} ms — this is reasonable for BLAS.
  The profiler overhead or graph traversal may be inflating times.
""")
