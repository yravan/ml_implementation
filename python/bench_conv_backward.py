"""
Benchmark different strategies for Conv2d backward matmuls.
Run this on YOUR machine — results vary hugely by BLAS backend
(Accelerate vs OpenBLAS vs MKL).
"""
import numpy as np
import time

def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def test_layer(B, C_in, C_out, kh, kw, H_out, W_out, label):
    CKK = C_in * kh * kw
    N = H_out * W_out

    go   = np.random.randn(B, C_out, N).astype(np.float32)
    cols = np.random.randn(B, CKK, N).astype(np.float32)
    Wmat = np.random.randn(C_out, CKK).astype(np.float32)

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"  go={go.shape}  cols={cols.shape}  Wmat={Wmat.shape}")
    print(f"  cols memory: {cols.nbytes/1e6:.1f} MB")
    print(f"{'='*70}")

    # ===== GRAD_WEIGHT strategies =====
    print(f"\n  --- grad_weight: (C_out, CKK) from go @ cols ---")

    # 1. Batched matmul + sum (original)
    def gw_batched():
        return np.matmul(go, cols.transpose(0, 2, 1)).sum(axis=0)
    ms = bench(gw_batched)
    ref = gw_batched()
    print(f"  1) batched matmul + sum:    {ms:7.2f} ms")

    # 2. Reshape to single GEMM
    def gw_reshape():
        go_flat = go.transpose(1, 0, 2).reshape(C_out, B * N)
        cols_flat = cols.transpose(1, 0, 2).reshape(CKK, B * N)
        return go_flat @ cols_flat.T
    ms = bench(gw_reshape)
    err = np.abs(gw_reshape() - ref).max()
    print(f"  2) reshape + single GEMM:   {ms:7.2f} ms  (err={err:.1e})")

    # 3. einsum
    def gw_einsum():
        return np.einsum('bon,bkn->ok', go, cols)
    ms = bench(gw_einsum)
    err = np.abs(gw_einsum() - ref).max()
    print(f"  3) einsum('bon,bkn->ok'):   {ms:7.2f} ms  (err={err:.1e})")

    # 4. tensordot
    def gw_tensordot():
        return np.tensordot(go, cols, axes=([0, 2], [0, 2]))
    ms = bench(gw_tensordot)
    err = np.abs(gw_tensordot() - ref).max()
    print(f"  4) tensordot axes=[0,2]:    {ms:7.2f} ms  (err={err:.1e})")

    # 5. Manual reshape with contiguous copy first
    go_c = np.ascontiguousarray(go.transpose(1, 0, 2)).reshape(C_out, B * N)
    cols_c = np.ascontiguousarray(cols.transpose(1, 0, 2)).reshape(CKK, B * N)
    def gw_precopied():
        return go_c @ cols_c.T
    ms_gemm = bench(gw_precopied)
    err = np.abs(gw_precopied() - ref).max()
    print(f"  5) pre-transposed GEMM:     {ms_gemm:7.2f} ms  (err={err:.1e})  [GEMM only, no copy]")

    # Measure copy cost alone
    def copy_cost():
        a = np.ascontiguousarray(go.transpose(1, 0, 2)).reshape(C_out, B * N)
        b = np.ascontiguousarray(cols.transpose(1, 0, 2)).reshape(CKK, B * N)
        return a, b
    ms_copy = bench(copy_cost)
    print(f"     transpose+copy cost:     {ms_copy:7.2f} ms")
    print(f"     total (copy+GEMM):       {ms_copy + ms_gemm:7.2f} ms")

    # ===== GRAD_COLS strategies =====
    print(f"\n  --- grad_cols: (B, CKK, N) from Wmat.T @ go ---")

    # 1. Broadcast matmul (original)
    def gc_broadcast():
        return np.matmul(Wmat.T, go)
    ms = bench(gc_broadcast)
    ref_gc = gc_broadcast()
    print(f"  1) broadcast np.matmul:     {ms:7.2f} ms")

    # 2. Reshape to single GEMM
    def gc_reshape():
        go_flat = go.transpose(1, 0, 2).reshape(C_out, B * N)
        return (Wmat.T @ go_flat).reshape(CKK, B, N).transpose(1, 0, 2)
    ms = bench(gc_reshape)
    err = np.abs(gc_reshape() - ref_gc).max()
    print(f"  2) reshape single GEMM:     {ms:7.2f} ms  (err={err:.1e})")

    # 3. Reuse pre-transposed go from grad_weight
    def gc_reuse():
        return (Wmat.T @ go_c).reshape(CKK, B, N).transpose(1, 0, 2)
    ms = bench(gc_reuse)
    err = np.abs(gc_reuse() - ref_gc).max()
    print(f"  3) reuse pre-transposed go: {ms:7.2f} ms  (err={err:.1e})  [if go_c already computed]")

    # 4. einsum
    def gc_einsum():
        return np.einsum('kc,bcn->bkn', Wmat.T, go)
    ms = bench(gc_einsum)
    err = np.abs(gc_einsum() - ref_gc).max()
    print(f"  4) einsum('kc,bcn->bkn'):   {ms:7.2f} ms  (err={err:.1e})")


# Typical CNN layer shapes
test_layer(32,   3,  64, 3, 3, 32, 32, "Layer 1:   3→64   3×3  32×32  (early, small C_in)")
test_layer(32,  64, 128, 3, 3, 16, 16, "Layer 2:  64→128  3×3  16×16  (mid)")
test_layer(32, 128, 256, 3, 3,  8,  8, "Layer 3: 128→256  3×3   8×8   (deep)")
test_layer(32, 256, 512, 3, 3,  4,  4, "Layer 4: 256→512  3×3   4×4   (deepest)")
test_layer(32,  64,  64, 1, 1, 16, 16, "1×1 conv: 64→64   1×1  16×16")