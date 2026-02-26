"""
Central loader for the unified C extension (_nn_c.so).

Usage from any nn_core module:

    from ._c_lib import get_c_lib, F32P, F64P, I64P, CI, CF

    lib = get_c_lib()          # None if unavailable
    if lib is not None:
        lib.im2col_f32(...)    # etc.

The .so is compiled on first call and cached for the process lifetime.
"""

import ctypes
import os
import pathlib
import platform
import subprocess
import warnings

# ---- ctypes shorthand (importable by consumers) ----
F32P = ctypes.POINTER(ctypes.c_float)
F64P = ctypes.POINTER(ctypes.c_double)
I64P = ctypes.POINTER(ctypes.c_longlong)
CI   = ctypes.c_int
CF   = ctypes.c_float

# ---- singleton ----
_lib = None


def get_c_lib():
    """Compile (if needed) and return the shared C library, or None."""
    global _lib
    if _lib is not None:
        return _lib

    src = pathlib.Path(__file__).parent / "_nn_c.c"
    so  = pathlib.Path(__file__).parent / "_nn_c.so"

    if not src.exists():
        warnings.warn(
            f"C source {src} not found — using pure-numpy fallback. "
            "Place _nn_c.c next to _c_lib.py for 2-5× speedup.",
            RuntimeWarning, stacklevel=3,
        )
        return None

    needs_compile = (
        not so.exists()
        or os.path.getmtime(src) > os.path.getmtime(so)
    )

    if needs_compile:
        system = platform.system()

        if system == "Linux":
            cmd = [
                "gcc", "-O3", "-march=native",
                "-ffast-math", "-fno-finite-math-only",
                "-fopenmp",
                "-shared", "-fPIC",
                "-o", str(so), str(src),
                "-lm",
            ]
        elif system == "Darwin":
            omp_prefix = None
            for prefix in ["/opt/homebrew", "/usr/local"]:
                if os.path.exists(f"{prefix}/opt/libomp/lib/libomp.dylib"):
                    omp_prefix = f"{prefix}/opt/libomp"
                    break

            base = [
                "clang", "-O3", "-mcpu=native",
                "-ffast-math", "-fno-finite-math-only",
            ]
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
                    "Run 'brew install libomp' for multi-threaded kernels.",
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
            subprocess.check_call(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            warnings.warn(
                "Failed to compile C extension — using pure-numpy fallback.",
                RuntimeWarning, stacklevel=3,
            )
            return None

    lib = ctypes.CDLL(str(so))

    # ------------------------------------------------------------------
    # Register argtypes for every exported symbol
    # ------------------------------------------------------------------

    # ---- im2col / col2im 2D: 2 ptrs + 12 ints ----
    for suffix, ptr_t in [("f32", F32P), ("f64", F64P)]:
        for name in [f"im2col_{suffix}", f"col2im_{suffix}"]:
            fn = getattr(lib, name)
            fn.argtypes = [ptr_t, ptr_t] + [CI] * 12
            fn.restype = None

    # ---- im2col / col2im 1D: 2 ptrs + 7 ints ----
    for suffix, ptr_t in [("f32", F32P), ("f64", F64P)]:
        for name in [f"im2col1d_{suffix}", f"col2im1d_{suffix}"]:
            fn = getattr(lib, name)
            fn.argtypes = [ptr_t, ptr_t] + [CI] * 7
            fn.restype = None

    # ---- conv2d forward: 4 ptrs + 14 ints ----
    #   x, weight, bias, out, B, C, H, W, C_out, kh, kw, sh, sw, dh, dw,
    #   H_out, W_out, groups
    lib.conv2d_forward_f32.argtypes = [F32P] * 4 + [CI] * 14
    lib.conv2d_forward_f32.restype = None

    # ---- conv2d backward_weight: 3 ptrs + 14 ints ----
    lib.conv2d_backward_weight_f32.argtypes = [F32P] * 3 + [CI] * 14
    lib.conv2d_backward_weight_f32.restype = None

    # ---- conv2d backward_input: 3 ptrs + 14 ints ----
    lib.conv2d_backward_input_f32.argtypes = [F32P] * 3 + [CI] * 14
    lib.conv2d_backward_input_f32.restype = None

    # ---- max_pool1d: f32, f32, i64, 7 ints ----
    lib.max_pool1d.argtypes = [F32P, F32P, I64P] + [CI] * 7
    lib.max_pool1d.restype = None

    # ---- max_pool1d_backward: f32, f32, i64, 4 ints ----
    lib.max_pool1d_backward.argtypes = [F32P, F32P, I64P] + [CI] * 4
    lib.max_pool1d_backward.restype = None

    # ---- max_pool2d: f32, f32, i64, 12 ints (B,C,H,W,Kh,Kw,Sh,Sw,Dh,Dw,H_out,W_out) ----
    lib.max_pool2d.argtypes = [F32P, F32P, I64P] + [CI] * 12
    lib.max_pool2d.restype = None

    # ---- max_pool2d_backward: f32, f32, i64, 6 ints ----
    lib.max_pool2d_backward.argtypes = [F32P, F32P, I64P] + [CI] * 6
    lib.max_pool2d_backward.restype = None

    # ---- avg_pool1d: 2 f32 ptrs + 9 ints (B,C,L,K,S,L_out,count_include_pad,pad,orig_L) ----
    lib.avg_pool1d.argtypes = [F32P, F32P] + [CI] * 9
    lib.avg_pool1d.restype = None
    lib.avg_pool1d_backward.argtypes = [F32P, F32P] + [CI] * 9
    lib.avg_pool1d_backward.restype = None

    # ---- avg_pool2d: 2 f32 ptrs + 15 ints ----
    lib.avg_pool2d.argtypes = [F32P, F32P] + [CI] * 15
    lib.avg_pool2d.restype = None
    lib.avg_pool2d_backward.argtypes = [F32P, F32P] + [CI] * 15
    lib.avg_pool2d_backward.restype = None

    # ---- adaptive_avg_pool2d: 2 f32 ptrs + 6 ints ----
    lib.adaptive_avg_pool2d.argtypes = [F32P, F32P] + [CI] * 6
    lib.adaptive_avg_pool2d.restype = None
    lib.adaptive_avg_pool2d_backward.argtypes = [F32P, F32P] + [CI] * 6
    lib.adaptive_avg_pool2d_backward.restype = None

    # ---- batchnorm_forward_train: 7 f32 ptrs + 3 ints + 1 float ----
    lib.batchnorm_forward_train.argtypes = [F32P] * 7 + [CI] * 3 + [CF]
    lib.batchnorm_forward_train.restype = None

    # ---- batchnorm_forward_eval: 6 f32 ptrs + 3 ints + 1 float ----
    lib.batchnorm_forward_eval.argtypes = [F32P] * 6 + [CI] * 3 + [CF]
    lib.batchnorm_forward_eval.restype = None

    # ---- batchnorm_backward: 7 f32 ptrs + 3 ints + 1 float ----
    lib.batchnorm_backward.argtypes = [F32P] * 7 + [CI] * 3 + [CF]
    lib.batchnorm_backward.restype = None

    _lib = lib
    return lib
