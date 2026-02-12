/*
 * High-performance im2col / col2im in C for numpy conv2d.
 * Compiled as shared library, called via ctypes.
 *
 * Optimizations:
 *   - Direct pointer arithmetic (no Python overhead)
 *   - memcpy fast path for stride_w=1 (contiguous rows)
 *   - Compiler auto-vectorization with -O3 -march=native
 *   - Both float32 and float64 variants
 *
 * Build: gcc -O3 -march=native -ffast-math -shared -fPIC -o _conv_c.so _conv_c.c
 */
#include <string.h>

/* ================================================================
 * im2col: (B, C, H, W) -> (B, C*kh*kw, H_out*W_out)
 * ================================================================ */

void im2col_f32(
    const float *__restrict__ x,
    float *__restrict__ cols,
    int B, int C, int H, int W,
    int kh, int kw,
    int sh, int sw,
    int dh, int dw,
    int H_out, int W_out
) {
    const int N = H_out * W_out;
    const int CKK = C * kh * kw;
    const int CHW = C * H * W;

    for (int b = 0; b < B; b++) {
        const float *xb = x + b * CHW;
        float *cb = cols + b * CKK * N;

        int col_idx = 0;
        for (int c = 0; c < C; c++) {
            const float *xc = xb + c * H * W;
            for (int ki = 0; ki < kh; ki++) {
                const int h_base = ki * dh;
                for (int kj = 0; kj < kw; kj++) {
                    float *col_row = cb + col_idx * N;
                    const int w_base = kj * dw;

                    if (sw == 1) {
                        /* Fast path: contiguous row copy */
                        for (int i = 0; i < H_out; i++) {
                            memcpy(col_row + i * W_out,
                                   xc + (h_base + i * sh) * W + w_base,
                                   W_out * sizeof(float));
                        }
                    } else {
                        for (int i = 0; i < H_out; i++) {
                            const float *xrow = xc + (h_base + i * sh) * W + w_base;
                            float *out = col_row + i * W_out;
                            for (int j = 0; j < W_out; j++) {
                                out[j] = xrow[j * sw];
                            }
                        }
                    }
                    col_idx++;
                }
            }
        }
    }
}

void im2col_f64(
    const double *__restrict__ x,
    double *__restrict__ cols,
    int B, int C, int H, int W,
    int kh, int kw,
    int sh, int sw,
    int dh, int dw,
    int H_out, int W_out
) {
    const int N = H_out * W_out;
    const int CKK = C * kh * kw;
    const int CHW = C * H * W;

    for (int b = 0; b < B; b++) {
        const double *xb = x + b * CHW;
        double *cb = cols + b * CKK * N;

        int col_idx = 0;
        for (int c = 0; c < C; c++) {
            const double *xc = xb + c * H * W;
            for (int ki = 0; ki < kh; ki++) {
                const int h_base = ki * dh;
                for (int kj = 0; kj < kw; kj++) {
                    double *col_row = cb + col_idx * N;
                    const int w_base = kj * dw;

                    if (sw == 1) {
                        for (int i = 0; i < H_out; i++) {
                            memcpy(col_row + i * W_out,
                                   xc + (h_base + i * sh) * W + w_base,
                                   W_out * sizeof(double));
                        }
                    } else {
                        for (int i = 0; i < H_out; i++) {
                            const double *xrow = xc + (h_base + i * sh) * W + w_base;
                            double *out = col_row + i * W_out;
                            for (int j = 0; j < W_out; j++)
                                out[j] = xrow[j * sw];
                        }
                    }
                    col_idx++;
                }
            }
        }
    }
}


/* ================================================================
 * col2im: (B, C*kh*kw, H_out*W_out) -> (B, C, H, W)
 * Output array MUST be pre-zeroed by caller.
 * ================================================================ */

void col2im_f32(
    const float *__restrict__ cols,
    float *__restrict__ x,
    int B, int C, int H, int W,
    int kh, int kw,
    int sh, int sw,
    int dh, int dw,
    int H_out, int W_out
) {
    const int N = H_out * W_out;
    const int CKK = C * kh * kw;
    const int CHW = C * H * W;

    for (int b = 0; b < B; b++) {
        const float *cb = cols + b * CKK * N;
        float *xb = x + b * CHW;

        int col_idx = 0;
        for (int c = 0; c < C; c++) {
            float *xc = xb + c * H * W;
            for (int ki = 0; ki < kh; ki++) {
                const int h_base = ki * dh;
                for (int kj = 0; kj < kw; kj++) {
                    const float *col_row = cb + col_idx * N;
                    const int w_base = kj * dw;

                    for (int i = 0; i < H_out; i++) {
                        float *xrow = xc + (h_base + i * sh) * W + w_base;
                        const float *src = col_row + i * W_out;
                        if (sw == 1) {
                            for (int j = 0; j < W_out; j++)
                                xrow[j] += src[j];
                        } else {
                            for (int j = 0; j < W_out; j++)
                                xrow[j * sw] += src[j];
                        }
                    }
                    col_idx++;
                }
            }
        }
    }
}

void col2im_f64(
    const double *__restrict__ cols,
    double *__restrict__ x,
    int B, int C, int H, int W,
    int kh, int kw,
    int sh, int sw,
    int dh, int dw,
    int H_out, int W_out
) {
    const int N = H_out * W_out;
    const int CKK = C * kh * kw;
    const int CHW = C * H * W;

    for (int b = 0; b < B; b++) {
        const double *cb = cols + b * CKK * N;
        double *xb = x + b * CHW;

        int col_idx = 0;
        for (int c = 0; c < C; c++) {
            double *xc = xb + c * H * W;
            for (int ki = 0; ki < kh; ki++) {
                const int h_base = ki * dh;
                for (int kj = 0; kj < kw; kj++) {
                    const double *col_row = cb + col_idx * N;
                    const int w_base = kj * dw;

                    for (int i = 0; i < H_out; i++) {
                        double *xrow = xc + (h_base + i * sh) * W + w_base;
                        const double *src = col_row + i * W_out;
                        if (sw == 1) {
                            for (int j = 0; j < W_out; j++)
                                xrow[j] += src[j];
                        } else {
                            for (int j = 0; j < W_out; j++)
                                xrow[j * sw] += src[j];
                        }
                    }
                    col_idx++;
                }
            }
        }
    }
}
