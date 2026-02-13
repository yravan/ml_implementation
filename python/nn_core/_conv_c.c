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
#include <omp.h>

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

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const float *xb = x + b * CHW;
        float *cb = cols + b * CKK * N;

        for (int c = 0; c < C; c++) {
            const float *xc = xb + c * H * W;
            for (int ki = 0; ki < kh; ki++) {
                const int h_base = ki * dh;
                for (int kj = 0; kj < kw; kj++) {
                    int col_idx = c * kh * kw + ki * kw + kj;
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

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const double *xb = x + b * CHW;
        double *cb = cols + b * CKK * N;

        for (int c = 0; c < C; c++) {
            const double *xc = xb + c * H * W;
            for (int ki = 0; ki < kh; ki++) {
                const int h_base = ki * dh;
                for (int kj = 0; kj < kw; kj++) {
                    int col_idx = c * kh * kw + ki * kw + kj;
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

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const float *cb = cols + b * CKK * N;
        float *xb = x + b * CHW;

        for (int c = 0; c < C; c++) {
            float *xc = xb + c * H * W;
            for (int ki = 0; ki < kh; ki++) {
                const int h_base = ki * dh;
                for (int kj = 0; kj < kw; kj++) {
                    int col_idx = c * kh * kw + ki * kw + kj;
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

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const double *cb = cols + b * CKK * N;
        double *xb = x + b * CHW;

        for (int c = 0; c < C; c++) {
            double *xc = xb + c * H * W;
            for (int ki = 0; ki < kh; ki++) {
                const int h_base = ki * dh;
                for (int kj = 0; kj < kw; kj++) {
                    int col_idx = c * kh * kw + ki * kw + kj;
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
                }
            }
        }
    }
}


/* ================================================================
 * Forward
 *
 * Parallel over B × C_out. Each thread computes one output channel
 * for one batch element — perfect load balance, zero conflicts.
 * ================================================================ */
void conv2d_forward_f32(
    const float *__restrict__ x,       /* (B, C, H, W) — already padded */
    const float *__restrict__ weight,  /* (C_out, Cg, kh, kw) */
    const float *__restrict__ bias,    /* (C_out,) or NULL */
    float *__restrict__ out,           /* (B, C_out, H_out, W_out) */
    int B, int C, int H, int W,
    int C_out, int kh, int kw,
    int sh, int sw, int dh, int dw,
    int H_out, int W_out,
    int groups
) {
    const int Cg  = C / groups;
    const int Cog = C_out / groups;
    const int HW  = H * W;
    const int ON  = H_out * W_out;
    const int KK  = kh * kw;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++) {
        for (int co = 0; co < C_out; co++) {
            const int g = co / Cog;
            const int ci_base = g * Cg;
            const float *w_co = weight + co * Cg * KK;
            float *out_co = out + b * C_out * ON + co * ON;
            const float bv = bias ? bias[co] : 0.0f;

            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = bv;
                    for (int ci = 0; ci < Cg; ci++) {
                        const float *xc = x + b * C * HW + (ci_base + ci) * HW;
                        const float *wc = w_co + ci * KK;
                        for (int ki = 0; ki < kh; ki++) {
                            const int ih = oh * sh + ki * dh;
                            const float *xrow = xc + ih * W;
                            const float *wrow = wc + ki * kw;
                            for (int kj = 0; kj < kw; kj++) {
                                sum += wrow[kj] * xrow[ow * sw + kj * dw];
                            }
                        }
                    }
                    out_co[oh * W_out + ow] = sum;
                }
            }
        }
    }
}


/* ================================================================
 * Backward: grad_weight
 *
 * Parallel over C_out. For each output channel, accumulate over
 * all (B, H_out, W_out) using Option B loop order: load each
 * grad_output element once, scatter-accumulate to the small
 * gw[co] buffer (Cg*kh*kw floats, fits in L1).
 * ================================================================ */
void conv2d_backward_weight_f32(
    const float *__restrict__ x,        /* (B, C, H, W) padded */
    const float *__restrict__ grad_out, /* (B, C_out, H_out, W_out) */
    float *__restrict__ grad_weight,    /* (C_out, Cg, kh, kw) */
    int B, int C, int H, int W,
    int C_out, int kh, int kw,
    int sh, int sw, int dh, int dw,
    int H_out, int W_out,
    int groups
) {
    const int Cg  = C / groups;
    const int Cog = C_out / groups;
    const int HW  = H * W;
    const int ON  = H_out * W_out;
    const int KK  = kh * kw;
    const int gw_size = Cg * KK;

    #pragma omp parallel for schedule(static)
    for (int co = 0; co < C_out; co++) {
        const int g = co / Cog;
        const int ci_base = g * Cg;
        float *gw = grad_weight + co * gw_size;

        /* Zero this output channel's weight gradient */
        memset(gw, 0, gw_size * sizeof(float));

        for (int b = 0; b < B; b++) {
            const float *go_bco = grad_out + b * C_out * ON + co * ON;

            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    const float g_val = go_bco[oh * W_out + ow];

                    for (int ci = 0; ci < Cg; ci++) {
                        const float *xc = x + b * C * HW + (ci_base + ci) * HW;
                        float *gwc = gw + ci * KK;
                        for (int ki = 0; ki < kh; ki++) {
                            const int ih = oh * sh + ki * dh;
                            const float *xrow = xc + ih * W;
                            float *gwrow = gwc + ki * kw;
                            for (int kj = 0; kj < kw; kj++) {
                                gwrow[kj] += g_val * xrow[ow * sw + kj * dw];
                            }
                        }
                    }
                }
            }
        }
    }
}


/* ================================================================
 * Backward: grad_input
 *
 * Parallel over B — each batch element writes to independent memory.
 * For each (co, oh, ow), scatter weight * grad_out to input positions.
 * Output array MUST be pre-zeroed by caller.
 * ================================================================ */
void conv2d_backward_input_f32(
    const float *__restrict__ weight,   /* (C_out, Cg, kh, kw) */
    const float *__restrict__ grad_out, /* (B, C_out, H_out, W_out) */
    float *__restrict__ grad_x,         /* (B, C, H, W) — pre-zeroed */
    int B, int C, int H, int W,
    int C_out, int kh, int kw,
    int sh, int sw, int dh, int dw,
    int H_out, int W_out,
    int groups
) {
    const int Cg  = C / groups;
    const int Cog = C_out / groups;
    const int HW  = H * W;
    const int ON  = H_out * W_out;
    const int KK  = kh * kw;

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        float *gx_b = grad_x + b * C * HW;

        for (int co = 0; co < C_out; co++) {
            const int g = co / Cog;
            const int ci_base = g * Cg;
            const float *w_co = weight + co * Cg * KK;
            const float *go_bco = grad_out + b * C_out * ON + co * ON;

            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    const float g_val = go_bco[oh * W_out + ow];

                    for (int ci = 0; ci < Cg; ci++) {
                        float *gxc = gx_b + (ci_base + ci) * HW;
                        const float *wc = w_co + ci * KK;
                        for (int ki = 0; ki < kh; ki++) {
                            const int ih = oh * sh + ki * dh;
                            float *gxrow = gxc + ih * W;
                            const float *wrow = wc + ki * kw;
                            for (int kj = 0; kj < kw; kj++) {
                                gxrow[ow * sw + kj * dw] += wrow[kj] * g_val;
                            }
                        }
                    }
                }
            }
        }
    }
}
