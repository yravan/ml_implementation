/*
 * Unified C extension for nn_core.
 *
 * Combines all compute kernels:
 *   - im2col / col2im  (1D + 2D, float32 + float64)
 *   - conv2d forward / backward_weight / backward_input
 *   - max pool 1D / 2D  (forward + backward)
 *   - avg pool 1D / 2D  (forward + backward)
 *   - adaptive avg pool 2D  (forward + backward)
 *   - batchnorm forward_train / forward_eval / backward
 *
 * Data layout: x[b * C * S + c * S + s]   (S = spatial product)
 *
 * Build:
 *   Linux : gcc -O3 -march=native -ffast-math -fno-finite-math-only
 *           -fopenmp -shared -fPIC -o _nn_c.so _nn_c.c -lm
 *   macOS : clang -O3 -mcpu=native -ffast-math -fno-finite-math-only
 *           -Xpreprocessor -fopenmp -I... -L... -lomp
 *           -shared -fPIC -o _nn_c.so _nn_c.c
 */
#include <math.h>
#include <string.h>
#include <float.h>
#include <omp.h>


/* ================================================================
 *  SECTION 1 — im2col / col2im
 * ================================================================ */

/* ---- 2D im2col: (B,C,H,W) -> (B, C*kh*kw, H_out*W_out) ---- */

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


/* ---- 2D col2im: (B, C*kh*kw, H_out*W_out) -> (B,C,H,W)
 *      Output MUST be pre-zeroed by caller. ---- */

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


/* ---- 1D im2col: (B,C,L) -> (B, C*K, L_out) ---- */

void im2col1d_f32(
    const float *__restrict__ x,
    float *__restrict__ cols,
    int B, int C, int L,
    int K,
    int s,
    int d,
    int L_out
) {
    const int CK = C * K;

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const float *xb = x + b * C * L;
        float *cb = cols + b * CK * L_out;
        for (int c = 0; c < C; c++) {
            const float *xc = xb + c * L;
            for (int k = 0; k < K; k++) {
                float *out = cb + (c * K + k) * L_out;
                if (s == 1 && d == 1) {
                    memcpy(out, xc + k, L_out * sizeof(float));
                } else {
                    for (int l = 0; l < L_out; l++) {
                        out[l] = xc[l * s + k * d];
                    }
                }
            }
        }
    }
}

void im2col1d_f64(
    const double *__restrict__ x,
    double *__restrict__ cols,
    int B, int C, int L,
    int K,
    int s,
    int d,
    int L_out
) {
    const int CK = C * K;

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const double *xb = x + b * C * L;
        double *cb = cols + b * CK * L_out;
        for (int c = 0; c < C; c++) {
            const double *xc = xb + c * L;
            for (int k = 0; k < K; k++) {
                double *out = cb + (c * K + k) * L_out;
                if (s == 1 && d == 1) {
                    memcpy(out, xc + k, L_out * sizeof(double));
                } else {
                    for (int l = 0; l < L_out; l++) {
                        out[l] = xc[l * s + k * d];
                    }
                }
            }
        }
    }
}


/* ---- 1D col2im: (B, C*K, L_out) -> (B,C,L)
 *      Output MUST be pre-zeroed by caller. ---- */

void col2im1d_f32(
    const float *__restrict__ cols,
    float *__restrict__ x,
    int B, int C, int L,
    int K,
    int s,
    int d,
    int L_out
) {
    const int CK = C * K;

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const float *cb = cols + b * CK * L_out;
        float *xb = x + b * C * L;
        for (int c = 0; c < C; c++) {
            float *xc = xb + c * L;
            for (int k = 0; k < K; k++) {
                const float *src = cb + (c * K + k) * L_out;
                if (s == 1 && d == 1) {
                    float *dst = xc + k;
                    for (int l = 0; l < L_out; l++) {
                        dst[l] += src[l];
                    }
                } else {
                    for (int l = 0; l < L_out; l++) {
                        xc[l * s + k * d] += src[l];
                    }
                }
            }
        }
    }
}

void col2im1d_f64(
    const double *__restrict__ cols,
    double *__restrict__ x,
    int B, int C, int L,
    int K,
    int s,
    int d,
    int L_out
) {
    const int CK = C * K;

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const double *cb = cols + b * CK * L_out;
        double *xb = x + b * C * L;
        for (int c = 0; c < C; c++) {
            double *xc = xb + c * L;
            for (int k = 0; k < K; k++) {
                const double *src = cb + (c * K + k) * L_out;
                if (s == 1 && d == 1) {
                    double *dst = xc + k;
                    for (int l = 0; l < L_out; l++) {
                        dst[l] += src[l];
                    }
                } else {
                    for (int l = 0; l < L_out; l++) {
                        xc[l * s + k * d] += src[l];
                    }
                }
            }
        }
    }
}


/* ================================================================
 *  SECTION 2 — Conv2d forward / backward
 * ================================================================ */

void conv2d_forward_f32(
    const float *__restrict__ x,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ out,
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

void conv2d_backward_weight_f32(
    const float *__restrict__ x,
    const float *__restrict__ grad_out,
    float *__restrict__ grad_weight,
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

void conv2d_backward_input_f32(
    const float *__restrict__ weight,
    const float *__restrict__ grad_out,
    float *__restrict__ grad_x,
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


/* ================================================================
 *  SECTION 3 — Pooling
 * ================================================================ */

/* ---- Max Pool 1D ---- */

void max_pool1d(
    const float *__restrict__ x,
    float *__restrict__ out,
    long long *__restrict__ indices,
    int B, int C, int L,
    int K, int S, int D,
    int L_out
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *xbc = x + b * C * L + c * L;
            float *outbc = out + b * C * L_out + c * L_out;
            long long *idxbc = indices + b * C * L_out + c * L_out;
            for (int l = 0; l < L_out; l++) {
                float maxval = xbc[l * S];
                int maxidx = l * S;
                for (int k = 1; k < K; k++) {
                    int pos = l * S + k * D;
                    float val = xbc[pos];
                    if (val > maxval) {
                        maxval = val;
                        maxidx = pos;
                    }
                }
                outbc[l] = maxval;
                idxbc[l] = maxidx;
            }
        }
    }
}

void max_pool1d_backward(
    const float *__restrict__ grad_out,
    float *__restrict__ grad_x,
    const long long *__restrict__ indices,
    int B, int C, int L,
    int L_out
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float *grad_xbc = grad_x + b * C * L + c * L;
            const float *grad_outbc = grad_out + b * C * L_out + c * L_out;
            const long long *idxbc = indices + b * C * L_out + c * L_out;
            for (int l = 0; l < L_out; l++) {
                int idx = idxbc[l];
                grad_xbc[idx] += grad_outbc[l];
            }
        }
    }
}

/* ---- Max Pool 2D ---- */

void max_pool2d(
    const float *__restrict__ x,
    float *__restrict__ out,
    long long *__restrict__ indices,
    int B, int C, int H, int W,
    int Kh, int Kw, int Sh, int Sw, int Dh, int Dw,
    int H_out, int W_out
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *xbc = x + b * C * H * W + c * H * W;
            float *outbc = out + b * C * H_out * W_out + c * H_out * W_out;
            long long *idxbc = indices + b * C * H_out * W_out + c * H_out * W_out;
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    const float *xbchw = xbc + h * Sh * W + w * Sw;
                    float max = xbchw[0];
                    long long maxidx = h * Sh * W + w * Sw;
                    for (int kh = 0; kh < Kh; kh++){
                        for (int kw = 0; kw < Kw; kw++){
                            float val = xbchw[kh * Dh * W + kw * Dw];
                            if (val > max){
                                max = val;
                                maxidx = h * Sh * W + w * Sw + kh * Dh * W + kw * Dw;
                            }
                        }
                    }
                    idxbc[h * W_out + w] = maxidx;
                    outbc[h * W_out + w] = max;
                }
            }
        }
    }
}

void max_pool2d_backward(
    const float *__restrict__ grad_out,
    float *__restrict__ grad_x,
    const long long *__restrict__ indices,
    int B, int C, int H, int W,
    int H_out, int W_out
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float *gxbc = grad_x + b * C * H * W + c * H * W;
            const float *gobc = grad_out + b * C * H_out * W_out + c * H_out * W_out;
            const long long *idxbc = indices + b * C * H_out * W_out + c * H_out * W_out;
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    long long idx = idxbc[h * W_out + w];
                    gxbc[idx] += gobc[h * W_out + w];
                }
            }
        }
    }
}

/* ---- Avg Pool 1D ---- */

void avg_pool1d(
    const float *__restrict__ x,
    float *__restrict__ out,
    int B, int C, int L,
    int K, int S,
    int L_out,
    int count_include_pad, int pad, int orig_L
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *xbc = x + b * C * L + c * L;
            float *outbc = out + b * C * L_out + c * L_out;
            for (int l = 0; l < L_out; l++) {
                float sum = 0.0f;
                int count = 0;
                for (int k = 0; k < K; k++) {
                    int pos = l * S + k;
                    sum += xbc[pos];
                    if (count_include_pad || (pos >= pad && pos < pad + orig_L)) {
                        count++;
                    }
                }
                outbc[l] = sum / (float)count;
            }
        }
    }
}

void avg_pool1d_backward(
    const float *__restrict__ grad_out,
    float *__restrict__ grad_x,
    int B, int C, int L,
    int K, int S,
    int L_out,
    int count_include_pad, int pad, int orig_L
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float *gxbc = grad_x + b * C * L + c * L;
            const float *gobc = grad_out + b * C * L_out + c * L_out;
            for (int l = 0; l < L_out; l++) {
                int count = 0;
                if (count_include_pad) {
                    count = K;
                } else {
                    for (int k = 0; k < K; k++) {
                        int pos = l * S + k;
                        if (pos >= pad && pos < pad + orig_L) count++;
                    }
                }
                float val = gobc[l] / (float)count;
                for (int k = 0; k < K; k++) {
                    gxbc[l * S + k] += val;
                }
            }
        }
    }
}

/* ---- Avg Pool 2D ---- */

void avg_pool2d(
    const float *__restrict__ x,
    float *__restrict__ out,
    int B, int C, int H, int W,
    int Kh, int Kw, int Sh, int Sw,
    int H_out, int W_out,
    int count_include_pad, int Ph, int Pw, int orig_H, int orig_W
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *xbc = x + b * C * H * W + c * H * W;
            float *outbc = out + b * C * H_out * W_out + c * H_out * W_out;
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < Kh; kh++) {
                        int ih = h * Sh + kh;
                        for (int kw = 0; kw < Kw; kw++) {
                            int iw = w * Sw + kw;
                            sum += xbc[ih * W + iw];
                            if (count_include_pad ||
                                (ih >= Ph && ih < Ph + orig_H &&
                                 iw >= Pw && iw < Pw + orig_W)) {
                                count++;
                            }
                        }
                    }
                    outbc[h * W_out + w] = sum / (float)count;
                }
            }
        }
    }
}

void avg_pool2d_backward(
    const float *__restrict__ grad_out,
    float *__restrict__ grad_x,
    int B, int C, int H, int W,
    int Kh, int Kw, int Sh, int Sw,
    int H_out, int W_out,
    int count_include_pad, int Ph, int Pw, int orig_H, int orig_W
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float *gxbc = grad_x + b * C * H * W + c * H * W;
            const float *gobc = grad_out + b * C * H_out * W_out + c * H_out * W_out;
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    int count = 0;
                    if (count_include_pad) {
                        count = Kh * Kw;
                    } else {
                        for (int kh = 0; kh < Kh; kh++) {
                            int ih = h * Sh + kh;
                            for (int kw = 0; kw < Kw; kw++) {
                                int iw = w * Sw + kw;
                                if (ih >= Ph && ih < Ph + orig_H &&
                                    iw >= Pw && iw < Pw + orig_W) {
                                    count++;
                                }
                            }
                        }
                    }
                    float val = gobc[h * W_out + w] / (float)count;
                    for (int kh = 0; kh < Kh; kh++) {
                        for (int kw = 0; kw < Kw; kw++) {
                            gxbc[(h * Sh + kh) * W + (w * Sw + kw)] += val;
                        }
                    }
                }
            }
        }
    }
}

/* ---- Adaptive Avg Pool 2D ---- */

void adaptive_avg_pool2d(
    const float *__restrict__ x,
    float *__restrict__ out,
    int B, int C, int H, int W,
    int H_out, int W_out
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *xbc = x + b * C * H * W + c * H * W;
            float *outbc = out + b * C * H_out * W_out + c * H_out * W_out;
            for (int i = 0; i < H_out; i++) {
                int h_start = i * H / H_out;
                int h_end = (i + 1) * H / H_out;
                for (int j = 0; j < W_out; j++) {
                    int w_start = j * W / W_out;
                    int w_end = (j + 1) * W / W_out;
                    float sum = 0.0f;
                    for (int h = h_start; h < h_end; h++) {
                        for (int w = w_start; w < w_end; w++) {
                            sum += xbc[h * W + w];
                        }
                    }
                    outbc[i * W_out + j] = sum / (float)((h_end - h_start) * (w_end - w_start));
                }
            }
        }
    }
}

void adaptive_avg_pool2d_backward(
    const float *__restrict__ grad_out,
    float *__restrict__ grad_x,
    int B, int C, int H, int W,
    int H_out, int W_out
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float *gxbc = grad_x + b * C * H * W + c * H * W;
            const float *gobc = grad_out + b * C * H_out * W_out + c * H_out * W_out;
            for (int i = 0; i < H_out; i++) {
                int h_start = i * H / H_out;
                int h_end = (i + 1) * H / H_out;
                for (int j = 0; j < W_out; j++) {
                    int w_start = j * W / W_out;
                    int w_end = (j + 1) * W / W_out;
                    float count = (float)((h_end - h_start) * (w_end - w_start));
                    float val = gobc[i * W_out + j] / count;
                    for (int h = h_start; h < h_end; h++) {
                        for (int w = w_start; w < w_end; w++) {
                            gxbc[h * W + w] += val;
                        }
                    }
                }
            }
        }
    }
}


/* ================================================================
 *  SECTION 4 — BatchNorm
 * ================================================================ */

void batchnorm_forward_train(
    const float *__restrict__ x,
    float *__restrict__ out,
    float *__restrict__ norm_x,
    const float *__restrict__ gamma,
    const float *__restrict__ beta,
    float *__restrict__ mean_out,
    float *__restrict__ var_out,
    int B, int C, int S,
    float eps
) {
    const int CS = C * S;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c++) {
        const int N = B * S;
        const float inv_N = 1.0f / (float)N;

        /* Pass 1: mean */
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            const float *xbc = x + b * CS + c * S;
            for (int s = 0; s < S; s++) {
                sum += xbc[s];
            }
        }
        const float mean = sum * inv_N;
        mean_out[c] = mean;

        /* Pass 2: variance */
        float var_sum = 0.0f;
        for (int b = 0; b < B; b++) {
            const float *xbc = x + b * CS + c * S;
            for (int s = 0; s < S; s++) {
                float d = xbc[s] - mean;
                var_sum += d * d;
            }
        }
        const float var = var_sum * inv_N;
        var_out[c] = var;

        /* Pass 3: normalize + affine */
        const float inv_std = 1.0f / sqrtf(var + eps);
        const float g = gamma[c];
        const float bt = beta[c];
        for (int b = 0; b < B; b++) {
            const float *xbc = x  + b * CS + c * S;
            float *nxbc       = norm_x + b * CS + c * S;
            float *obc        = out    + b * CS + c * S;
            for (int s = 0; s < S; s++) {
                float nx = (xbc[s] - mean) * inv_std;
                nxbc[s] = nx;
                obc[s]  = nx * g + bt;
            }
        }
    }
}

void batchnorm_forward_eval(
    const float *__restrict__ x,
    float *__restrict__ out,
    const float *__restrict__ gamma,
    const float *__restrict__ beta,
    const float *__restrict__ running_mean,
    const float *__restrict__ running_var,
    int B, int C, int S,
    float eps
) {
    const int CS = C * S;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c++) {
        const float mean    = running_mean[c];
        const float inv_std = 1.0f / sqrtf(running_var[c] + eps);
        const float g       = gamma[c];
        const float bt      = beta[c];
        const float scale = inv_std * g;
        const float shift = bt - mean * scale;

        for (int b = 0; b < B; b++) {
            const float *xbc = x   + b * CS + c * S;
            float *obc       = out + b * CS + c * S;
            for (int s = 0; s < S; s++) {
                obc[s] = xbc[s] * scale + shift;
            }
        }
    }
}

void batchnorm_backward(
    const float *__restrict__ grad_out,
    const float *__restrict__ norm_x,
    const float *__restrict__ gamma,
    const float *__restrict__ var,
    float *__restrict__ grad_x,
    float *__restrict__ grad_gamma,
    float *__restrict__ grad_beta,
    int B, int C, int S,
    float eps
) {
    const int CS = C * S;
    const int N  = B * S;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c++) {

        /* Pass 1: reduction sums */
        float sum_go    = 0.0f;
        float sum_go_nx = 0.0f;
        for (int b = 0; b < B; b++) {
            const float *gobc = grad_out + b * CS + c * S;
            const float *nxbc = norm_x   + b * CS + c * S;
            for (int s = 0; s < S; s++) {
                float go = gobc[s];
                float nx = nxbc[s];
                sum_go    += go;
                sum_go_nx += go * nx;
            }
        }
        grad_beta[c]  = sum_go;
        grad_gamma[c] = sum_go_nx;

        /* Pass 2: grad_x */
        const float coeff = gamma[c] / ((float)N * sqrtf(var[c] + eps));
        for (int b = 0; b < B; b++) {
            const float *gobc = grad_out + b * CS + c * S;
            const float *nxbc = norm_x   + b * CS + c * S;
            float       *gxbc = grad_x   + b * CS + c * S;
            for (int s = 0; s < S; s++) {
                gxbc[s] = coeff * (
                    (float)N * gobc[s] - sum_go - nxbc[s] * sum_go_nx
                );
            }
        }
    }
}
