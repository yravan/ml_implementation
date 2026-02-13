/*
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
#include <float.h>
#include <omp.h>

/* ================================================================
 * im2col: (B, C, H, W) -> (B, C*kh*kw, H_out*W_out)
 * ================================================================ */

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
            float *gxbc = grad_x + b * C * H * W + c * H * W;              /* grad_x not x, not const */
            const float *gobc = grad_out + b * C * H_out * W_out + c * H_out * W_out;  /* grad_out not out, const */
            const long long *idxbc = indices + b * C * H_out * W_out + c * H_out * W_out;  /* const */
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    long long idx = idxbc[h * W_out + w];   /* semicolon */
                    gxbc[idx] += gobc[h * W_out + w];       /* semicolon */
                }
            }
        }
    }
}


/* ================================================================
 * Avg Pool 1D — forward
 * count_include_pad=1: divide by K always
 * count_include_pad=0: pass pad, orig_L; count real elements per window
 * ================================================================ */
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

/* ================================================================
 * Avg Pool 1D — backward
 * Distributes grad evenly across each window.
 * ================================================================ */
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

/* ================================================================
 * Avg Pool 2D — forward
 * ================================================================ */
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

/* ================================================================
 * Avg Pool 2D — backward
 * ================================================================ */
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

/* ================================================================
 * Adaptive Avg Pool 2D — forward
 * Uses PyTorch's floor-based bin edges:
 *   bin i covers [i * H / H_out, (i+1) * H / H_out)
 * ================================================================ */
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

/* ================================================================
 * Adaptive Avg Pool 2D — backward
 * ================================================================ */
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
