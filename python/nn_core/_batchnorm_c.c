/*
 * Fused BatchNorm kernels in C with OpenMP.
 *
 * All functions take a generic spatial size S:
 *   - BatchNorm1d with (B, C)    input: S = 1
 *   - BatchNorm1d with (B, C, L) input: S = L
 *   - BatchNorm2d with (B, C, H, W):    S = H * W
 *
 * Data layout: x[b * C * S + c * S + s]
 *
 * Forward training fuses:  mean + var (pass 1-2) → normalize + affine (pass 3)
 * Forward eval fuses:      normalize + affine (single pass)
 * Backward fuses:          grad_beta + grad_gamma + sums (pass 1) → grad_x (pass 2)
 *
 * Build: clang -O3 -mcpu=native -ffast-math -fno-finite-math-only
 *        -fopenmp -shared -fPIC -o _batchnorm_c.so _batchnorm_c.c
 */
#include <math.h>
#include <omp.h>

/* ================================================================
 * Forward — training mode
 *
 * Inputs:  x (B, C, S), gamma (C,), beta (C,)
 * Outputs: out (B, C, S), norm_x (B, C, S), mean (C,), var (C,)
 *
 * 3 passes per channel:
 *   1. accumulate sum        → mean
 *   2. accumulate (x-mean)²  → var
 *   3. normalize + scale + shift → out, norm_x
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

        /* --- Pass 1: mean --- */
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            const float *xbc = x + b * CS + c * S;
            for (int s = 0; s < S; s++) {
                sum += xbc[s];
            }
        }
        const float mean = sum * inv_N;
        mean_out[c] = mean;

        /* --- Pass 2: variance --- */
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

        /* --- Pass 3: normalize + affine --- */
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


/* ================================================================
 * Forward — eval mode
 *
 * Single pass: normalize with running stats + affine.
 * No norm_x output needed (no backward in eval).
 * ================================================================ */
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
        /* Precompute: out = (x - mean) * inv_std * g + bt
         *                 = x * (inv_std * g) + (bt - mean * inv_std * g) */
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


/* ================================================================
 * Backward
 *
 * Inputs:  grad_out (B,C,S), norm_x (B,C,S), gamma (C,), var (C,)
 * Outputs: grad_x (B,C,S), grad_gamma (C,), grad_beta (C,)
 *
 * Uses the fused formula (no intermediate arrays):
 *   grad_x = (gamma / (N * std)) * (N * dout - sum_dout - norm_x * sum_dout_nx)
 *
 * 2 passes per channel:
 *   1. sum_dout, sum_dout_nx, grad_gamma, grad_beta
 *   2. grad_x
 * ================================================================ */
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

        /* --- Pass 1: reduction sums --- */
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

        /* --- Pass 2: grad_x --- */
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