#include <math.h>
#include <omp.h>

void sgd_step_f32(
    float *__restrict__ params,
    const float *__restrict__ grads,
    float *__restrict__ velocity,
    int N,
    float lr, float momentum, float dampening, float nesterov,
    float weight_decay                    /* no trailing comma */
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float g = grads[i];               /* semicolon */
        if (weight_decay > 1e-8f)
            g += weight_decay * params[i];
        velocity[i] *= momentum;           /* semicolon */
        velocity[i] += g * (1.0f - dampening);  /* semicolon, 1.0f not 1 */
        if (nesterov > 0.5f)              /* no colon */
            params[i] -= (velocity[i] * momentum + g) * lr;  /* semicolon */
        else
            params[i] -= velocity[i] * lr;  /* semicolon */
    }
}


void sgdw_step_f32(
    float *__restrict__ params,
    const float *__restrict__ grads,
    float *__restrict__ velocity,
    int N,
    float lr, float momentum, float dampening, float nesterov,
    float weight_decay                    /* no trailing comma */
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        if (weight_decay > 1e-8f)
            params[i] *= (1 - lr * weight_decay);
        velocity[i] *= momentum;           /* semicolon */
        velocity[i] += grads[i] * (1.0f - dampening);  /* semicolon, 1.0f not 1 */
        if (nesterov > 0.5f)              /* no colon */
            params[i] -= (velocity[i] * momentum + grads[i]) * lr;  /* semicolon */
        else
            params[i] -= velocity[i] * lr;  /* semicolon */
    }
}


void adam_step_f32(
    float *__restrict__ params,
    const float *__restrict__ grads,
    float *__restrict__ exp_avg,
    float *__restrict__ exp_avg_sq,
    int N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float g = grads[i];
        if (weight_decay > 1e-8f)
            g += weight_decay * params[i];

        exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * grads[i];
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * grads[i] * grads[i];

        float m_hat = exp_avg[i] / bc1;
        float v_hat = exp_avg_sq[i] / bc2;

        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adamw_step_f32(
    float *__restrict__ params,
    const float *__restrict__ grads,
    float *__restrict__ exp_avg,
    float *__restrict__ exp_avg_sq,
    int N,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        if (weight_decay > 1e-8f)
            params[i] *= (1.0f - weight_decay * lr);

        exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * grads[i];
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * grads[i] * grads[i];

        float m_hat = exp_avg[i] / bc1;
        float v_hat = exp_avg_sq[i] / bc2;

        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}