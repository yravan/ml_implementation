"""
Diffusion Models Module
=======================

Comprehensive implementation of diffusion-based generative models including:

1. NOISE SCHEDULE (noise_schedule.py)
   - Linear, cosine, quadratic, exponential schedules
   - Pre-computed coefficients for efficiency
   - Signal-to-noise ratio analysis

2. DDPM - Denoising Diffusion Probabilistic Models (ddpm.py)
   - Forward process: add noise step-by-step
   - Reverse process: learn to denoise
   - Training via noise prediction loss
   - Sampling via iterative denoising
   - Posterior computation and KL divergence

3. DDIM - Fast Sampling (ddim.py)
   - Accelerated sampling using subset of timesteps
   - 10-50x speedup vs DDPM (1000→50 steps)
   - Deterministic sampling for reproducibility
   - Latent space interpolation and editing

4. SCORE MATCHING (score_matching.py)
   - Score function: gradient of log-density
   - Training via denoising score matching
   - SDE framework for continuous-time models
   - Likelihood computation via probability flow ODE
   - VP (variance-preserving) and VE (variance-exploding) schedules

5. CLASSIFIER-FREE GUIDANCE (classifier_free_guidance.py)
   - Conditional generation without separate classifier
   - Train on both conditional and unconditional objectives
   - Guided sampling with adjustable strength
   - Negative prompt support
   - Multi-modal condition support

6. LATENT DIFFUSION (latent_diffusion.py)
   - Diffuse in compressed latent space (10-50x faster)
   - VAE encoder/decoder for efficient compression
   - Enables high-resolution generation
   - Text-to-image generation
   - Inpainting and image editing

QUICK START:
============

Training DDPM:
    config = DDPMConfig(num_steps=1000)
    model = DDPM(denoiser_network, config)

    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = model.training_step(batch, optimizer)
            loss.backward()
            optimizer.step()

Sampling:
    with torch.no_grad():
        samples = model.sample(batch_size=32, sample_shape=(3, 64, 64))

Text-to-Image with Latent Diffusion:
    model = TextToImageDiffusion(denoiser, vae, text_encoder)
    images = model.generate_from_text(
        "a cat sitting on a chair",
        height=512, width=512,
        guidance_scale=7.5
    )

Fast Sampling with DDIM:
    ddim_model = DDIM(denoiser, DDIMConfig(num_sample_steps=50))
    with torch.no_grad():
        samples = ddim_model.sample_fast(batch_size=32, sample_shape=(3, 64, 64))
        # 50 steps instead of 1000, much faster!

THEORETICAL FOUNDATIONS:
==========================

Forward Diffusion (Q):
    q(x_t|x_0) = N(x_t; √ᾱ_t * x_0, (1-ᾱ_t) * I)
    Gradually add noise to data over T steps

Reverse Process (P):
    p_θ(x_{t-1}|x_t) = N(μ_θ(x_t, t), σ_t² * I)
    Learn to denoise in reverse direction

Training:
    L = E[||ε - ε_θ(x_t, t)||²_2]
    Predict the noise added in forward process

Sampling:
    Start with x_T ~ N(0,I), iteratively denoise to x_0

KEY PAPERS:
===========

[1] Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
    https://arxiv.org/abs/2006.11239
    - Foundation of modern diffusion models
    - Noise prediction objective
    - DDPM algorithm

[2] Song et al., "Denoising Diffusion Implicit Models" (2021)
    https://arxiv.org/abs/2010.02502
    - DDIM for fast sampling
    - Generalized reverse process

[3] Song et al., "Score-Based Generative Modeling through SDEs" (2021)
    https://arxiv.org/abs/2011.13456
    - Score-based perspective
    - Continuous-time models
    - Probability flow ODE

[4] Ho & Salimans, "Classifier-Free Diffusion Guidance" (2021)
    https://arxiv.org/abs/2207.12598
    - Conditional generation without classifier
    - Guidance mechanism

[5] Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion" (2022)
    https://arxiv.org/abs/2112.10752
    - Stable Diffusion
    - Latent space diffusion
    - Text-to-image generation

COMPARISON WITH OTHER GENERATIVE MODELS:
========================================

vs GANs:
  + Easier to train (more stable)
  + Better mode coverage (less mode collapse)
  - Slower sampling (iterative vs. single forward pass)
  - Worse sample quality at same model size (recently improving)

vs VAEs:
  + Better sample quality
  + More flexible (don't require explicit KL bound)
  - Slower sampling
  - Less interpretable latent space

vs Autoregressive (GPT, etc):
  + Parallelizable (can sample all timesteps)
  + Better for images
  - Slower than autoregressive for text

TRAINING TIPS:
==============

1. Noise Schedule:
   - Linear often works, cosine slightly better
   - Longer T (1000) better quality, slower sampling
   - 500-1000 typical for most applications

2. Loss Weighting:
   - Uniform: simplest
   - SNR-weighted: balances timesteps
   - Min-SNR: handles high-variance timesteps

3. Guidance Scale:
   - 0: unconditional
   - 1-5: gentle guidance
   - 7-15: strong guidance
   - >15: mode collapse risk

4. Learning Rate:
   - Warmup recommended (e.g., 1000 steps)
   - Cosine decay schedule
   - Typical: 1e-4 for images

5. Batch Size:
   - Larger batches help (32-256 typical)
   - Accumulate gradients if GPU memory limited

SAMPLING SPEEDUPS:
==================

1. DDIM (10-50x faster)
   - Swap DDPM sampling for DDIM
   - Minimal quality loss with ~50 steps

2. Latent Diffusion (10-50x faster)
   - Diffuse in compressed space
   - Requires VAE pre-training

3. Combined (100-500x speedup!)
   - Latent diffusion + DDIM
   - Enables real-time generation

4. Quantization & Distillation
   - Model compression techniques
   - Further speedups possible

MODULES & CLASSES:
==================
"""

from .noise_schedule import (
    NoiseSchedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    QuadraticNoiseSchedule,
    ExponentialNoiseSchedule,
)

from .ddpm import (
    DDPM,
    DDPMConfig,
)

from .ddim import (
    DDIM,
    DDIMConfig,
)

from .score_matching import (
    ScoreBasedGenerativeModel,
    ScoreMatchingConfig,
    VPSchedule,
    VESchedule,
)

from .classifier_free_guidance import (
    ConditionalDiffusionModel,
    ClassifierFreeGuidanceConfig,
    MultiConditionDiffusion,
)

from .latent_diffusion import (
    LatentDiffusionModel,
    LatentDiffusionConfig,
    VariationalAutoencoder,
    TextToImageDiffusion,
)

__all__ = [
    # Noise Schedules
    "NoiseSchedule",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule",
    "QuadraticNoiseSchedule",
    "ExponentialNoiseSchedule",

    # DDPM
    "DDPM",
    "DDPMConfig",

    # DDIM
    "DDIM",
    "DDIMConfig",

    # Score-Based
    "ScoreBasedGenerativeModel",
    "ScoreMatchingConfig",
    "VPSchedule",
    "VESchedule",

    # Classifier-Free Guidance
    "ConditionalDiffusionModel",
    "ClassifierFreeGuidanceConfig",
    "MultiConditionDiffusion",

    # Latent Diffusion
    "LatentDiffusionModel",
    "LatentDiffusionConfig",
    "VariationalAutoencoder",
    "TextToImageDiffusion",
]
