"""
Classifier-Free Guidance for Conditional Diffusion Models
===========================================================

Overview:
---------
Classifier-free guidance enables conditional generation from diffusion models
WITHOUT requiring a separate classifier. Instead, we train the diffusion model
with unconditional and conditional targets, then use both during sampling to
guide generation toward a desired condition.

Key Innovation: By training on both conditional (with label/prompt) and
unconditional (without) inputs, we can steer generation without external classifiers.

MOTIVATION:
===========

In conditional generation, we want to sample from p(x|c) where c is a condition
(e.g., class label, text prompt, image).

Traditional approaches:
    1. CLASSIFIER GUIDANCE: Use a separate trained classifier p(c|x) to guide sampling
       Problem: Requires training extra classifier; can be adversarial

    2. CONDITIONAL DIFFUSION: Train p_θ(x_{t-1}|x_t, c) with condition as input
       Problem: Doesn't naturally enable "stronger" conditioning

CLASSIFIER-FREE GUIDANCE:
=========================

Solution: Train ONE network on BOTH conditional AND unconditional tasks:

    Training:
        p_θ(x_{t-1}|x_t, c) trained on: (x_t, c) pairs
        p_θ(x_{t-1}|x_t)    trained on: (x_t, null) pairs

    During training, randomly set c = null for unconditional objective.

    Loss:
        L_total = E_{c} L_θ(x_t, x_{t-1}, c) + E_null L_θ(x_t, x_{t-1}, null)

Where null is a special "no condition" token.

SAMPLING WITH GUIDANCE:
=======================

At sampling time, we use BOTH predictions to steer generation:

Let:
    ε_θ(x_t, t, c) = noise prediction conditioned on c
    ε_θ(x_t, t, ∅) = unconditional noise prediction

Modified noise prediction (with guidance scale w):
    ε̃_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + w * (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))

    Where:
    - w = 0: Unconditional generation
    - w = 1: Standard conditional generation (no guidance)
    - w > 1: Stronger conditioning, pushes toward c

Intuition:
    - (ε_θ(x_t, t, ∅)): "Base" unconditional direction
    - (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅)): "Difference" from unconditional when c is known
    - w * difference: Amplify the conditional signal

CONNECTION TO CLASSIFIER GUIDANCE:
==================================

Classifier guidance (old approach) modifies the score:
    ∇_x log p(x|c) = ∇_x log p(x) + w * ∇_x log p(c|x)

Where the second term comes from a trained classifier.

Classifier-FREE guidance approximates this WITHOUT a classifier:
    ∇_x log p(x|c) ≈ ∇_x log p(x|∅) + w * (∇_x log p(x|c) - ∇_x log p(x|∅))

Key: The difference (conditional - unconditional) acts as a learned "classifier signal"!

MATHEMATICAL FORMULATION:
=========================

In score-based terms (with score = -ε/σ):
    s_θ(x_t, t, c) = unconditional score
    s_θ(x_t, t, c) = conditional score

Guided score:
    s̃_θ(x_t, t, c) = s_θ(x_t, t, ∅) + w * (s_θ(x_t, t, c) - s_θ(x_t, t, ∅))

This gives the update:
    x_{t-1} = x_t + (∇t/2) * s̃_θ(x_t, t, c) + √(∇t) * z

GUIDANCE STRENGTHS:
===================

w = 0.0: Pure unconditional generation
w = 1.0: No guidance (just conditional model)
w ∈ (1, 2): Mild guidance, balanced quality and adherence to condition
w ∈ (2, 5): Strong guidance, high condition adherence
w > 5.0: Very strong guidance, may reduce diversity, possible artifacts
w >> 1.0: Degenerate mode collapse, poor sample quality

The trade-off: Higher w increases adherence to condition but reduces sample quality.

TRAINING DETAILS:
=================

Unconditional dropout: During training, replace condition c with null token
with probability p_uncond (typically 0.1 or 0.2).

Data augmentation:
    For each batch:
        - Half: Use actual condition c
        - Half: Use null condition ∅ (unconditional)

    Or: Randomly drop condition with probability p_uncond for each sample

Loss:
    L = E[L_mse(ε_θ(x_t, t, c), ε_true)] when c is provided
    + E[L_mse(ε_θ(x_t, t, ∅), ε_true)] when c is dropped/null

Network architecture:
    The network sees:
        - Input: x_t, t, c (where c can be embedding of label/prompt)
        - If c is null: use special null_embedding or zero-out c input

CONDITION TYPES:
================

1. CLASS LABELS (categorical):
    c = one-hot or embedding of class
    Easy to implement, well-studied

2. TEXT PROMPTS (semantic):
    c = text encoder output (CLIP, BERT, etc.)
    More flexible, enables fine-grained control
    Used in Stable Diffusion, DALL-E 2

3. IMAGES (visual):
    c = encoder output of reference image
    Used for inpainting, style transfer, etc.

4. CONTINUOUS ATTRIBUTES:
    c = continuous values (e.g., brightness, size)
    Direct numerical input to network

5. STRUCTURED CONDITIONS:
    c = multiple components (label + text + image)
    Concatenate or process separately

NEGATIVE PROMPTS:
=================

Modern extension: Negative prompts to steer away from certain content.

With negative prompt c_neg:
    ε̃_θ = ε_θ(x_t, t, ∅)
         + w_pos * (ε_θ(x_t, t, c_pos) - ε_θ(x_t, t, ∅))
         - w_neg * (ε_θ(x_t, t, c_neg) - ε_θ(x_t, t, ∅))

Where:
    - w_pos: Positive prompt weight (push toward c_pos)
    - w_neg: Negative prompt weight (push away from c_neg)

Common usage: c_neg = "blurry, low quality, artifact"

ADVANTAGES:
===========
1. NO EXTRA CLASSIFIER: All info in diffusion model
2. FLEXIBLE: Works with any condition type
3. TRAINING SIMPLE: Just unconditional dropout
4. SAMPLING FLEXIBLE: Adjust w at test time
5. WORKS WELL: Effective guidance in practice

DISADVANTAGES:
==============
1. Requires unconditional training (2x memory/compute during training)
2. Quality-guidance trade-off (higher w → worse quality sometimes)
3. Requires careful tuning of w
4. Can produce mode-collapse at high w

REFERENCES:
-----------
[1] "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2021)
    https://arxiv.org/abs/2207.12598
    Introduced classifier-free guidance

[2] "Stable Diffusion" (Rombach et al., 2022)
    https://arxiv.org/abs/2112.10752
    Large-scale application of classifier-free guidance

[3] "DALL-E 2" (Ramesh et al., 2022)
    https://cdn.openai.com/papers/DALL-E_2.pdf
    Sophisticated guidance with negative prompts
"""

import numpy as np
from typing import Optional, Union, Dict, Tuple
from dataclasses import dataclass

from python.nn_core import Module
from ddpm import DDPM, DDPMConfig


@dataclass
class ClassifierFreeGuidanceConfig(DDPMConfig):
    """Configuration for classifier-free guidance in diffusion models."""

    # Guidance
    use_classifier_free_guidance: bool = True
    unconditional_dropout_prob: float = 0.1  # Probability of dropping condition during training

    # Condition
    condition_embedding_dim: Optional[int] = None  # Dimension of condition embedding (if not using raw condition)
    condition_type: str = "embedding"  # "embedding", "label", "text", "image"

    # Sampling
    guidance_scale: float = 7.5  # w in the formula: ε̃ = ε_uncond + w * (ε_cond - ε_uncond)
    negative_prompt_weight: float = 0.0  # Weight for negative prompt steering


class ConditionalDiffusionModel(DDPM):
    """
    Conditional diffusion model base class.

    Extended from DDPM to support conditions c during training and sampling.
    """

    def __init__(
        self,
        denoiser: Module,
        config: ClassifierFreeGuidanceConfig = ClassifierFreeGuidanceConfig()
    ):
        """
        Initialize conditional diffusion model.

        Args:
            denoiser: Network that predicts ε_θ(x_t, t, c)
                      Should accept (x_t, t, c) and output noise prediction
            config: Configuration
        """
        super().__init__(denoiser, config)
        self.config = config

    def forward(
        self,
        x_t: np.ndarray,
        t: np.ndarray,
        condition: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass of the conditional denoiser.

        Args:
            x_t: Noisy sample
            t: Timestep
            condition: Condition vector c (e.g., label embedding, text encoding)

        Returns:
            Predicted noise ε_θ(x_t, t, c)
        """
        raise NotImplementedError(
            "Call self.denoiser(x_t, t, condition). "
            "Handle case where condition is None (unconditional)."
        )

    def training_step(
        self,
        x0: np.ndarray,
        condition: np.ndarray,
        optimizer: Optional[object] = None
    ) -> np.ndarray:
        """
        Training step with classifier-free guidance.

        Algorithm:
        1. Sample timesteps t uniformly
        2. Sample noise ε
        3. Compute x_t = forward_process(x0, t, ε)
        4. With probability p_uncond:
           - Predict: ε_θ(x_t, t, null)    [unconditional]
        5. Otherwise:
           - Predict: ε_θ(x_t, t, condition) [conditional]
        6. Compute loss: ||ε - ε_θ||²
        7. Return loss

        Args:
            x0: Clean samples
            condition: Condition vectors
            optimizer: Optimizer instance

        Returns:
            Loss for backprop
        """
        raise NotImplementedError(
            "Sample timesteps and noise. "
            "With probability p_uncond, set condition to null_embedding. "
            "Call forward(x_t, t, condition) to predict noise. "
            "Compute MSE loss and return."
        )

    def get_unconditional_embedding(
        self,
        batch_size: int
    ) -> np.ndarray:
        """
        Get null/unconditional embedding.

        Returns a special embedding that represents "no condition".

        Options:
            1. Zero vector: zeros(batch_size, condition_dim)
            2. Learned null token: self.null_token (trainable parameter)
            3. Special token from encoder: encoder(null_text)

        Args:
            batch_size: Batch size

        Returns:
            Null embedding, shape (batch_size, condition_dim)
        """
        raise NotImplementedError(
            "Return null embedding. "
            "Can be zeros, or a learned null_token buffer. "
            "Must match condition dimension."
        )

    def sample_with_guidance(
        self,
        batch_size: int,
        sample_shape: Tuple[int, ...],
        condition: np.ndarray,
        guidance_scale: Optional[float] = None,
        negative_condition: Optional[np.ndarray] = None,
        negative_weight: Optional[float] = None,
        return_trajectory: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """
        Sample from conditional distribution with classifier-free guidance.

        Two-prediction approach:

        For single positive condition (no negative):
            ε̃_θ = ε_θ(x_t, t, ∅) + w * (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))

        With negative condition:
            ε̃_θ = ε_θ(x_t, t, ∅)
                 + w_pos * (ε_θ(x_t, t, c_pos) - ε_θ(x_t, t, ∅))
                 - w_neg * (ε_θ(x_t, t, c_neg) - ε_θ(x_t, t, ∅))

        Args:
            batch_size: Number of samples to generate
            sample_shape: Shape of each sample
            condition: Positive condition(s) to guide toward
            guidance_scale: Guidance weight w (default: config.guidance_scale)
            negative_condition: Optional negative condition(s) to steer away from
            negative_weight: Weight for negative condition (default: config.negative_prompt_weight)
            return_trajectory: Return full denoising trajectory

        Returns:
            Generated samples, or (samples, trajectory) if return_trajectory=True
        """
        raise NotImplementedError(
            "Implement guided sampling with two predictions: unconditional and conditional. "
            "Loop through denoising steps: "
            "  - ε_uncond = self.denoiser(x_t, t, null_embedding) "
            "  - ε_cond = self.denoiser(x_t, t, condition) "
            "  - ε_guided = ε_uncond + w * (ε_cond - ε_uncond) "
            "  - x_{t-1} = sample from p_θ(x_{t-1}|x_t) using ε_guided. "
            "If negative_condition provided, also compute ε_neg and subtract w_neg * difference."
        )

    def sample(
        self,
        batch_size: int,
        sample_shape: Tuple[int, ...],
        condition: Optional[np.ndarray] = None,
        guidance_scale: Optional[float] = None,
        return_trajectory: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """
        Sample with optional guidance.

        If condition is None: Unconditional sampling (w=0)
        If condition is provided: Conditional sampling with guidance

        Args:
            batch_size: Number of samples
            sample_shape: Sample shape
            condition: Optional condition for guidance
            guidance_scale: Guidance weight
            return_trajectory: Return trajectory

        Returns:
            Generated samples
        """
        raise NotImplementedError(
            "If condition is None, call parent sample(). "
            "Otherwise call sample_with_guidance() with provided condition and guidance_scale."
        )

    def interpolate_in_condition_space(
        self,
        condition_1: np.ndarray,
        condition_2: np.ndarray,
        num_steps: int = 5,
        guidance_scale: float = 7.5,
        num_samples: int = 1
    ) -> list:
        """
        Interpolate between two conditions to explore smooth transitions.

        Generates samples for linearly interpolated conditions:
            c(α) = (1 - α) * c_1 + α * c_2, where α ∈ [0, 1]

        Useful for:
            - Understanding condition space
            - Creating smooth morphing videos
            - Testing model's interpolation behavior

        Args:
            condition_1: First condition
            condition_2: Second condition
            num_steps: Number of interpolation steps
            guidance_scale: Guidance weight during sampling
            num_samples: Number of samples per interpolation step

        Returns:
            List of samples for each interpolation step
        """
        raise NotImplementedError(
            "Create linear interpolation: c_interp = (1-α)*c_1 + α*c_2 for α in [0,1]. "
            "For each interpolated condition, sample using sample_with_guidance(). "
            "Return list of samples."
        )

    def steer_generation(
        self,
        condition_source: np.ndarray,
        condition_target: np.ndarray,
        x_t_start: np.ndarray,
        t_start: int,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Start from a partially denoised sample and steer toward target condition.

        Enables in-painting style operations: start with a noisy sample,
        then steer it toward a target condition.

        Algorithm:
        1. Start with x_t_start at timestep t_start (partially noised)
        2. For each t from t_start down to 0:
           - Compute guidance from source→target direction
           - Apply modified guidance: μ + strength * guidance_term
           - Sample x_{t-1}

        Args:
            condition_source: Source condition (where we started)
            condition_target: Target condition (where we want to go)
            x_t_start: Partially denoised sample at t_start
            t_start: Starting timestep
            strength: How strongly to steer toward target (0-1)

        Returns:
            Steered sample at t=0
        """
        raise NotImplementedError(
            "Implement guided steering. "
            "Compute difference in predictions: ε_target - ε_source. "
            "Apply as additional guidance term during reverse process."
        )


class MultiConditionDiffusion(ConditionalDiffusionModel):
    """
    Diffusion model supporting multiple condition types simultaneously.

    Can combine:
        - Class labels
        - Text prompts
        - Reference images
        - Continuous attributes
    """

    def __init__(
        self,
        denoiser: Module,
        condition_encoders: Dict[str, Module],
        config: ClassifierFreeGuidanceConfig = ClassifierFreeGuidanceConfig()
    ):
        """
        Initialize multi-condition model.

        Args:
            denoiser: Network that accepts multiple condition types
            condition_encoders: Dictionary mapping condition type to encoder
                                (e.g., {"text": text_encoder, "image": image_encoder})
            config: Configuration
        """
        super().__init__(denoiser, config)
        self.condition_encoders = condition_encoders

    def encode_condition(
        self,
        condition: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Encode multi-modal condition to unified representation.

        Args:
            condition: Dictionary with keys matching condition_encoders

        Returns:
            Concatenated encoded representation
        """
        raise NotImplementedError(
            "For each condition type in condition dict: "
            "  - Look up encoder in self.condition_encoders "
            "  - Encode the condition "
            "Concatenate all encoded conditions. "
            "Return concatenated embedding."
        )

    def sample_with_multi_condition(
        self,
        batch_size: int,
        sample_shape: Tuple[int, ...],
        conditions: Dict[str, np.ndarray],
        guidance_scales: Dict[str, float],
        return_trajectory: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """
        Sample with multiple condition types and independent guidance weights.

        Args:
            batch_size: Number of samples
            sample_shape: Sample shape
            conditions: Dict with "text", "image", etc.
            guidance_scales: Dict with guidance weight for each condition type
            return_trajectory: Return trajectory

        Returns:
            Generated samples
        """
        raise NotImplementedError(
            "Encode conditions using encode_condition(). "
            "During sampling, compute separate guidance terms for each condition type. "
            "Combine guidance terms with appropriate weights. "
            "Return samples."
        )


def create_guidance_scale_comparison(
    model: ConditionalDiffusionModel,
    condition: np.ndarray,
    guidance_scales: list = [0.0, 1.0, 3.0, 7.5, 15.0],
    num_samples: int = 1,
    sample_shape: Tuple[int, ...] = (3, 64, 64)
) -> Dict[float, np.ndarray]:
    """
    Generate samples at different guidance scales to understand the effect.

    Args:
        model: Trained conditional diffusion model
        condition: Condition to guide generation
        guidance_scales: List of w values to test
        num_samples: Number of samples per guidance scale
        sample_shape: Sample shape

    Returns:
        Dictionary mapping guidance_scale → samples
    """
    raise NotImplementedError(
        "For each guidance scale: call model.sample_with_guidance(). "
        "Collect results in dictionary. "
        "Return dict."
    )
