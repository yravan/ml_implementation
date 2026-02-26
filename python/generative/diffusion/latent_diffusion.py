"""
Latent Diffusion Models
=======================

Overview:
---------
Latent diffusion performs diffusion in a learned low-dimensional latent space
rather than in the high-dimensional pixel space. This dramatically reduces
computational cost while maintaining generation quality.

Key Idea: Instead of diffusing images x ∈ R^{H×W×C}, we:
    1. Encode images to latent codes: z = E(x)
    2. Diffuse in latent space: z_t = noise(z)
    3. Train denoiser in latent space: ε_θ(z_t, t, c)
    4. Decode generated latents: x̂ = D(z_0)

MOTIVATION:
===========

Pixel-space diffusion is EXPENSIVE:
    - Image size: 512×512×3 = 786,432 dimensions
    - Requires many denoising steps (1000)
    - Each step requires forward pass through large network

Latent diffusion solution:
    - Encode to ~4% of original size (e.g., 64×64×4 for 512×512 images)
    - Same denoising steps in compressed space
    - 16-50x speedup while maintaining quality!

ARCHITECTURE:
==============

Three main components:

1. ENCODER E: R^{H×W×C} → R^{h×w×c}
   Maps images to latent codes
   Typically: VAE encoder or similar
   Compression ratio: typical (H×W) → (H/8×W/8)

2. DIFFUSION MODEL: Operates on z_t
   Standard diffusion in latent space
   Can add classifier-free guidance with text conditions

3. DECODER D: R^{h×w×c} → R^{H×W×C}
   Reconstructs images from latents
   Typically: VAE decoder or similar

Full pipeline:
    Image x → Encoder E(x) = z → Noise z_t ~ q(z_t|z)
    → Diffusion model ε_θ(z_t, t) → Denoised z_0
    → Decoder D(z_0) → Reconstructed image x̂

LATENT SPACE PROPERTIES:
======================

The latent space should have desirable properties:

1. CONTINUITY: Nearby points in z-space → similar images
   Enables smooth interpolation and editing

2. EFFICIENCY: Low-dimensional but expressive
   Captures image features without redundancy

3. REGULARIZATION: Not just any autoencoder
   Uses VAE with KL divergence to prevent mode collapse
   Ensures smooth, well-behaved latent distribution

4. SEMANTIC ALIGNMENT: Latent dimensions correlate with semantic features
   Enables attribute-based editing

MATHEMATICAL FORMULATION:
=========================

Forward process in latent space:
    q(z_t|z_0) = N(z_t; √ᾱ_t * z_0, (1 - ᾱ_t) * I)

Same as pixel-space diffusion, just applied to z instead of x!

Reverse process:
    p_θ(z_{t-1}|z_t) = N(z_{t-1}; μ_θ(z_t, t), σ_t² * I)

Where μ_θ predicts noise: μ_θ = (1/√α_t) * [z_t - (β_t/√(1-ᾱ_t)) * ε_θ(z_t, t)]

VARIATIONAL LOWER BOUND:
========================

For latent diffusion, the ELBO is:

    log p_θ(x) ≥ E_q [log p_θ(D(z_0)|z_0)] - KL(q(z|x) || p(z))

Where:
    - First term: Reconstruction quality in image space
    - Second term: KL divergence between encoder and prior

In practice:
    log p_θ(x|D(z_0)) ≈ -||x - D(ẑ_0)||² (MSE loss)

Training objective:
    L = E_x,t,c,ε [||ε - ε_θ(E(x)_t, t, c)||²]

Where x is image, c is condition (optional).

ADVANTAGES OF LATENT DIFFUSION:
================================
1. 10-50x faster sampling than pixel-space
2. Enables high-resolution generation (requires latent compression)
3. Efficient training on limited GPU memory
4. Works with ANY autoencoder (not just VAE)
5. Compatible with all diffusion improvements (DDIM, guidance, etc.)

DISADVANTAGES:
===============
1. Quality limited by autoencoder reconstruction quality
2. Artifacts can propagate from encoder/decoder
3. Requires pre-training autoencoder
4. Less interpretable than pixel-space (hidden in latent space)

STABLE DIFFUSION ARCHITECTURE:
=============================

Landmark application: Stable Diffusion uses:
    - Encoder: VAE encoder, compression ratio 8×
    - Diffusion: U-Net in latent space with ~1B parameters
    - Decoder: VAE decoder
    - Conditions: Text embeddings from CLIP
    - Guidance: Classifier-free guidance (7.5 scale)

Enables:
    - Text-to-image generation
    - Fast inference on consumer GPUs
    - Open-source deployment

VARIATIONAL AUTOENCODER (VAE):
==============================

For learned latent space, we use VAE:

Encoder: q_φ(z|x) = N(μ_φ(x), σ_φ²(x))
Decoder: p_θ(x|z) = N(D_θ(z), σ_decode² * I)

Training objective:
    L_VAE = -E_q [log p_θ(x|z)] + KL(q_φ(z|x) || p(z))
          = Reconstruction_loss + KL_reg

Where:
    - Reconstruction: How well decoder reconstructs x from z
    - KL regularization: Ensures q_φ(z|x) ≈ N(0, I)

Scaling parameter β ∈ [0, 1]:
    L = Reconstruction + β * KL

    β = 0: Pure reconstruction (ignore KL, mode collapse)
    β = 1: Standard VAE (balance reconstruction and regularization)
    β > 1: Aggressive regularization (less detail, more diversity)

LATENT SPACE INTERPOLATION AND EDITING:
======================================

One advantage of latent diffusion:

1. INTERPOLATION:
   For z_1 = E(x_1) and z_2 = E(x_2):
   z_interp(α) = (1-α) * z_1 + α * z_2
   x_interp = D(z_interp) → smooth morphing

2. LINEAR ATTRIBUTE EDITING:
   If latent dimension i correlates with attribute a:
   z_edited = z + λ * ∇_z a(z)  (push in attribute direction)
   x_edited = D(z_edited)

3. DIFFUSION-BASED EDITING:
   Encode, diffuse to time t, apply edits, denoise back to pixel space

CONDITIONAL LATENT DIFFUSION:
=============================

Can add conditions to latent diffusion:
    ε_θ(z_t, t, c) with condition c (e.g., text prompt)

Text conditioning (Stable Diffusion approach):
    1. Encode text to embeddings: c = text_encoder(prompt)
    2. Pass embedding to diffusion network
    3. Use classifier-free guidance during sampling

Image conditioning (for inpainting, control):
    1. Encode reference image: c_img = E(x_ref)
    2. Pass to diffusion network or use as constraint
    3. Constrain generation to match in masked regions

PRACTICAL TRAINING:
===================

1. Pre-train VAE on image reconstruction
   - Standard VAE loss
   - Ensure encoder/decoder quality

2. Train latent diffusion model
   - Use pre-trained (frozen) encoder/decoder
   - Train diffusion in latent space
   - Can add conditions (text, images, etc.)

3. Fine-tuning options:
   - Keep encoder/decoder frozen: Faster, stable
   - Fine-tune encoder/decoder: Better quality, slower
   - Joint training: From scratch, most flexible

REFERENCES:
-----------
[1] "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
    https://arxiv.org/abs/2112.10752
    Stable Diffusion paper - canonical latent diffusion work

[2] "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014)
    https://arxiv.org/abs/1312.6114
    VAE foundation for learning latent spaces

[3] "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    https://arxiv.org/abs/2006.11239
    Original DDPM, applies directly to latent space
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass

from python.nn_core import Module, Parameter
from ddpm import DDPM, DDPMConfig
from classifier_free_guidance import ConditionalDiffusionModel, ClassifierFreeGuidanceConfig


@dataclass
class LatentDiffusionConfig(ClassifierFreeGuidanceConfig):
    """Configuration for latent diffusion models."""

    # Latent space
    latent_channels: int = 4  # Number of channels in latent space
    latent_height: int = 64  # Latent space height (for 512x512 images: 64)
    latent_width: int = 64   # Latent space width
    latent_scale_factor: int = 8  # Compression ratio (image_size / latent_size)

    # VAE configuration
    vae_scale_factor: float = 0.18215  # Scaling factor for latent distribution (Stable Diffusion default)

    # Conditioning
    text_encoder_dim: int = 768  # CLIP text encoder dimension


class VariationalAutoencoder(Module):
    """
    Variational Autoencoder for learning a latent space.

    Encodes high-resolution images to low-dimensional latent codes
    and decodes them back with minimal loss.

    Architecture:
        Encoder: Series of conv layers with downsampling → latent distribution
        Decoder: Series of deconv layers with upsampling → reconstructed image

    Loss:
        L_VAE = Reconstruction + β * KL_divergence
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_dims: list = [64, 128, 256, 512],
        beta: float = 1.0
    ):
        """
        Initialize VAE.

        Args:
            in_channels: Number of input channels (3 for RGB)
            latent_channels: Number of latent channels
            hidden_dims: List of hidden dimensions for encoder/decoder layers
            beta: Weight for KL divergence term (0=reconstruction only, 1=balanced)
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        self.beta = beta

        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Latent distribution parameters
        self.fc_mu = self._build_mean_layer()
        self.fc_var = self._build_variance_layer()

    def _build_encoder(self) -> Module:
        """
        Build encoder network.

        Should progressively downsample spatial dimensions
        while increasing channels.

        Returns:
            Sequential encoder module
        """
        raise NotImplementedError(
            "Build encoder with downsampling convolutions. "
            "Each block: Conv2d → BatchNorm → ReLU → optional MaxPool. "
            "Progressively increase channels, decrease spatial size."
        )

    def _build_decoder(self) -> Module:
        """
        Build decoder network.

        Should progressively upsample spatial dimensions
        while decreasing channels.

        Returns:
            Sequential decoder module
        """
        raise NotImplementedError(
            "Build decoder with upsampling convolutions. "
            "Each block: Conv2d → BatchNorm → ReLU → Upsample. "
            "Progressively decrease channels, increase spatial size."
        )

    def _build_mean_layer(self) -> Module:
        """Build final layer to predict mean μ of latent distribution."""
        raise NotImplementedError(
            "Create linear/conv layer to map encoder output to mean. "
            "Output dimension: (batch_size, latent_channels, latent_h, latent_w)"
        )

    def _build_variance_layer(self) -> Module:
        """Build final layer to predict log-variance log(σ²) of latent distribution."""
        raise NotImplementedError(
            "Create linear/conv layer to map encoder output to log variance. "
            "Output dimension: (batch_size, latent_channels, latent_h, latent_w)"
        )

    def encode(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode image to latent distribution.

        Args:
            x: Image tensor, shape (batch_size, channels, height, width)

        Returns:
            (latent_dist_mean, latent_dist_log_var, latent_samples)
            where latent_samples ~ N(mean, exp(log_var))
        """
        raise NotImplementedError(
            "Pass x through encoder to get features. "
            "Compute mean = fc_mu(features) and log_var = fc_var(features). "
            "Sample z = mean + exp(0.5 * log_var) * eps, where eps ~ N(0,I). "
            "Return (mean, log_var, z)."
        )

    def decode(
        self,
        z: np.ndarray
    ) -> np.ndarray:
        """
        Decode latent code to image.

        Args:
            z: Latent code, shape (batch_size, latent_channels, latent_h, latent_w)

        Returns:
            Reconstructed image, shape (batch_size, in_channels, height, width)
        """
        raise NotImplementedError(
            "Pass z through decoder to reconstruct image. "
            "Return reconstructed image."
        )

    def forward(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass: encode and decode.

        Args:
            x: Input image

        Returns:
            (reconstructed_image, mean, log_var)
        """
        raise NotImplementedError(
            "Call encode(x) to get mean, log_var, z. "
            "Call decode(z) to get reconstruction. "
            "Return (reconstruction, mean, log_var)."
        )

    def compute_loss(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        mean: np.ndarray,
        log_var: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute VAE loss.

        Loss = Reconstruction + β * KL divergence

        Reconstruction: -E_q [log p(x|z)]
            Typically MSE: ||x - x_recon||²

        KL divergence: KL(q(z|x) || p(z))
            For q = N(μ, σ²) and p = N(0, I):
            KL = 0.5 * Σ[μ² + σ² - log(σ²) - 1]
               = 0.5 * Σ[μ² + exp(log_var) - log_var - 1]

        Args:
            x: Original image
            x_recon: Reconstructed image
            mean: Mean of q(z|x)
            log_var: Log variance of q(z|x)

        Returns:
            (total_loss, kl_loss)
        """
        raise NotImplementedError(
            "Compute MSE reconstruction loss: ||x - x_recon||². "
            "Compute KL divergence: 0.5 * sum[μ² + exp(log_var) - log_var - 1]. "
            "Total loss = reconstruction + beta * kl_loss. "
            "Return (total_loss, kl_loss) for monitoring."
        )

    def sample(
        self,
        num_samples: int,
        output_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Generate samples by sampling from prior p(z) = N(0, I).

        Args:
            num_samples: Number of samples to generate
            output_shape: Output image shape (channels, height, width)

        Returns:
            Generated images
        """
        raise NotImplementedError(
            "Sample z ~ N(0, I) with shape (num_samples, latent_channels, latent_h, latent_w). "
            "Decode: x = decode(z). "
            "Return x."
        )


class LatentDiffusionModel(ConditionalDiffusionModel):
    """
    Latent Diffusion Model.

    Combines:
    1. VAE encoder/decoder for latent space
    2. Diffusion model in latent space
    3. Optional conditional generation (text, etc.)

    Efficient high-resolution image generation by diffusing
    in compressed latent space rather than pixel space.
    """

    def __init__(
        self,
        denoiser: Module,
        vae: VariationalAutoencoder,
        config: LatentDiffusionConfig = LatentDiffusionConfig()
    ):
        """
        Initialize latent diffusion model.

        Args:
            denoiser: Network for diffusion in latent space
            vae: Pre-trained VAE for encoding/decoding
            config: LatentDiffusionConfig
        """
        super().__init__(denoiser, config)
        self.vae = vae
        self.config = config
        self.vae_scale_factor = config.vae_scale_factor

    def encode_image_to_latent(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Encode image to latent space using VAE encoder.

        Args:
            x: Image tensor, shape (batch_size, channels, height, width)

        Returns:
            Latent code z, shape (batch_size, latent_channels, latent_h, latent_w)
        """
        raise NotImplementedError(
            "Call vae.encode(x) to get (mean, log_var, z). "
            "Return z (or optionally just mean for deterministic encoding). "
            "Apply scaling: z = z * vae_scale_factor."
        )

    def decode_latent_to_image(
        self,
        z: np.ndarray
    ) -> np.ndarray:
        """
        Decode latent code to image using VAE decoder.

        Args:
            z: Latent code, shape (batch_size, latent_channels, latent_h, latent_w)

        Returns:
            Reconstructed image
        """
        raise NotImplementedError(
            "Unscale latent: z = z / vae_scale_factor. "
            "Call vae.decode(z). "
            "Return decoded image."
        )

    def forward_diffusion_in_latent_space(
        self,
        x: np.ndarray,
        t: np.ndarray,
        noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode image to latent and apply diffusion.

        Args:
            x: Image to corrupt
            t: Timestep
            noise: Optional pre-sampled noise

        Returns:
            (z_t, noise): Noisy latent and noise
        """
        raise NotImplementedError(
            "Encode image: z = encode_image_to_latent(x). "
            "Apply diffusion: z_t = add_noise_to_sample(z, t, noise). "
            "Return (z_t, noise)."
        )

    def training_step(
        self,
        x: np.ndarray,
        condition: Optional[np.ndarray] = None,
        optimizer: Optional[object] = None
    ) -> np.ndarray:
        """
        Training step in latent space.

        Args:
            x: Batch of images
            condition: Optional condition (e.g., text embedding)
            optimizer: Optimizer instance

        Returns:
            Loss for backprop
        """
        raise NotImplementedError(
            "Encode images to latent space: z = encode_image_to_latent(x). "
            "Sample timesteps and noise. "
            "Apply diffusion to latent: z_t. "
            "Predict noise: ε_θ = denoiser(z_t, t, condition). "
            "Compute loss: ||noise - ε_θ||². "
            "Handle unconditional dropout for classifier-free guidance. "
            "Return loss."
        )

    def sample(
        self,
        batch_size: int,
        height: int,
        width: int,
        condition: Optional[np.ndarray] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        return_trajectory: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """
        Generate images by diffusing in latent space.

        Algorithm:
        1. Sample z_T ~ N(0, I) in latent space
        2. For t = T, T-1, ..., 1:
           - Predict noise in latent space: ε_θ(z_t, t, condition)
           - Apply guidance if provided
           - Compute z_{t-1}
        3. Decode final latent to image: x = decode(z_0)

        Args:
            batch_size: Number of samples
            height: Output image height (must be multiple of latent_scale_factor)
            width: Output image width
            condition: Optional condition for guidance (e.g., text embedding)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            return_trajectory: Return latent trajectory

        Returns:
            Generated images, shape (batch_size, channels, height, width)
        """
        raise NotImplementedError(
            "Compute latent dimensions from height/width and latent_scale_factor. "
            "Sample z_T ~ N(0, latent_shape). "
            "Denoise in latent space using sample_with_guidance(). "
            "Decode final z_0 to image space. "
            "Return images (clipped to valid range if needed)."
        )

    def inpaint(
        self,
        x_original: np.ndarray,
        mask: np.ndarray,
        condition: Optional[np.ndarray] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Inpainting: regenerate masked regions while preserving unmasked parts.

        Algorithm:
        1. Encode original image: z_0 = encode(x_original)
        2. Encode mask to latent space
        3. Forward diffuse z_0 to timestep t_start: z_t ~ q(z_t|z_0)
        4. Denoising loop from t_start down to 0:
           - Predict noise: ε_θ(z_t, t, condition)
           - Compute z_{t-1}
           - Inpaint: blend z_{t-1} with original in masked regions
        5. Decode: x = decode(z_0)

        Args:
            x_original: Original image to inpaint
            mask: Binary mask (1=keep, 0=regenerate)
            condition: Optional condition for guidance
            guidance_scale: Guidance scale
            num_inference_steps: Number of steps
            strength: How much to regenerate (0=keep original, 1=fully regenerate)

        Returns:
            Inpainted image
        """
        raise NotImplementedError(
            "Encode original image and mask. "
            "Compute t_start from strength. "
            "Forward diffuse original to t_start. "
            "Denoise from t_start with inpainting constraint: "
            "  z_{t-1} = (1 - mask) * z_original + mask * z_denoised. "
            "Decode result. "
            "Return inpainted image."
        )

    def edit_image(
        self,
        x_original: np.ndarray,
        prompt_edit: Optional[str] = None,
        condition_original: Optional[np.ndarray] = None,
        condition_edit: Optional[np.ndarray] = None,
        num_inference_steps: int = 50,
        start_step: int = 0,
        guidance_scale: float = 7.5
    ) -> np.ndarray:
        """
        Edit an image by changing the condition/prompt.

        Procedure (Prompt-to-Prompt editing):
        1. Encode original image: z_0 = encode(x)
        2. Forward diffuse to intermediate timestep t_start: z_t
        3. Denoise from t_start using new condition/prompt
        4. Blend with original to maintain structure

        Args:
            x_original: Original image to edit
            prompt_edit: New prompt for editing
            condition_original: Embedding of original condition
            condition_edit: Embedding of new condition
            num_inference_steps: Total steps
            start_step: Which step to start editing from (0=full replacement, high=subtle)
            guidance_scale: Guidance scale for new condition

        Returns:
            Edited image
        """
        raise NotImplementedError(
            "Implement prompt-to-prompt or similar editing. "
            "Encode original, diffuse to intermediate step, "
            "then denoise with new condition from that point. "
            "Return edited image."
        )

    def interpolate_images(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        num_steps: int = 5,
        condition: Optional[np.ndarray] = None,
        guidance_scale: float = 7.5
    ) -> list:
        """
        Interpolate between two images smoothly.

        Algorithm:
        1. Encode both images: z1 = encode(x1), z2 = encode(x2)
        2. Create interpolation: z_α = (1-α)*z1 + α*z2 for α ∈ [0,1]
        3. Denoise each interpolated latent
        4. Decode to images

        Args:
            x1: First image
            x2: Second image
            num_steps: Number of interpolation steps
            condition: Optional condition to guide interpolation
            guidance_scale: Guidance scale

        Returns:
            List of interpolated images
        """
        raise NotImplementedError(
            "Encode both images. "
            "Linearly interpolate in latent space. "
            "For each interpolated latent, denoise using sample_with_guidance(). "
            "Decode to images. "
            "Return list of images."
        )


class TextToImageDiffusion(LatentDiffusionModel):
    """
    Text-to-Image diffusion model (like Stable Diffusion).

    Generates images from text prompts using:
    - Text encoder (e.g., CLIP) to encode prompts
    - Latent diffusion model
    - Classifier-free guidance for strong prompt adherence
    """

    def __init__(
        self,
        denoiser: Module,
        vae: VariationalAutoencoder,
        text_encoder: Module,
        config: LatentDiffusionConfig = LatentDiffusionConfig()
    ):
        """
        Initialize text-to-image model.

        Args:
            denoiser: Diffusion network
            vae: VAE for latent encoding
            text_encoder: Text encoder (e.g., CLIP)
            config: Configuration
        """
        super().__init__(denoiser, vae, config)
        self.text_encoder = text_encoder

    def encode_prompt(
        self,
        prompt: Union[str, list]
    ) -> np.ndarray:
        """
        Encode text prompt(s) to embedding vector(s).

        Args:
            prompt: Single prompt string or list of prompts

        Returns:
            Embedding tensor, shape (batch_size, embedding_dim) or (1, embedding_dim)
        """
        raise NotImplementedError(
            "Tokenize prompt(s) using text encoder's tokenizer. "
            "Pass tokens through text encoder. "
            "Extract final embedding (typically last token or pooled). "
            "Return embedding tensor."
        )

    def generate_from_text(
        self,
        prompt: Union[str, list],
        negative_prompt: Union[str, list, None] = None,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        return_latents: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate images from text prompt(s).

        Args:
            prompt: Text prompt(s) describing desired image(s)
            negative_prompt: Text prompt(s) describing what NOT to generate
            height: Output image height (multiple of 8)
            width: Output image width (multiple of 8)
            guidance_scale: Classifier-free guidance strength (7.5 typical)
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            return_latents: If True, also return latent codes before decoding

        Returns:
            Generated images, shape (batch_size, 3, height, width)
            Or (images, latents) if return_latents=True
        """
        raise NotImplementedError(
            "Encode positive prompt(s). "
            "Encode negative prompt(s) if provided. "
            "Call sample() with guidance using both prompts. "
            "Return generated images (and optionally latents)."
        )

    def run_pipeline(
        self,
        prompt: str,
        **kwargs
    ) -> np.ndarray:
        """
        Convenience method to generate image from prompt.

        Args:
            prompt: Text description
            **kwargs: Additional arguments for generate_from_text()

        Returns:
            Generated image
        """
        raise NotImplementedError(
            "Call generate_from_text(prompt, **kwargs). "
            "Return first image in batch."
        )
