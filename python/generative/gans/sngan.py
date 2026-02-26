"""
Spectral Normalization GAN (SN-GAN)

Original paper: "Spectral Normalization for Generative Adversarial Network"
(Miyato et al., 2018)
https://arxiv.org/abs/1802.05957

Spectral Normalization (SN) provides an alternative to gradient penalty for
enforcing the Lipschitz constraint on the discriminator. Instead of penalizing
gradients directly, SN normalizes weight matrices by their largest singular
value (spectral radius), which indirectly controls the Lipschitz constant.

THEORY:
--------
Key Insight:
  The Lipschitz constant of a feedforward neural network can be bounded by
  the product of Lipschitz constants of individual layers:

    L(D) ≤ ∏_i L(D_i)

  where D_i is layer i (convolution, linear, activation).

  For linear layers (weights W):
    L(W) = σ_max(W)  (largest singular value = spectral radius)

  For nonlinearities:
    L(ReLU) = 1
    L(LeakyReLU(α)) = max(1, α)

Spectral Normalization:
  Normalize weight matrices by their largest singular value:

    W_SN = W / σ_max(W)

  This makes σ_max(W_SN) = 1, so L(W_SN) = 1.

  By normalizing all weight layers, we get:
    L(D) ≤ ∏_i 1 = 1  (1-Lipschitz constraint)

ADVANTAGES OVER GRADIENT PENALTY:
  1. COMPUTATIONAL EFFICIENCY: No extra forward/backward pass needed
  2. NO PENALTY TERM: Constraint built into weights, not loss
  3. DETERMINISTIC: No randomness in spectral computation
  4. STABLE: Works well with various optimizers and LRs
  5. PRACTICAL: Easy to implement (simple matrix normalization)
  6. COMPLEMENTARY: Can combine with other stabilization techniques

ADVANTAGES OVER WEIGHT CLIPPING:
  1. NO CAPACITY LIMITATION: Full weight magnitudes allowed
  2. SMOOTH CONSTRAINT: Differentiable operation (unlike clipping)
  3. PRINCIPLED: Theoretically justified Lipschitz bound
  4. EFFECTIVE: Gradient flow remains strong

CHALLENGES:
  1. Computing spectral radius is expensive (requires SVD or power iteration)
  2. Need efficient approximation for large matrices
  3. Requires careful implementation for computational efficiency
  4. May require tuning of iteration count in power method

SPECTRAL RADIUS COMPUTATION:
  Computing exact σ_max(W) via SVD is expensive: O(n²m) for n×m matrix.

  Power Iteration Method (efficient approximation):
    Iteratively approximate σ_max without full SVD:

    1. Start with random vector u
    2. For k iterations:
       v = W^T u / ||W^T u||     (right singular vector)
       u = W v / ||W v||          (left singular vector)
    3. σ ≈ u^T W v ≈ σ_max(W)

    Typically: k=1 update per training step (very efficient)
    Convergence: σ ≈ σ_max within a few iterations

IMPLEMENTATION:
  1. Reshape weight matrix: (out_channels, in_channels * kernel_h * kernel_w)
  2. Maintain u vector (same dimension as out_channels)
  3. Each forward pass:
     a. Compute v = W^T u / ||W^T u||
     b. Compute u_new = W v / ||W v||
     c. Update u ← u_new
     d. Normalize: W_SN = W / (u^T W v)

PROPERTIES:
  1. Makes discriminator 1-Lipschitz without explicit penalty
  2. Stabilizes training significantly
  3. Works well with standard optimizers (SGD, Adam)
  4. Can be applied to any linear layer (Conv2d, Linear)
  5. Minimal computational overhead with power iteration

TRAINING DYNAMICS:
  - More stable than Vanilla GAN, DCGAN
  - Better than weight clipping (smoother gradients)
  - Comparable to gradient penalty but more efficient
  - Works especially well on high-resolution synthesis

ARCHITECTURAL CONSIDERATIONS:
  1. Apply to discriminator layers (essential for Lipschitz bound)
  2. Optional for generator (can apply but not required)
  3. Typically applied to Conv2d and Linear layers
  4. Skip final classification layer (or apply)
  5. Combine with other techniques (instance normalization, attention)

MATHEMATICAL FORMULATION:
  Weight normalization:
    W_SN = W / σ_max(W)

  Power iteration:
    σ_max ≈ max_u ||W u|| = max_u (u^T W^T W u)^(1/2)

  Efficient computation:
    u_{t+1} = W v_t / ||W v_t||
    v_{t+1} = W^T u_{t+1} / ||W^T u_{t+1}||
    σ_t = u_t^T W v_t

  Where t is the iteration number (typically k=1 per step)

PAPER CONTRIBUTIONS:
  1. Proposed spectral normalization for weight matrices
  2. Showed connection to Lipschitz constraint
  3. Provided efficient power iteration approximation
  4. Demonstrated superior training stability
  5. Achieved high-quality generation on ImageNet 128×128
  6. Showed compatibility with existing GAN architectures

RELATED METHODS:
  - Weight Clipping (WGAN): Crude constraint enforcement
  - Gradient Penalty (WGAN-GP): Penalty-based approach
  - Self-Attention GAN (SAGAN): Combines SN with attention
  - Progressive GAN: Uses SN throughout training
  - StyleGAN: Builds on SN foundations
"""

import numpy as np
from typing import Tuple, Optional, List
from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.conv.conv2d import Conv2d
from python.nn_core.normalization.batchnorm import BatchNorm2d
from python.nn_core.activations.relu import ReLU, LeakyReLU


class SpectralNorm(Module):
    """
    Spectral Normalization wrapper for weight matrices.

    Normalizes weight matrices by their largest singular value (spectral radius),
    enforcing a 1-Lipschitz constraint on the layer.

    Uses power iteration for efficient approximation of the largest singular value.
    """

    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        """
        Args:
            module: Module to apply spectral normalization to
            name: Name of weight parameter to normalize (typically 'weight')
            n_power_iterations: Number of power iterations per forward pass (typically 1)
        """
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations

        # TODO: Implement spectral normalization wrapper
        # Hint: Architecture:
        #   1. Get weight matrix W from module
        #   2. Reshape W to 2D: (out_features, in_features * kernel_h * kernel_w)
        #   3. Initialize random u vector: shape (out_features,)
        #   4. Register as buffer: register_buffer('u', u)
        #   5. In forward: Apply power iteration, normalize weights
        #
        # Implementation pattern:
        #   def forward(self, *args, **kwargs):
        #       self._update_u()  # Update u via power iteration
        #       self._normalize_weight()  # Normalize by sigma_max
        #       return self.module(*args, **kwargs)
        raise NotImplementedError(
            "Spectral normalization wrapper not implemented. "
            "Hint: Reshape weight to 2D, maintain u vector, implement power iteration."
        )

    def _update_u(self):
        """
        Update u vector using power iteration.

        Approximates the largest singular vector of the weight matrix.

        Algorithm (power iteration):
        1. v = W^T u / ||W^T u||          (right singular vector)
        2. u_new = W v / ||W v||          (left singular vector)
        3. Update u ← u_new

        This converges to the dominant singular value/vector pair.
        """
        raise NotImplementedError(
            "Power iteration update not implemented. "
            "Hint: Compute v = normalize(W.T @ u), then u = normalize(W @ v)."
        )

    def _normalize_weight(self):
        """
        Normalize weight matrix by its largest singular value.

        Divides weight by σ_max so that σ_max(W_normalized) = 1,
        making the layer 1-Lipschitz continuous.
        """
        raise NotImplementedError(
            "Weight normalization not implemented. "
            "Hint: Compute sigma_max = u.T @ W @ v, then divide W by sigma_max."
        )

    def forward(self, *args, **kwargs):
        """Forward pass with spectral normalization."""
        raise NotImplementedError("Forward pass not implemented")


def spectral_norm(module: nn.Module, n_power_iterations: int = 1) -> nn.Module:
    """
    Helper function to apply spectral normalization to a module.

    Usage:
        layer = spectral_norm(nn.Conv2d(3, 64, 4, 2, 1))

    Args:
        module: Module to apply SN to
        n_power_iterations: Number of power iterations per forward

    Returns:
        SpectralNorm wrapped module
    """
    return SpectralNorm(module, n_power_iterations=n_power_iterations)


class SNGenerator(Module):
    """
    Generator for SN-GAN. Standard architecture (optional SN on generator).

    Usually SN is applied to discriminator. Generator may use SN but it's not required.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        feature_maps: int = 64,
        img_channels: int = 3,
        img_size: int = 64,
    ):
        """
        Args:
            latent_dim: Latent dimension
            feature_maps: Base feature maps
            img_channels: Output channels
            img_size: Output image size
        """
        super(SNGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.img_size = img_size

        # TODO: Implement generator
        # Hint: Standard DCGAN-style architecture:
        #   1. Linear: latent_dim -> feature_maps*8*4*4
        #   2. Reshape to spatial
        #   3. ConvTranspose2d blocks (optional: wrap with spectral_norm)
        #   4. Output: Tanh activation
        raise NotImplementedError(
            "Generator architecture not implemented. "
            "Hint: Use standard DCGAN generator."
        )

    def forward(self, z):
        """
        Args:
            z: Noise vector (batch_size, latent_dim)

        Returns:
            Generated image (batch_size, img_channels, img_size, img_size)
        """
        raise NotImplementedError("Forward pass not implemented")


class SNDiscriminator(Module):
    """
    Discriminator with Spectral Normalization applied to all weight matrices.

    Spectral normalization is applied to Conv2d and Linear layers.
    This ensures the discriminator is 1-Lipschitz continuous, enabling
    more stable training than standard GANs.

    Architecture for 64x64 RGB:
        Input: (batch, 3, 64, 64)
        -> Conv2d(SN): 3 -> 64, stride=2
        -> LeakyReLU(0.2)
        -> Conv2d(SN): 64 -> 128, stride=2
        -> LeakyReLU(0.2)
        -> Conv2d(SN): 128 -> 256, stride=2
        -> LeakyReLU(0.2)
        -> Conv2d(SN): 256 -> 512, stride=2
        -> LeakyReLU(0.2)
        -> Flatten
        -> Linear(SN): 512*4*4 -> 1 (output logit)

    Key Point: SN applied to all layers, including output Linear layer.
    """

    def __init__(
        self,
        feature_maps: int = 64,
        img_channels: int = 3,
        img_size: int = 64,
    ):
        """
        Args:
            feature_maps: Base feature maps
            img_channels: Input channels
            img_size: Input image size
        """
        super(SNDiscriminator, self).__init__()
        self.feature_maps = feature_maps
        self.img_size = img_size

        # TODO: Implement discriminator with spectral normalization
        # Hint: Architecture pattern:
        #   1. Conv2d(SN): img_channels -> feature_maps, kernel=4, stride=2, padding=1
        #      + LeakyReLU(0.2)
        #   2. Conv2d(SN): feature_maps -> 2*feature_maps, kernel=4, stride=2, padding=1
        #      + LeakyReLU(0.2)
        #   3. Repeat for more layers (typically 4-5 total conv layers)
        #   4. Flatten
        #   5. Linear(SN): flattened -> 1 (output logit, no activation)
        #
        # Spectral norm application:
        #   conv_layer = spectral_norm(nn.Conv2d(...))
        #   linear_layer = spectral_norm(nn.Linear(...))
        raise NotImplementedError(
            "Discriminator architecture not implemented. "
            "Hint: Use spectral_norm wrapper on Conv2d and Linear layers."
        )

    def forward(self, x):
        """
        Args:
            x: Image batch (batch_size, img_channels, img_size, img_size)

        Returns:
            Classification logit (batch_size, 1)
        """
        raise NotImplementedError("Forward pass not implemented")


class SNGAN:
    """
    Spectral Normalization GAN trainer.

    Uses spectral normalization instead of weight clipping or gradient penalty
    to enforce the Lipschitz constraint on the discriminator.

    Benefits:
    1. More efficient than gradient penalty (no extra forward/backward)
    2. Better gradients than weight clipping
    3. Stable convergence
    4. Works with standard optimizers
    """

    def __init__(
        self,
        generator: SNGenerator,
        discriminator: SNDiscriminator,
        latent_dim: int = 100,
        device: str = "cpu",
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        beta1: float = 0.0,
        beta2: float = 0.9,
    ):
        """
        Args:
            generator: Generator network
            discriminator: Discriminator with spectral normalization
            latent_dim: Latent dimension
            device: Device to train on
            lr_g: Generator learning rate
            lr_d: Discriminator learning rate
            beta1: Adam beta1 (default 0.0 for SN-GAN, not typical 0.5)
            beta2: Adam beta2 (default 0.9, not typical 0.999)
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        # TODO: Initialize optimizers
        # Note: Different betas than WGAN-GP
        # Default: beta1=0.0, beta2=0.9 (from paper)
        raise NotImplementedError(
            "Optimizer initialization not implemented. "
            "Hint: Create optimizers with specified betas."
        )

    def sample_noise(self, batch_size: int):
        """Sample latent vectors."""
        return np.random.randn(batch_size, self.latent_dim).astype(np.float32)

    def train_step(
        self,
        real_data,
        n_critic: int = 1,
    ) -> Tuple[float, float]:
        """
        Single SN-GAN training iteration.

        Args:
            real_data: Batch of real images
            n_critic: Number of discriminator updates per generator update

        Returns:
            (d_loss, g_loss)

        SN-GAN TRAINING:
        ================
        Similar to standard GAN but with spectral normalization on discriminator.

        Discriminator:
          1. Forward real data: D(x_real)
          2. Forward fake data: D(G(z))
          3. Compute loss: BCE(D(x), labels)
          4. Spectral norm automatically applied (no explicit penalty)

        Generator:
          1. Generate fake: G(z)
          2. Forward through D: D(G(z))
          3. Compute loss: BCE(D(fake), real_labels)

        The spectral normalization built into weights ensures:
          - Each layer is 1-Lipschitz
          - Overall discriminator is 1-Lipschitz
          - No capacity limitations (unlike weight clipping)
          - No computational overhead (power iteration built into forward pass)
        """
        batch_size = real_data.shape[0]

        real_label = 0.9  # Label smoothing
        fake_label = 0.0

        # TODO: Implement discriminator training (n_critic iterations)
        # Similar to Vanilla GAN but discriminator has spectral norm built in
        raise NotImplementedError(
            "Discriminator training step not implemented. "
            "Hint: Forward pass through SN-regularized discriminator automatically."
        )

        # TODO: Implement generator training
        raise NotImplementedError(
            "Generator training step not implemented."
        )

    def generate_samples(self, num_samples: int = 16):
        """Generate samples."""
        self.generator.eval()
        z = self.sample_noise(num_samples)
        fake_data = self.generator(z)
        self.generator.train()
        return fake_data

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        raise NotImplementedError("Checkpoint saving not implemented.")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        raise NotImplementedError("Checkpoint loading not implemented.")


def train_sngan(
    generator: SNGenerator,
    discriminator: SNDiscriminator,
    train_loader,
    num_epochs: int = 50,
    latent_dim: int = 100,
    n_critic: int = 1,
) -> Tuple[list, list]:
    """
    Training loop for SN-GAN.

    Args:
        generator: Generator network
        discriminator: Discriminator with spectral normalization
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device to train on
        latent_dim: Latent dimension
        n_critic: Number of discriminator updates per generator update

    Returns:
        (g_losses, d_losses): Loss lists

    SN-GAN ADVANTAGES:
    ==================
    1. EFFICIENT: No gradient penalty computation overhead
    2. STABLE: Spectral normalization built into forward pass
    3. FLEXIBLE: Works with different optimizers and LRs
    4. SCALABLE: Efficiently scales to high-resolution generation
    5. PRACTICAL: Easy to implement and integrate

    CONVERGENCE:
    ============
    - Typically converges in 20-30 epochs (CIFAR-10)
    - Loss curves are smooth and interpretable
    - Better training stability than Vanilla GAN/DCGAN
    - Comparable to WGAN-GP but more efficient
    """
    gan = SNGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=latent_dim,
    )

    g_losses = []
    d_losses = []

    # TODO: Implement training loop
    raise NotImplementedError(
        "Training loop not implemented. "
        "Hint: Standard epoch/batch loop."
    )

    return g_losses, d_losses


# MATHEMATICAL FOUNDATION:
# ========================
#
# SPECTRAL RADIUS (Largest Singular Value):
# ---
# σ_max(W) = max ||W u|| where ||u|| = 1
#
# For matrix W, this is the largest singular value from SVD: W = U Σ V^T
#
# LIPSCHITZ CONSTANT OF LINEAR LAYER:
# --
# For a linear map W: L(W) = σ_max(W)
#
# If σ_max(W) ≤ 1, then W is 1-Lipschitz:
#   ||W u1 - W u2|| ≤ ||u1 - u2||
#
# COMPOSITION OF LIPSCHITZ FUNCTIONS:
# ---
# If f is L_f-Lipschitz and g is L_g-Lipschitz:
#   (f ∘ g) is L_f * L_g - Lipschitz
#
# For neural network D = f_k ∘ ... ∘ f_1:
#   L(D) ≤ ∏_i L(f_i)
#
# If each layer has L ≤ 1:
#   L(D) ≤ 1  (1-Lipschitz network)
#
# SPECTRAL NORMALIZATION:
# -------
# Normalize each weight matrix:
#   W_SN = W / σ_max(W)
#
# This ensures σ_max(W_SN) = 1, so L(W_SN) = 1.
#
# POWER ITERATION:
# ----
# To compute σ_max without full SVD, use power iteration:
#
#   Initialize u randomly
#   For t = 1, 2, ..., k:
#     v_t = W^T u_t / ||W^T u_t||
#     u_{t+1} = W v_t / ||W v_t||
#   σ ≈ u_k^T W v_k
#
# Converges to σ_max with linear convergence rate.
# Typically k=1 per training step is sufficient.
#
# COMPUTATIONAL COMPLEXITY:
# -----------
# With SN: O(m*n) per iteration (matrix-vector multiply)
# With SVD: O(min(m^2*n, m*n^2)) per update (expensive)
# With Gradient Penalty: O(batch_size) extra forward/backward
#
# SN is most efficient!
#
# MATHEMATICAL GUARANTEE:
# ---------
# With spectral normalization applied to all layers:
#   L(D) = ∏_i σ_max(W_i) = 1 * 1 * ... * 1 = 1
#
# Therefore D is 1-Lipschitz continuous everywhere.
#
# WEIGHT RESHAPING:
# ---------
# For Conv2d layer (out_channels, in_channels, kernel_h, kernel_w):
#   Reshape to (out_channels, in_channels * kernel_h * kernel_w)
#   This treats convolution as linear map on flattened inputs
#
# CONVERGENCE PROPERTIES:
# ---------
# 1. Power iteration converges exponentially to σ_max
# 2. Error at step t: O(λ_2^t) where λ_2 is second-largest singular value
# 3. Typically t=1 sufficient (error < 5%)
# 4. t=2-3 for higher accuracy
#
# PRACTICAL CONSIDERATIONS:
# ---------
# 1. Update u every forward pass (not every training step)
# 2. Use same u across batches (maintain between steps)
# 3. Initialize u randomly once, update smoothly
# 4. For non-square matrices, use W^T W formulation
# 5. Monitor σ_max values during training (should stabilize around 1)
