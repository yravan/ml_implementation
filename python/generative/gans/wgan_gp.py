"""
Wasserstein GAN with Gradient Penalty (WGAN-GP)

Original paper: "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017)
https://arxiv.org/abs/1704.00028

WGAN-GP is widely considered one of the best practical GAN formulations due to
its exceptional training stability, interpretable loss values, and elimination
of mode collapse. It fixes the weight clipping limitation of vanilla WGAN by
using gradient penalty instead.

THEORY & MOTIVATION:
---------------------
WGAN Uses weight clipping to enforce the 1-Lipschitz constraint on the critic.

Problems with Weight Clipping:
  1. CRUDE CONSTRAINT: Only approximately enforces 1-Lipschitz
  2. CAPACITY LIMITATION: Clips all weights to small range, reducing model capacity
  3. VANISHING GRADIENTS: Critic weights converge to boundaries (-c or c)
  4. INEFFICIENT OPTIMIZATION: Ignores actual gradient magnitudes
  5. FAILURE MODE: Weights cluster at boundaries, creating dead zones

Example: A weight at 0.01 and a weight at -0.01 are treated equally by clipping,
even though their gradient magnitudes might be very different. The model wastes
capacity by using the full [-c, c] range inefficiently.

Gradient Penalty Solution:
  Instead of clipping weights, add a regularization term to the loss:

    L_gp = λ * E_x̂[(||∇_x̂ D(x̂)||_2 - 1)^2]

  where x̂ is sampled uniformly on the line segment between real and fake samples:
    x̂ = α*x_real + (1-α)*x_fake, α ~ Uniform[0, 1]

Why This Works:
  1. FLEXIBILITY: No weight constraints, full model capacity
  2. SOFT CONSTRAINT: Penalizes violation but doesn't clip
  3. EFFICIENCY: Naturally selects important weight magnitudes
  4. SMOOTHNESS: Gradients flow smoothly throughout training
  5. MATHEMATICALLY ELEGANT: Directly enforces desired constraint

ADVANTAGES OVER WEIGHT CLIPPING:
  1. Larger critic capacity (no [-0.01, 0.01] bounds)
  2. Better gradient flow (no vanishing gradients)
  3. More stable convergence
  4. Better sample quality in fewer iterations
  5. Simpler implementation (just add regularization term)
  6. More reliable with different architectures/hyperparameters

MATHEMATICAL FORMULATION:
  Critic Loss with Gradient Penalty:
    L_D = -E_x~p_real[D(x)] + E_z~p(z)[D(G(z))] + λ*L_gp

  Gradient Penalty:
    L_gp = E_x̂[(||∇_x̂ D(x̂)||_2 - 1)^2]

  where x̂ = α*x_real + (1-α)*x_fake with α ~ Uniform[0, 1]

  This penalizes ||∇_x D(x)|| ≠ 1, encouraging 1-Lipschitz constraint everywhere.

IMPLEMENTATION DETAILS:
  1. Sample α uniformly from [0, 1] (shape: batch_size, 1, 1, 1 for images)
  2. Create interpolated samples: x̂ = α*x_real + (1-α)*x_fake
  3. Forward through critic: D(x̂)
  4. Compute gradients: ∇_x̂ D(x̂)
  5. Compute gradient norm: ||∇_x̂ D(x̂)||_2
  6. Penalize deviation from 1: (||∇|| - 1)^2
  7. Add to critic loss with weight λ

Gradient Computation in PyTorch:
  Required: torch.autograd.grad() with create_graph=True

  gradients = torch.autograd.grad(
      outputs=D(x̂),
      inputs=x̂,
      grad_outputs=torch.ones_like(D(x̂)),
      create_graph=True,
      retain_graph=False
  )

  - torch.autograd.grad(): Manually compute gradients
  - create_graph=True: Allow second-order derivatives (for penalty term)
  - grad_outputs: Seeds for backward pass (ones for scalar output)
  - returns: Tuple of gradients w.r.t. x̂

KEY HYPERPARAMETERS:
  1. Lambda (λ): Gradient penalty weight
     - Typical range: 10 to 100
     - 10 is most common (good balance)
     - Higher λ: Stricter 1-Lipschitz enforcement (smoother but slower)
     - Lower λ: Looser constraint (faster convergence but less stable)

  2. Critic Updates per Generator Update (k_critic):
     - Typical: 5 (same as WGAN)
     - Can be 1 for WGAN-GP (more stable than WGAN)
     - Some use adaptive scheduling

  3. Learning Rates:
     - Same for generator and critic (e.g., 0.0002)
     - Can use Adam now (not just RMSprop like WGAN)
     - Default: Adam with β1=0.5, β2=0.999

LOSS FUNCTION PROPERTIES:
  Critic Loss:
    L_D = -E[D(real)] + E[D(fake)] + 10*E[(||∇D(x̂)|| - 1)^2]
    - First two terms: Wasserstein distance
    - Last term: Gradient penalty (soft 1-Lipschitz enforcement)

  Generator Loss:
    L_G = -E[D(G(z))]
    - Same as WGAN (no gradient penalty needed)

CONVERGENCE PROPERTIES:
  1. STABILITY: Training curves are smooth and predictable
  2. INTERPRETABILITY: Loss values directly indicate sample quality
  3. NO MODE COLLAPSE: Generator naturally covers full data distribution
  4. CONVERGENCE: Provably converges to Nash equilibrium
  5. ROBUSTNESS: Works well across different architectures

TRAINING DYNAMICS:
  1. Loss values are real-valued, not probabilities
  2. Critic loss can be negative (unlike binary classification loss)
  3. Generator loss typically starts high and decreases
  4. No more vanishing gradient problem
  5. Training typically converges in 20-30 epochs (CIFAR-10)

MONITORING DURING TRAINING:
  1. Watch critic loss: Should gradually decrease (become more negative)
  2. Watch generator loss: Should gradually decrease (become more negative)
  3. Monitor gradient norm: Should cluster around 1.0
  4. Check Inception Score or FID for sample quality
  5. Both losses should show smooth trends

COMMON MISTAKES:
  1. Forgetting to set create_graph=True in torch.autograd.grad()
  2. Using wrong shape for α sampling (must broadcast correctly)
  3. Forgetting .detach() on fake samples in discriminator
  4. Using sigmoid/tanh on critic output (should be linear)
  5. Using too large λ (overly strict, slow convergence)
  6. Using too small λ (insufficient constraint, unstability)

ARCHITECTURE CONSIDERATIONS:
  1. Critic: Same as DCGAN discriminator but with linear output
  2. Generator: Same as DCGAN generator
  3. No weight clipping needed!
  4. Can use more layers and filters (no capacity limitation)
  5. BatchNorm generally works fine (unlike weight clipping WGAN)

PAPER CONTRIBUTIONS:
  1. Identified weight clipping as source of training issues
  2. Proposed gradient penalty as elegant replacement
  3. Proved convergence guarantees with gradient penalty
  4. Demonstrated state-of-the-art sample quality
  5. Showed compatibility with different optimizers (Adam, RMSprop)
  6. Provided comprehensive empirical validation

RELATION TO OTHER METHODS:
  - WGAN (Weight Clipping): Predecessor with capacity limitations
  - Spectral Normalization (SN-GAN): Alternative constraint method
  - Hinge Loss GAN: Related distance-based approach
  - Non-saturating GAN: Different loss formulation
"""

import numpy as np
from typing import Tuple, Optional
from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.conv.conv2d import Conv2d
from python.nn_core.normalization.batchnorm import BatchNorm2d
from python.nn_core.activations.relu import ReLU, LeakyReLU


class GPGenerator(Module):
    """
    Generator for WGAN-GP. Standard DCGAN-style architecture.

    No constraints on weights (unlike critic), just standard generator.
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
            latent_dim: Dimension of input noise
            feature_maps: Base number of feature maps
            img_channels: Number of output channels
            img_size: Output image size
        """
        super(GPGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.img_size = img_size

        # TODO: Implement generator
        # Hint: DCGAN-style with ConvTranspose2d blocks
        # Architecture for 64x64 RGB:
        #   1. Linear: 100 -> 512*4*4
        #   2. Reshape to (512, 4, 4)
        #   3. ConvTranspose2d: 512 -> 256, kernel=4, stride=2, padding=1
        #      + BatchNorm2d + ReLU (14x14)
        #   4. ConvTranspose2d: 256 -> 128, kernel=4, stride=2, padding=1
        #      + BatchNorm2d + ReLU (28x28)
        #   5. ConvTranspose2d: 128 -> 64, kernel=4, stride=2, padding=1
        #      + BatchNorm2d + ReLU (56x56)
        #   6. ConvTranspose2d: 64 -> 3, kernel=4, stride=2, padding=1
        #      + Tanh (64x64)
        raise NotImplementedError(
            "Generator architecture not implemented. "
            "Hint: Use standard DCGAN generator with ConvTranspose2d blocks."
        )

    def forward(self, z):
        """
        Args:
            z: Noise vector (batch_size, latent_dim)

        Returns:
            Generated image (batch_size, img_channels, img_size, img_size)
        """
        raise NotImplementedError("Forward pass not implemented")


class GPCritic(Module):
    """
    Critic for WGAN-GP. DCGAN-style discriminator with linear output.

    Key Features:
    1. Linear output layer (no sigmoid/tanh)
    2. No weight clipping (gradient penalty does this)
    3. Can have more capacity than weight-clipped WGAN
    4. Outputs real-valued scores (not probabilities)

    Gradient Penalty is applied to this critic's gradients w.r.t. interpolated samples.
    """

    def __init__(
        self,
        feature_maps: int = 64,
        img_channels: int = 3,
        img_size: int = 64,
    ):
        """
        Args:
            feature_maps: Base number of feature maps
            img_channels: Number of input channels
            img_size: Input image size
        """
        super(GPCritic, self).__init__()
        self.feature_maps = feature_maps
        self.img_size = img_size

        # TODO: Implement critic
        # Hint: DCGAN-style discriminator with linear output
        # Architecture for 64x64 RGB:
        #   1. Conv2d: 3 -> 64, kernel=4, stride=2, padding=1
        #      + LeakyReLU(0.2) (32x32)
        #      NO BatchNorm on first layer
        #   2. Conv2d: 64 -> 128, kernel=4, stride=2, padding=1
        #      + BatchNorm2d + LeakyReLU(0.2) (16x16)
        #   3. Conv2d: 128 -> 256, kernel=4, stride=2, padding=1
        #      + BatchNorm2d + LeakyReLU(0.2) (8x8)
        #   4. Conv2d: 256 -> 512, kernel=4, stride=2, padding=1
        #      + BatchNorm2d + LeakyReLU(0.2) (4x4)
        #   5. Flatten
        #   6. Linear: 512*4*4 -> 1 (output layer, NO activation)
        raise NotImplementedError(
            "Critic architecture not implemented. "
            "Hint: Use DCGAN discriminator with linear output (no sigmoid/tanh)."
        )

    def forward(self, x):
        """
        Args:
            x: Image batch (batch_size, img_channels, img_size, img_size)

        Returns:
            Critic score (batch_size, 1) - real-valued score
        """
        raise NotImplementedError("Forward pass not implemented")


class WGANGP:
    """
    WGAN with Gradient Penalty trainer.

    Most stable and effective GAN training method as of 2017.
    Combines Wasserstein distance with gradient penalty constraint.

    Key Properties:
    1. STABLE: Training curves are smooth and interpretable
    2. NO MODE COLLAPSE: Generator covers full distribution
    3. INTERPRETABLE LOSS: Can directly assess training progress
    4. NO VANISHING GRADIENTS: Even when supports separate
    5. ROBUST: Works reliably across architectures and hyperparameters
    """

    def __init__(
        self,
        generator: GPGenerator,
        critic: GPCritic,
        latent_dim: int = 100,
        device: str = "cpu",
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        lambda_gp: float = 10.0,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        """
        Args:
            generator: Generator network
            critic: Critic network
            latent_dim: Dimension of latent noise
            device: Device to train on
            lr_g: Learning rate for generator
            lr_d: Learning rate for critic (typically same as generator)
            lambda_gp: Gradient penalty weight (typical: 10)
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
        """
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp

        # TODO: Initialize optimizers
        # Hint: Create Adam-like optimizers for both generator and critic
        raise NotImplementedError(
            "Optimizer initialization not implemented. "
            "Hint: Create optimizers for both generator and critic."
        )

    def sample_noise(self, batch_size: int):
        """Sample latent vectors."""
        return np.random.randn(batch_size, self.latent_dim).astype(np.float32)

    def compute_gradient_penalty(
        self,
        critic: GPCritic,
        real_samples,
        fake_samples,
    ):
        """
        Compute gradient penalty for WGAN-GP.

        This is the core innovation: Instead of clipping weights, we penalize
        the gradient norm, enforcing the 1-Lipschitz constraint via regularization.

        Algorithm:
        ----------
        1. Sample interpolation coefficient α ∈ [0, 1]
           Shape: (batch_size, 1, 1, 1) for 4D images
           Allows broadcasting: α*x_real + (1-α)*x_fake

        2. Create interpolated samples:
           x̂ = α*x_real + (1-α)*x_fake
           x̂ ∈ [0, 1] lies on line segment between real and fake

        3. Forward through critic with gradient tracking:
           Use requires_grad=True for x̂ so we can compute ∇_x̂

        4. Compute gradients of critic output w.r.t. x̂:
           ∇_x̂ D(x̂) using torch.autograd.grad()

        5. Compute gradient norm:
           ||∇_x̂ D(x̂)||_2 (L2 norm of gradient vector)

        6. Penalize deviation from 1:
           (||∇|| - 1)^2
           This is minimized when ||∇|| = 1

        7. Average over batch:
           L_gp = E_x̂[(||∇_x̂ D(x̂)||_2 - 1)^2]

        Mathematical Details:
        - α samples uniformly from [0, 1]
        - For images: α shape is (batch_size, 1, 1, 1) to broadcast correctly
        - Gradient computation requires autograd tracking
        - create_graph=True allows backprop through the gradient computation itself

        Args:
            critic: Critic network
            real_samples: Real image batch (batch_size, channels, height, width)
            fake_samples: Fake image batch (same shape)

        Returns:
            gradient_penalty: Scalar tensor representing L_gp

        Interpretation:
        - When ||∇|| ≈ 1: L_gp ≈ 0 (constraint satisfied)
        - When ||∇|| >> 1: L_gp >> 0 (penalty applied)
        - When ||∇|| << 1: L_gp >> 0 (also penalized, but less common)
        """
        batch_size = real_samples.shape[0]

        # TODO: Implement gradient penalty computation
        # Step-by-step:
        #   1. Sample α uniformly from [0, 1]
        #      alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        #   2. Create interpolated samples (IMPORTANT: requires_grad=True)
        #      x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        #   3. Forward through critic
        #      critic_output = critic(x_hat)
        #   4. Compute gradients using torch.autograd.grad()
        #      gradients = torch.autograd.grad(
        #          outputs=critic_output,
        #          inputs=x_hat,
        #          grad_outputs=torch.ones_like(critic_output),
        #          create_graph=True,
        #          retain_graph=True,
        #      )[0]
        #   5. Reshape gradients to (batch_size, -1) for norm computation
        #   6. Compute L2 norm: torch.norm(gradients, p=2, dim=1)
        #      Should be shape (batch_size,)
        #   7. Compute penalty: (norm - 1)^2
        #   8. Return mean over batch: penalty.mean()
        raise NotImplementedError(
            "Gradient penalty computation not implemented. "
            "Hint: Sample alpha, interpolate, forward critic, compute gradients, "
            "compute norm, penalize deviation from 1."
        )

    def train_step(
        self,
        real_data,
        n_critic: int = 5,
    ) -> Tuple[float, float, float]:
        """
        Single WGAN-GP training iteration.

        Args:
            real_data: Batch of real images (batch_size, channels, height, width)
            n_critic: Number of critic updates per generator update (default: 5)

        Returns:
            (critic_loss, generator_loss, gp_loss): Training losses

        WGAN-GP TRAINING ALGORITHM:
        ===========================

        FOR EACH CRITIC UPDATE (repeat n_critic times):
        -----------------------------------------------
        1. Sample real batch x ~ p_data
        2. Sample noise z ~ p(z), generate fake: x_fake = G(z)

        3. Forward through critic:
           d_real = Critic(x_real)        # Score for real samples
           d_fake = Critic(x_fake.detach())  # Score for fake samples

        4. Compute Wasserstein distance:
           w_distance = E[d_real] - E[d_fake]

        5. Compute gradient penalty (see compute_gradient_penalty):
           gp = E_x̂[(||∇_x̂ D(x̂)|| - 1)^2]

        6. Total critic loss:
           loss_c = -E[d_real] + E[d_fake] + λ*gp

        7. Backward pass and update critic:
           loss_c.backward()
           optim_critic.step()
           optim_critic.zero_grad()

        AFTER ALL CRITIC UPDATES:
        -----------------------
        1. Sample noise z ~ p(z)
        2. Generate fake: x_fake = G(z)

        3. Forward through critic:
           d_fake = Critic(x_fake)

        4. Generator loss:
           loss_g = -E[d_fake]
           (Generator wants to maximize critic's score for fakes)

        5. Backward pass and update generator:
           loss_g.backward()
           optim_generator.step()
           optim_generator.zero_grad()

        LOSS FUNCTION DETAILS:
        ---------------------
        Critic Loss:
          L_c = -E_real[D(x)] + E_fake[D(G(z))] + λ*E_x̂[(||∇D(x̂)|| - 1)^2]

          Three components:
          1. -E[D(real)]: Want high scores for real
          2. +E[D(fake)]: Want low scores for fake
          3. +λ*GP: Enforce 1-Lipschitz via gradient penalty

        Generator Loss:
          L_g = -E_fake[D(G(z))]

          Generator wants to maximize D(G(z)) (or minimize -D(G(z)))

        GRADIENT FLOW:
        -----
        Critic Update:
          - Gradients flow: Real -> Critic -> backward
          - Gradients flow: Fake -> Critic (detached) -> backward
          - Gradients flow: Interpolated -> Critic -> gradient -> backward (via GP)
          - Critic weights updated

        Generator Update:
          - Gradients flow: Noise -> Generator -> Critic -> backward
          - Only discriminator weights are updated by critic loss
          - Generator weights updated by generator loss

        NUMERICAL STABILITY:
        ----------
        - Output logits are real-valued (not bounded to [0,1])
        - No log applied to loss (no log(0) issues)
        - Gradient penalty naturally regularizes gradients
        - Both losses can be negative

        MONITORING LOSSES:
        --------
        Healthy Training:
          - Critic loss: Decreases (becomes more negative) over training
          - Generator loss: Decreases over training
          - Gradient penalty: Should be small (~0) if constraint satisfied
          - Both curves should be smooth

        Unhealthy Signs:
          - Critic loss increasing: Maybe λ too high, or architecture issues
          - Generator loss increasing: Usually means critic dominates
          - Gradient penalty huge: λ too high relative to problem scale
          - Oscillating losses: Learning rates too high
        """
        batch_size = real_data.shape[0]

        # ===== CRITIC TRAINING LOOP =====
        for _ in range(n_critic):
            # TODO: Implement critic training step
            # 1. Zero critic gradients: optim_critic.zero_grad()
            # 2. Sample real batch (already have real_data)
            # 3. Sample noise and generate fake:
            #    z = self.sample_noise(batch_size)
            #    fake_data = self.generator(z)
            # 4. Forward real through critic (no detach):
            #    d_real = self.critic(real_data)
            # 5. Forward fake through critic (WITH DETACH):
            #    d_fake = self.critic(fake_data.detach())
            # 6. Wasserstein distance:
            #    wasserstein = d_real.mean() - d_fake.mean()
            # 7. Compute gradient penalty:
            #    gp = self.compute_gradient_penalty(
            #        self.critic, real_data, fake_data.detach()
            #    )
            # 8. Total critic loss:
            #    loss_critic = -wasserstein + self.lambda_gp * gp
            # 9. Backward:
            #    loss_critic.backward()
            #    optim_critic.step()
            raise NotImplementedError(
                "Critic training step not implemented. "
                "Hint: Compute Wasserstein distance + gradient penalty, "
                "backward, and update critic. Detach fake samples."
            )

        # ===== GENERATOR TRAINING STEP =====
        # TODO: Implement generator training step
        # 1. Zero generator gradients: optim_generator.zero_grad()
        # 2. Sample noise:
        #    z = self.sample_noise(batch_size)
        # 3. Generate fake:
        #    fake_data = self.generator(z)
        # 4. Forward through critic (NO DETACH - want gradients):
        #    d_fake = self.critic(fake_data)
        # 5. Generator loss (minimize -D(G(z))):
        #    loss_generator = -d_fake.mean()
        # 6. Backward:
        #    loss_generator.backward()
        #    optim_generator.step()
        raise NotImplementedError(
            "Generator training step not implemented. "
            "Hint: Generate fake samples, compute -E[D(G(z))], backward, update."
        )

    def generate_samples(self, num_samples: int = 16):
        """Generate samples from the generator."""
        self.generator.eval()
        z = self.sample_noise(num_samples)
        fake_data = self.generator(z)
        self.generator.train()
        return fake_data

    def save_checkpoint(self, path: str):
        """Save generator and critic weights."""
        raise NotImplementedError(
            "Checkpoint saving not implemented. "
            "Hint: Save state_dicts and optimizers."
        )

    def load_checkpoint(self, path: str):
        """Load generator and critic weights."""
        raise NotImplementedError(
            "Checkpoint loading not implemented."
        )


def train_wgan_gp(
    generator: GPGenerator,
    critic: GPCritic,
    train_loader,
    num_epochs: int = 50,
    latent_dim: int = 100,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
) -> Tuple[list, list, list]:
    """
    Training loop for WGAN-GP.

    Args:
        generator: Generator network
        critic: Critic network
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device to train on
        latent_dim: Latent dimension
        n_critic: Number of critic updates per generator update
        lambda_gp: Gradient penalty weight

    Returns:
        (g_losses, c_losses, gp_losses): Loss lists for monitoring

    WGAN-GP TRAINING BENEFITS:
    ============================
    1. EXCEPTIONAL STABILITY: Loss curves are smooth and predictable
    2. NO MODE COLLAPSE: Generator covers full data distribution
    3. INTERPRETABLE LOSSES: Can directly track training progress
    4. NO VANISHING GRADIENTS: Strong signal throughout training
    5. ROBUSTNESS: Works reliably with different architectures
    6. SCALABILITY: Works well on high-resolution images
    7. THEORETICALLY SOUND: Provable convergence guarantees

    MONITORING DURING TRAINING:
    ==========================
    - Critic loss should decrease (become more negative)
    - Generator loss should decrease
    - Gradient penalty should be small and stable
    - Both loss curves should be smooth (not oscillating wildly)
    - As training progresses, Inception Score should increase

    CONVERGENCE INDICATORS:
    ======================
    - Loss values plateau: Training has converged
    - Smooth curves: Good hyperparameters chosen
    - Balanced G/D losses: Neither dominates the other
    - GP stays small: Gradient penalty is effective

    TROUBLESHOOTING:
    ================
    If training is unstable:
    1. Check if λ_gp is appropriate (try 10 first)
    2. Verify learning rates (0.0002 typical)
    3. Check batch size (128 or larger often helps)
    4. Ensure critic forward is correct (linear output)
    5. Verify gradient penalty computation (create_graph=True)

    If sample quality is poor:
    1. Train longer (30-50+ epochs)
    2. Increase critic capacity (more layers/filters)
    3. Increase generator capacity
    4. Try different λ_gp values
    5. Check that losses are actually decreasing

    Example Loss Values (CIFAR-10, after convergence):
    - Critic loss: -2.5 to -3.5 (negative because of Wasserstein)
    - Generator loss: -2.5 to -3.0
    - Gradient penalty: 0.001 to 0.01 (very small)
    """
    gan = WGANGP(
        generator=generator,
        critic=critic,
        latent_dim=latent_dim,
        lambda_gp=lambda_gp,
    )

    g_losses = []
    c_losses = []
    gp_losses = []

    # TODO: Implement training loop
    # For each epoch:
    #   For each batch in train_loader:
    #     g_loss, c_loss, gp = gan.train_step(batch, n_critic=n_critic)
    #     Accumulate losses
    #   Log epoch averages
    #   Optionally save samples or checkpoints
    raise NotImplementedError(
        "Training loop not implemented. "
        "Hint: Standard epoch/batch loop calling train_step. "
        "Return accumulated loss lists."
    )

    return g_losses, c_losses, gp_losses


# COMPREHENSIVE MATHEMATICAL REFERENCE:
# ======================================
#
# WASSERSTEIN DISTANCE:
# --------------------
# Definition (Kantorovich formulation):
#   W(p, q) = sup_{D ∈ Lip1} E_x~p[D(x)] - E_x~q[D(x)]
#
# Where Lip1 is the set of 1-Lipschitz functions.
#
# LIPSCHITZ CONTINUOUS FUNCTIONS:
# -------
# A function D is L-Lipschitz continuous if:
#   |D(x1) - D(x2)| ≤ L * ||x1 - x2||  for all x1, x2
#
# For 1-Lipschitz (L=1):
#   |D(x1) - D(x2)| ≤ ||x1 - x2||
#
# This means ||∇D(x)|| ≤ 1 almost everywhere (gradient norm bounded by 1).
#
# GRADIENT PENALTY ENFORCEMENT:
# ---
# L_gp = E_x̂[(||∇_x̂ D(x̂)|| - 1)^2]
#
# This is minimized when ||∇D(x̂)|| = 1, enforcing 1-Lipschitz constraint.
#
# WGAN-GP OBJECTIVE:
# ---------
# Generator:
#   min_G E_z[D(G(z))]
#
# Critic:
#   max_D E_x~p_real[D(x)] - E_z[D(G(z))] - λ*E_x̂[(||∇D(x̂)|| - 1)^2]
#
# Equivalently, minimize critic loss:
#   L_D = -E[D(real)] + E[D(fake)] + λ*L_gp
#
# ADVANTAGES OF GRADIENT PENALTY VS WEIGHT CLIPPING:
# ----------
# Weight Clipping (WGAN):
#   - Pros: Simple, interpretable, enforces constraint globally
#   - Cons: Capacity limited, weak gradients, inefficient
#   - Failure: Weights cluster at boundaries
#
# Gradient Penalty (WGAN-GP):
#   - Pros: Full capacity, strong gradients, efficient
#   - Cons: Slightly more complex (requires gradient computation)
#   - Success: Gradients naturally regulated
#
# INTERPOLATION SAMPLING:
# ----
# x̂ = α*x_real + (1-α)*x_fake
# where α ~ Uniform[0, 1]
#
# Why sample on interpolated path?
#   - Covers the space where critic should enforce constraint
#   - Between real and fake is where constraints matter most
#   - Numerically stable (convex combination, no extrapolation)
#
# GRADIENT COMPUTATION:
# ---
# torch.autograd.grad(
#     outputs=D(x̂),
#     inputs=x̂,
#     grad_outputs=torch.ones_like(D(x̂)),
#     create_graph=True,
#     retain_graph=False
# )
#
# Parameters:
#   - outputs: Scalar we're differentiating (D(x̂))
#   - inputs: Variable we're differentiating w.r.t. (x̂)
#   - grad_outputs: Weights for output (ones since D outputs scalar)
#   - create_graph=True: Allow backprop through this gradient (needed for penalty)
#   - retain_graph=False: Free memory by not retaining computation graph
#
# LOSS FUNCTION COMPOSITION:
# --
# Total Critic Loss:
#   L_D = L_wasserstein + λ*L_gp
#       = -E[D(real)] + E[D(fake)] + λ*E[(||∇|| - 1)^2]
#
# Three distinct components:
#   1. L_real = -E[D(real)]: Maximize scores on real data
#   2. L_fake = +E[D(fake)]: Minimize scores on fake data
#   3. L_gp = λ*E[(||∇|| - 1)^2]: Enforce 1-Lipschitz
#
# EXPECTED BEHAVIOR:
# ---------
# Healthy Training:
#   - L_wasserstein trends toward lower values (more negative)
#   - L_gp stays small (less than 0.1 typically)
#   - Both losses decrease smoothly
#   - Sample quality increases over time
#
# Unhealthy Patterns:
#   - L_gp >> L_wasserstein: λ too large
#   - L_gp ≈ 0 but L_wasserstein high: λ too small
#   - Oscillating losses: LR too high
#   - Both losses increasing: Architecture/data issue
#
# HYPERPARAMETER TUNING:
# ----------
# λ (Gradient Penalty Weight):
#   - Default: 10 (works well in most cases)
#   - Too high (λ > 50): Over-constrained, slow training
#   - Too low (λ < 1): Under-constrained, unstable
#   - Sweet spot: 1-10 for most problems
#
# Learning Rate:
#   - Generator: 0.0001 - 0.0002
#   - Critic: 0.0001 - 0.0002 (usually same)
#   - Can use higher LR with gradient penalty (less sensitive)
#
# Batch Size:
#   - Larger batches (128+) improve stability
#   - Smaller batches (32) can work but less stable
#   - Gradient penalty computed per-batch
#
# Critic Updates (n_critic):
#   - Default: 5 (critic trained 5x per generator update)
#   - Can be lower with WGAN-GP (sometimes 1 works!)
#   - Higher n_critic = more stable but slower
#
# CONVERGENCE THEORY:
# ---------
# Gulrajani et al. (2017) proved:
#   1. WGAN-GP critic can represent any 1-Lipschitz function
#   2. Training achieves Nash equilibrium
#   3. No mode collapse at equilibrium
#   4. Convergence is guaranteed under standard assumptions
#   5. Gradient penalty provides efficient constraint enforcement
#
# PRACTICAL IMPACT:
# --------
# WGAN-GP became the de-facto standard GAN training approach because:
#   1. Exceptional training stability (smooth learning curves)
#   2. Elimination of mode collapse (full support coverage)
#   3. Interpretable loss values (directly indicates quality)
#   4. Strong empirical results (state-of-the-art sample quality)
#   5. Robustness to hyperparameter choices
#   6. Scalability (works well for high-resolution generation)
#   7. Theoretical guarantees (convergence proof)
