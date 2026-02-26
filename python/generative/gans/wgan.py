"""
Wasserstein GAN (WGAN)

Original paper: "Wasserstein GAN" (Arjovsky et al., 2017)
https://arxiv.org/abs/1701.07957

WGAN addresses fundamental training instabilities in GANs by replacing the
Jensen-Shannon divergence (from Vanilla GAN) with the Wasserstein distance,
which provides more meaningful gradient signals even when supports don't overlap.

THEORY:
--------
Vanilla GAN Problem:
  - Uses KL divergence or JS divergence as implicit objective
  - When supports of p_real and p_g don't overlap, divergence is constant
  - Provides zero gradient signal to generator -> vanishing gradients
  - Mode collapse stems from this fundamental issue

Wasserstein Distance (Earth Mover Distance):
  W(p_real, p_g) = inf_{γ ∈ Π} E_{(x,y)~γ}[||x - y||]

  where Π is the set of all joint distributions γ(x,y) with marginals p_real and p_g.

  This measures the minimum "cost" to transport one distribution to another.

Key Advantages:
  1. Continuous and differentiable everywhere (even when supports don't overlap)
  2. Provides meaningful gradients: ∇W tells direction to move p_g toward p_real
  3. Convergence of W(p_real, p_g) → 0 correlates with visual quality improvement
  4. Much more stable training than JS divergence
  5. No mode collapse issues (generator learns to cover full support)

WASSERSTEIN GAN OBJECTIVE:
  W(p_real, p_g) = max_{D ∈ 1-Lipschitz} E_x~p_real[D(x)] - E_x~p_g[D(x)]

Note: The critic (discriminator) must be 1-Lipschitz continuous.

Dual formulation (what we actually optimize):
  max_D E_x~p_real[D(x)] - E_z~p(z)[D(G(z))]  [Critic maximizes Wasserstein]
  min_G E_z~p(z)[D(G(z))]                      [Generator minimizes it]

Critical Difference from Vanilla GAN:
  - Critic outputs are NOT probabilities (no sigmoid/tanh on final layer)
  - Critic outputs are real-valued scores (linear output)
  - No log applied to outputs (uses L1 loss, not BCE)
  - Generator loss is -E_z[D(G(z))], not log(1-D(G(z)))

ENFORCING LIPSCHITZ CONSTRAINT:
  To ensure critic is 1-Lipschitz, WGAN uses Weight Clipping:

  - After each discriminator update, clip weights to [-0.01, 0.01]
  - This forces ||∇D|| ≤ 1 approximately
  - Very simple but has drawbacks:
    * Can lead to vanishing gradients for discriminator
    * Limits model capacity
    * Critic becomes harder to train

  (Weight Clipping is crude; Gradient Penalty is better - see wgan_gp.py)

TRAINING ADVANTAGES:
  1. Meaningful loss values: W(p_real, p_g) ↔ visual quality
  2. Stable convergence: Loss curves are smooth and interpretable
  3. No mode collapse: Generator covers full support
  4. Better gradient flow: Non-vanishing gradients throughout training
  5. Less hyperparameter sensitivity than Vanilla GAN

TRAINING DYNAMICS:
  - Critic typically trained more than generator (5-10 updates per G update)
  - No vanishing gradient problem
  - Loss values are interpretable (lower = better)
  - Training curves much smoother than Vanilla GAN

LOSS FUNCTION COMPARISON:
  Vanilla GAN:
    D_loss = -log D(x) - log(1 - D(G(z)))
    G_loss = -log D(G(z))
    D output: sigmoid ∈ [0, 1] (probability)

  WGAN:
    D_loss = -E[D(x)] + E[D(G(z))]  (with weight clipping)
    G_loss = -E[D(G(z))]
    D output: Linear ∈ ℝ (score)

HYPERPARAMETERS:
  - Critic learning rate: Usually higher than generator (e.g., 0.00005)
  - Generator learning rate: Lower than critic (e.g., 0.00005)
  - Number of critic updates: 5-10 per generator update
  - Weight clip value: 0.01 (standard)
  - No learning rate scheduling needed (training is stable)

CONVERGENCE PROPERTIES:
  - Convergence guaranteed (Arjovsky et al., 2017)
  - Nash equilibrium exists and is unique
  - No mode collapse at equilibrium
  - Visual quality improves as loss decreases

LIMITATIONS OF WEIGHT CLIPPING:
  - Capacity constraints: Model can't represent full hypothesis class
  - Vanishing gradients: Critics tend to exploit boundaries
  - Unstable optimization: Oscillating loss values possible
  - Not all weight configurations are 1-Lipschitz

  -> WGAN-Gradient Penalty (wgan_gp.py) is preferred

PAPER CONTRIBUTIONS:
  1. Identified JS divergence as source of training instability
  2. Proposed Wasserstein distance as superior objective
  3. Proved convergence and mode coverage properties
  4. Demonstrated significantly better training stability
  5. Showed interpretable loss values correlate with sample quality
"""

import numpy as np
from typing import Tuple, Optional
from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.conv.conv2d import Conv2d
from python.nn_core.normalization.batchnorm import BatchNorm2d
from python.nn_core.activations.relu import ReLU, LeakyReLU


class WGANGenerator(Module):
    """
    Generator for WGAN. Similar to DCGAN generator.

    Output layer: Linear activation (not tanh) since we're generating
    unnormalized samples. For images, scale to appropriate range post-generation.

    Architecture mirrors DCGAN but designed to work with critic (not discriminator).
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
            latent_dim: Dimension of latent noise
            feature_maps: Base number of feature maps
            img_channels: Number of output channels
            img_size: Output image size
        """
        super(WGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.img_size = img_size

        # TODO: Implement WGAN generator
        # Hint: Similar to DCGAN generator:
        #   1. Linear: latent_dim -> feature_maps*8*4*4
        #   2. Reshape to spatial
        #   3. ConvTranspose2d blocks with BatchNorm and ReLU
        #   4. Final output: Tanh for [-1, 1] or identity for unnormalized
        raise NotImplementedError(
            "WGAN Generator architecture not implemented. "
            "Hint: Use DCGAN-style architecture with ConvTranspose2d blocks."
        )

    def forward(self, z):
        """
        Args:
            z: Latent vector (batch_size, latent_dim)

        Returns:
            Generated image (batch_size, img_channels, img_size, img_size)
        """
        raise NotImplementedError("Forward pass not implemented")


class WGANCritic(Module):
    """
    Critic network for WGAN (replaces discriminator from Vanilla GAN).

    Key Differences from Discriminator:
    1. Output layer: Linear activation (outputs real-valued scores, not probabilities)
    2. No sigmoid/tanh on final layer
    3. Outputs are not bounded to [0, 1]
    4. Loss function is linear: -E[D(x)] + E[D(G(z))]

    Architecture: DCGAN-style CNN with linear output

    Weight Clipping:
    - After each training step, weights are clipped to [-clip_value, clip_value]
    - This enforces approximate 1-Lipschitz constraint
    - Clip value typically 0.01
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
        super(WGANCritic, self).__init__()
        self.feature_maps = feature_maps
        self.img_size = img_size

        # TODO: Implement WGAN critic
        # Hint: Similar to DCGAN discriminator:
        #   1. Conv2d blocks with stride=2 (no maxpool)
        #   2. LeakyReLU(0.2) activation
        #   3. BatchNorm2d after conv layers
        #   4. Final layer: Linear to 1 (NO activation - outputs real number)
        #   5. No sigmoid/tanh on output
        raise NotImplementedError(
            "WGAN Critic architecture not implemented. "
            "Hint: Use DCGAN-style discriminator but with linear output "
            "(no sigmoid/tanh activation on final layer)."
        )

    def forward(self, x):
        """
        Args:
            x: Image batch (batch_size, img_channels, img_size, img_size)

        Returns:
            Critic score (batch_size, 1) - real-valued, not probability
        """
        raise NotImplementedError("Forward pass not implemented")

    def clip_weights(self, clip_value: float = 0.01):
        """
        Clip weights to enforce Lipschitz constraint.

        Args:
            clip_value: Value to clip weights to [-clip_value, clip_value]

        Called after each critic update to maintain 1-Lipschitz constraint.

        Implementation:
          for p in self.parameters():
              p.data.clamp_(-clip_value, clip_value)
        """
        raise NotImplementedError(
            "Weight clipping not implemented. "
            "Hint: Use p.data.clamp_(-clip_value, clip_value) for each parameter."
        )


class WGAN:
    """
    WGAN trainer with Wasserstein distance objective and weight clipping.

    Key differences from Vanilla GAN:
    1. Critic outputs real-valued scores (not probabilities)
    2. Loss is linear: E[critic(real)] - E[critic(fake)]
    3. Critic is trained multiple times per generator update (k_critic >= 5)
    4. Weight clipping enforces 1-Lipschitz constraint
    5. No vanishing gradient problem
    6. Loss values are interpretable and correlate with quality
    """

    def __init__(
        self,
        generator: WGANGenerator,
        critic: WGANCritic,
        latent_dim: int = 100,
        device: str = "cpu",
        lr_g: float = 0.00005,
        lr_d: float = 0.00005,
        clip_value: float = 0.01,
    ):
        """
        Args:
            generator: WGAN Generator
            critic: WGAN Critic
            latent_dim: Latent dimension
            device: Device to train on
            lr_g: Learning rate for generator
            lr_d: Learning rate for critic (typically same as G)
            clip_value: Weight clipping value
        """
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.clip_value = clip_value

        # TODO: Initialize optimizers
        # Note: RMSprop typically works better than Adam for WGAN
        # (Adam accumulates gradient history which interferes with weight clipping)
        raise NotImplementedError(
            "Optimizer initialization not implemented. "
            "Hint: Create optimizers for critic and generator."
        )

    def sample_noise(self, batch_size: int):
        """Sample latent vectors from standard normal."""
        return np.random.randn(batch_size, self.latent_dim).astype(np.float32)

    def train_step(
        self,
        real_data,
        k_critic: int = 5,
    ) -> Tuple[float, float]:
        """
        Single WGAN training iteration.

        Args:
            real_data: Batch of real images
            k_critic: Number of critic updates per generator update

        Returns:
            (critic_loss, generator_loss)

        WGAN TRAINING ALGORITHM:
        ========================

        1. Critic Update (k_critic times):
           For each iteration:
            a. Sample real batch x ~ p_data
            b. Sample noise z ~ p(z)
            c. Generate fake: x_fake = G(z)
            d. Forward through critic:
                 D_real = Critic(x)
                 D_fake = Critic(x_fake.detach())
            e. Wasserstein distance: W = E[D_real] - E[D_fake]
            f. Critic loss (maximize): -E[D_real] + E[D_fake]
            g. Backprop and update critic
            h. Clip critic weights to [-clip_value, clip_value]

        2. Generator Update:
           a. Sample noise z ~ p(z)
           b. Generate fake: x_fake = G(z)
           c. Forward through critic: D_fake = Critic(x_fake)
           d. Generator loss (minimize): -E[D_fake]
              (Generator wants to maximize D_fake)
           e. Backprop and update generator
           f. NO weight clipping on generator

        Key Differences from Vanilla GAN:
        - No sigmoid/tanh on critic output (real-valued scores)
        - Loss is linear: E[D_real] - E[D_fake], not log probabilities
        - Critic must be clipped after each update
        - Typically k_critic = 5-10 (train critic much more than generator)
        - Optimizer works better with weight clipping

        Loss Interpretation:
        - critic_loss = E[D_real] - E[D_fake] ≈ Wasserstein distance
        - Negative generator_loss = E[D_fake]
        - Lower loss = better alignment between p_real and p_g
        - Loss is NOT bounded to [-1, 1] like binary cross-entropy
        """
        batch_size = real_data.shape[0]

        # TODO: Implement critic training (k_critic iterations)
        # For each iteration:
        #   1. Zero critic gradients
        #   2. Forward real data: critic_real = Critic(real_data)
        #   3. Compute real loss: -mean(critic_real)
        #   4. Sample z, generate fake: G(z)
        #   5. Forward fake: critic_fake = Critic(fake.detach())
        #   6. Compute fake loss: mean(critic_fake)
        #   7. Total critic loss = fake_loss + real_loss
        #   8. Backward and update critic
        #   9. Clip critic weights: critic.clip_weights(clip_value)
        raise NotImplementedError(
            "Critic training step not implemented. "
            "Hint: Maximize E[D(x)] - E[D(G(z))], then clip weights."
        )

        # TODO: Implement generator training
        # 1. Zero generator gradients
        # 2. Sample z
        # 3. Generate fake: G(z)
        # 4. Forward through critic: critic_fake = Critic(fake)
        # 5. Generator loss: -mean(critic_fake)
        #    (Generator wants high critic scores for fake samples)
        # 6. Backward and update generator
        # 7. NO clipping on generator weights
        raise NotImplementedError(
            "Generator training step not implemented. "
            "Hint: Minimize -E[D(G(z))] to maximize critic's scoring."
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


def train_wgan(
    generator: WGANGenerator,
    critic: WGANCritic,
    train_loader,
    num_epochs: int = 50,
    latent_dim: int = 100,
    k_critic: int = 5,
) -> Tuple[list, list]:
    """
    Training loop for WGAN.

    Args:
        generator: WGAN Generator
        critic: WGAN Critic
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device to train on
        latent_dim: Latent dimension
        k_critic: Number of critic updates per generator update

    Returns:
        (g_losses, c_losses): Generator and critic losses

    WGAN Training Benefits:
    - Stable convergence (smooth loss curves)
    - Interpretable loss values (correlate with quality)
    - No mode collapse
    - Better gradient flow
    - Loss can be monitored directly (no need for IS/FID)
    """
    gan = WGAN(
        generator=generator,
        critic=critic,
        latent_dim=latent_dim,
    )

    g_losses = []
    c_losses = []

    # TODO: Implement training loop
    # For each epoch:
    #   For each batch in train_loader:
    #     Call gan.train_step(batch, k_critic=k_critic)
    #     Log losses
    raise NotImplementedError(
        "Training loop not implemented. "
        "Hint: Standard epoch/batch loop with train_step calls."
    )

    return g_losses, c_losses


# MATHEMATICAL FORMULATION:
# =========================
#
# Wasserstein Distance:
#   W(p_real, p_g) = sup_{D ∈ 1-Lipschitz} E_x~p_real[D(x)] - E_x~p_g[D(x)]
#
# WGAN Objective:
#   max_D E_x~p_real[D(x)] - E_z[D(G(z))]     [Critic maximizes]
#   min_G E_z[D(G(z))]                         [Generator minimizes]
#
# Loss Functions:
#   Critic loss: -E[D(real)] + E[D(fake)] (to be maximized, so minimize negative)
#   Generator loss: -E[D(fake)] (minimize this, or equivalently max E[D(fake)])
#
# Lipschitz Constraint:
#   ||D(x1) - D(x2)|| / ||x1 - x2|| ≤ 1
#   Weight clipping: w ∈ [-c, c] approximately enforces this
#
# Wasserstein Distance Properties:
#   1. Continuous everywhere (even when supports don't overlap)
#   2. Meaningful gradients: ∇W points direction to improve p_g
#   3. Mode coverage: Incentivizes full support coverage
#   4. Convergence signal: W ↓ correlates with quality ↑
#
# Comparison to Vanilla GAN:
#   Vanilla GAN:
#     - Uses JS divergence (JS(p_real, p_g))
#     - Zero gradient when supports don't overlap
#     - Binary classification framing (sigmoid output)
#     - Prone to mode collapse and vanishing gradients
#
#   WGAN:
#     - Uses Wasserstein distance (Earth Mover Distance)
#     - Continuous gradient everywhere
#     - Regression framing (real-valued output)
#     - Mode collapse resolved, stable training
#
# Weight Clipping Drawback:
#   - Critic capacity limited by clipping range
#   - Can lead to weak gradients
#   - Better solution: Gradient Penalty (see wgan_gp.py)
#
# Optimizer Choice:
#   - RMSprop preferred (not Adam)
#   - Adam accumulates gradient history which interferes with clipping
#   - RMSprop works better with changing weight magnitudes
#
# Hyperparameter Notes:
#   - Learning rate: Same for G and D (0.00005 typical)
#   - k_critic: Usually 5 (critic trained 5x per G update)
#   - Clip value: 0.01 standard (range [-0.01, 0.01])
#   - Batch size: 64 typical (CIFAR-10)
#   - No learning rate decay needed (stable training)
