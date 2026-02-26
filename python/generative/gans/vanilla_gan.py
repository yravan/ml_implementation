"""
Vanilla GAN (Generative Adversarial Network)

Original paper: "Generative Adversarial Nets" (Goodfellow et al., 2014)
https://arxiv.org/abs/1406.2661

The fundamental GAN framework consists of two neural networks in competition:
- Generator G: Maps random noise z ~ p(z) to fake samples x_fake = G(z)
- Discriminator D: Classifies samples as real (x ~ p_data) or fake (G(z))

THEORY:
--------
The adversarial game is a min-max optimization:

    min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p(z)}[log(1 - D(G(z)))]

From the discriminator's perspective:
  - Maximizes log D(x) for real data (wants D(real) ≈ 1)
  - Maximizes log(1 - D(G(z))) for fake data (wants D(fake) ≈ 0)

From the generator's perspective:
  - Minimizes log(1 - D(G(z))) to fool the discriminator
  - Or equivalently, maximizes log D(G(z))

CONVERGENCE PROPERTIES:
- At Nash Equilibrium: D(x) = 1/2 for all x (discriminator cannot distinguish)
- p_g converges to p_data (generator learns the true distribution)
- Theoretical convergence guaranteed under ideal conditions (Goodfellow et al., 2014)

TRAINING CHALLENGES:
- Mode Collapse: Generator learns to produce only a subset of the true distribution
  - Discriminator becomes too good, creating zero gradients for the generator
  - Generator gets stuck in local optima

- Vanishing Gradients: When D(G(z)) → 0, log(1 - D(G(z))) saturates
  - Early in training, before G improves, gradients become very small
  - Remedy: Use max_z log D(G(z)) instead (non-saturating loss)

- Training Instability: Oscillating loss values, divergence
  - Hyperparameter sensitivity (lr, batch size, architecture)
  - Requires careful balancing of D and G training rates
  - Empirical heuristics: train D k times, then G once

TRAINING TRICKS:
1. Use max_z log D(G(z)) loss instead of min_z log(1 - D(G(z)))
2. Label smoothing: Use soft targets (0.9 for real, 0.1 for fake)
3. One-sided label smoothing: Only smooth real labels
4. Spectral normalization in discriminator layers
5. Batch normalization in generator (but NOT in discriminator output layer)
6. Monitor mode collapse using Inception Score or FID metrics
7. Early stopping before generator collapses into few modes

ARCHITECTURE GUIDELINES:
- Generator: Use transposed convolutions (Conv2dTranspose) for upsampling
- Discriminator: Use regular convolutions with stride for downsampling
- Activations: ReLU in generator (except output tanh/sigmoid), LeakyReLU in discriminator
- Normalization: Batch norm in generator, optionally in discriminator (but skip D's final layer)

OPTIMIZATION:
- Typical approach: Adam optimizer for both G and D
- Learning rates: Generally lr_G = 0.0002, lr_D = 0.0002, beta1 = 0.5
- Schedule: Can be constant or decayed (e.g., exponential decay)
"""

import numpy as np
from typing import Tuple, Optional
from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.conv.conv2d import Conv2d
from python.nn_core.normalization.batchnorm import BatchNorm2d
from python.nn_core.activations.relu import ReLU, LeakyReLU


class Generator(Module):
    """
    Generator network that maps latent vectors to data space.

    Input:
        z: latent vector from p(z), typically N(0, I) of dimension latent_dim

    Output:
        x_fake: Generated samples in data space (same shape as real data)

    Architecture example for image generation (28x28 grayscale):
        - FC layer: latent_dim -> 128 * 7 * 7
        - Conv2dTranspose: 128 -> 64, kernel=4, stride=2, padding=1 (14x14)
        - Conv2dTranspose: 64 -> 32, kernel=4, stride=2, padding=1 (28x28)
        - Conv2dTranspose: 32 -> 1, kernel=4, stride=1, padding=1 (28x28)
        - Output: Tanh activation for [-1, 1] range
    """

    def __init__(self, latent_dim: int = 100, channels: int = 1, img_size: int = 28):
        """
        Args:
            latent_dim: Dimension of input noise vector
            channels: Number of output channels (1 for grayscale, 3 for RGB)
            img_size: Size of generated image (assumes square images)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # TODO: Implement fully connected + convolutional layers
        # Hint: Use nn.Sequential with:
        #   - nn.Linear for latent_dim -> intermediate size
        #   - nn.BatchNorm1d for latent vector
        #   - nn.ReLU activation
        #   - Reshape to (batch, channels, h, w)
        #   - nn.ConvTranspose2d blocks
        #   - nn.BatchNorm2d after each conv (except last)
        #   - nn.ReLU activation (except last layer -> nn.Tanh)
        raise NotImplementedError(
            "Generator architecture not implemented. "
            "Hint: Start with a fully connected layer expanding latent_dim to "
            "a spatial size (e.g., 128*7*7 for 28x28 images), reshape, then use "
            "ConvTranspose2d blocks to upsample to target image size."
        )

    def forward(self, z):
        """
        Args:
            z: Latent vector (batch_size, latent_dim)

        Returns:
            Generated images (batch_size, channels, img_size, img_size)
        """
        raise NotImplementedError("Forward pass not implemented")


class Discriminator(Module):
    """
    Discriminator network that classifies samples as real or fake.

    Input:
        x: Real or fake samples

    Output:
        logit: Score indicating realness (higher = more real)

    Architecture notes:
        - Use Conv2d with stride=2 for downsampling (no max pooling)
        - Use LeakyReLU with negative_slope=0.2 throughout
        - NO batch norm in output layer (can destabilize training)
        - Final layer: Single neuron with no activation (or sigmoid for binary classification)
    """

    def __init__(self, channels: int = 1, img_size: int = 28):
        """
        Args:
            channels: Number of input channels (1 for grayscale, 3 for RGB)
            img_size: Size of input image
        """
        super(Discriminator, self).__init__()
        self.img_size = img_size

        # TODO: Implement convolutional layers
        # Hint: Use nn.Sequential with:
        #   - nn.Conv2d blocks with increasing feature maps (e.g., 32->64->128)
        #   - Stride=2 for downsampling
        #   - nn.LeakyReLU(0.2) activation
        #   - Optionally nn.BatchNorm2d (but NOT on first conv or last layer)
        #   - Flatten to (batch_size, features)
        #   - Final nn.Linear to 1 output neuron
        #   - Output activation: None (or Sigmoid for binary classification)
        raise NotImplementedError(
            "Discriminator architecture not implemented. "
            "Hint: Build a CNN that progressively downsamples input with Conv2d "
            "(stride=2) blocks using LeakyReLU(0.2). Final layer should output "
            "a single logit score (no batch norm on final layer)."
        )

    def forward(self, x):
        """
        Args:
            x: Samples to classify (batch_size, channels, img_size, img_size)

        Returns:
            Classification logit (batch_size, 1)
        """
        raise NotImplementedError("Forward pass not implemented")


class VanillaGAN:
    """
    Vanilla GAN trainer implementing the adversarial game.

    The training loop alternates between:
    1. Discriminator step: Maximize log D(x) + log(1 - D(G(z)))
    2. Generator step: Maximize log D(G(z)) (non-saturating loss)
    """

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        latent_dim: int = 100,
        device: str = "cpu",
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        beta1: float = 0.5,
    ):
        """
        Args:
            generator: Generator network
            discriminator: Discriminator network
            latent_dim: Dimension of latent noise vector
            device: Device to train on ("cpu" or "cuda")
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            beta1: Beta1 parameter for Adam optimizer
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        # TODO: Initialize optimizers
        # Hint: Create optimizers for G and D with specified learning rates
        raise NotImplementedError(
            "Optimizer initialization not implemented. "
            "Hint: Create two optimizers for G and D with specified learning rates."
        )

    def sample_noise(self, batch_size: int):
        """Sample latent vectors from standard normal distribution."""
        return np.random.randn(batch_size, self.latent_dim).astype(np.float32)

    def train_step(
        self,
        real_data,
        n_critic: int = 1,
    ) -> Tuple[float, float]:
        """
        Single training iteration with one discriminator step and one generator step.

        Args:
            real_data: Batch of real samples (batch_size, channels, h, w)
            n_critic: Number of discriminator updates per generator update

        Returns:
            (d_loss, g_loss): Discriminator and generator losses

        ALGORITHM:
        ----------
        1. Discriminator Step (k times):
           a. Sample real batch x ~ p_data, fake batch G(z) with z ~ p(z)
           b. Compute D_loss = -log D(x) - log(1 - D(G(z)))
                              = L_bce(D(x), 1) + L_bce(D(G(z)), 0)
           c. Backward and update D with optim_d.step()

        2. Generator Step:
           a. Sample noise z ~ p(z)
           b. Compute G_loss = log(1 - D(G(z))) [SATURATING]
                          or = -log D(G(z))       [NON-SATURATING, preferred]
           c. Backward and update G with optim_g.step()

        IMPORTANT GRADIENT HANDLING:
        - Only compute discriminator gradients during D step
        - Detach G(z) when computing D loss (don't backprop through G)
        - Only compute generator gradients during G step
        - Detach D when computing G loss if using double backprop

        LABEL SMOOTHING (optional):
        - Real labels: Use 0.9 instead of 1.0 (one-sided smoothing)
        - Fake labels: Use 0.0 (keep fixed)

        Returns:
            tuple: (discriminator_loss, generator_loss)
        """
        batch_size = real_data.shape[0]

        # Labels for real and fake samples
        real_label = 0.9  # One-sided label smoothing
        fake_label = 0.0

        # TODO: Implement discriminator training loop (n_critic iterations)
        # For each iteration:
        #   1. Zero discriminator gradients
        #   2. Forward real data through discriminator
        #   3. Compute loss for real samples: BCE(D(real), real_label)
        #   4. Sample noise and generate fake data
        #   5. Forward fake data through discriminator (detach G output)
        #   6. Compute loss for fake samples: BCE(D(fake), fake_label)
        #   7. Total D loss = real_loss + fake_loss
        #   8. Backward pass and optimize
        raise NotImplementedError(
            "Discriminator training step not implemented. "
            "Hint: In a loop for n_critic times, compute BCE loss for both "
            "real and fake samples, detach the generator output to avoid "
            "backprop through G, then update the discriminator."
        )

        # TODO: Implement generator training step
        # 1. Zero generator gradients
        # 2. Sample noise z
        # 3. Generate fake samples G(z)
        # 4. Forward through discriminator
        # 5. Compute loss: -log D(G(z)) (using real_label for non-saturating loss)
        # 6. Backward pass and optimize
        raise NotImplementedError(
            "Generator training step not implemented. "
            "Hint: Compute D(G(z)) without detaching, use non-saturating loss "
            "(BCE with real_label to maximize D(G(z))), then update G."
        )

    def generate_samples(self, num_samples: int = 16):
        """
        Generate samples from the generator.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated samples (num_samples, channels, h, w)
        """
        self.generator.eval()
        z = self.sample_noise(num_samples)
        fake_data = self.generator(z)
        self.generator.train()
        return fake_data

    def save_checkpoint(self, path: str):
        """Save generator and discriminator weights."""
        raise NotImplementedError(
            "Checkpoint saving not implemented. "
            "Hint: Use torch.save() to save state_dicts of both G and D."
        )

    def load_checkpoint(self, path: str):
        """Load generator and discriminator weights."""
        raise NotImplementedError(
            "Checkpoint loading not implemented. "
            "Hint: Use torch.load() to restore state_dicts of both G and D."
        )


def train_vanilla_gan(
    generator: Generator,
    discriminator: Discriminator,
    train_loader,
    num_epochs: int = 50,
    latent_dim: int = 100,
) -> Tuple[list, list]:
    """
    Training loop for Vanilla GAN.

    Args:
        generator: Generator network
        discriminator: Discriminator network
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device to train on
        latent_dim: Latent dimension

    Returns:
        (g_losses, d_losses): Lists of generator and discriminator losses per epoch

    TRAINING MONITORING:
    - Track average losses per epoch
    - Monitor for mode collapse (low diversity in generated samples)
    - Check if losses are diverging or oscillating wildly
    - Use Inception Score or FID to measure sample quality
    """
    gan = VanillaGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=latent_dim,
    )

    g_losses = []
    d_losses = []

    # TODO: Implement training loop
    # For each epoch:
    #   1. Iterate through batches in train_loader
    #   2. Call gan.train_step() for each batch
    #   3. Accumulate losses
    #   4. Log average epoch losses
    # Monitor for training instability and mode collapse
    raise NotImplementedError(
        "Training loop not implemented. "
        "Hint: Use nested loops (epochs, batches), call train_step(), "
        "accumulate losses, and track metrics."
    )

    return g_losses, d_losses


# MATHEMATICAL FORMULATION SUMMARY:
# ==================================
#
# Value Function (Min-Max Game):
#   V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
#
# Discriminator Objective (Maximization):
#   max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]
#
# Generator Objective (Minimization):
#   min_G E_z[log(1 - D(G(z)))]
#   (Equivalently: max_G E_z[log D(G(z))] - non-saturating)
#
# At Nash Equilibrium:
#   D*(x) = p_data(x) / (p_data(x) + p_g(x))
#   D*(x) = 1/2 when p_g = p_data
#   JSD(p_data || p_g) = log(4) at equilibrium
#
# Cross-Entropy Loss Implementation:
#   D_loss = BCE(D(x), 1) + BCE(D(G(z)), 0)
#   G_loss = BCE(D(G(z)), 1)  [non-saturating]
#
# Gradient Flow:
#   For D: gradients flow backward through discriminator only
#   For G: gradients flow through discriminator into generator
#
# Training Stability Conditions:
#   1. D learns faster than G early on (train D k times per G update)
#   2. Use non-saturating loss to avoid vanishing gradients
#   3. Label smoothing to reduce overfitting
#   4. Monitor Inception Score / FID for sample quality
