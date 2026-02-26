"""
Deep Convolutional GAN (DCGAN)

Original paper: "Unsupervised Representation Learning with Deep Convolutional
Generative Adversarial Networks" (Radford et al., 2015)
https://arxiv.org/abs/1511.06434

DCGAN introduced architectural guidelines that significantly stabilized GAN training.
These guidelines became the foundation for most subsequent deep generative models.

THEORY:
--------
DCGAN improves upon Vanilla GAN by:

1. ARCHITECTURAL STABILITY:
   - Replaces max pooling with strided convolutions (discriminator)
   - Replaces pooling with fractional-strided convolutions (generator)
   - Uses batch normalization in both G and D (though not on D's output layer)
   - Employs ReLU in generator (except final tanh layer)
   - Uses LeakyReLU in discriminator with slope 0.2

2. THEORETICAL INSIGHTS:
   - Strided convolutions allow the network to learn its own spatial downsampling
   - Batch normalization stabilizes training by reducing internal covariate shift
   - These design choices empirically reduce mode collapse and training divergence

3. CONVOLUTIONAL INDUCTIVE BIAS:
   - Leverages spatial structure in image data
   - Hierarchical feature learning (low-level edges -> high-level objects)
   - Much fewer parameters than fully connected networks
   - Better sample quality and faster convergence than Vanilla GAN

TRAINING DYNAMICS:
   - Convergence is more stable than Vanilla GAN
   - Mode collapse still possible but less severe
   - Training typically converges within 25-30 epochs on CIFAR-10
   - Learning curves are more predictable

ARCHITECTURAL GUIDELINES (from the paper):
   1. Replace pooling with strided convolutions (discriminator)
   2. Replace pooling with fractional-strided convolutions (generator)
   3. Batch normalization in both G and D (except D output layer)
   4. ReLU in generator (except final tanh)
   5. LeakyReLU in discriminator (slope=0.2)
   6. No fully connected layers except output
   7. Avoid binary classification layer (use intermediate logits instead)

LAYER CONFIGURATION PRINCIPLES:
   - Generator: FC -> Reshape -> ConvTranspose2d blocks -> Tanh
   - Discriminator: Conv2d blocks -> Flatten -> Linear output
   - Feature maps grow in generator, shrink in discriminator
   - Typical progression: 1 -> 64 -> 128 -> 256 -> 512 in higher-resolution networks

HYPERPARAMETER SENSITIVITY:
   - Still sensitive to learning rate (typically 0.0002)
   - Beta1 = 0.5 works better than 0.9 (momentum is important)
   - Batch size: 128 recommended (larger batches = more stability)
   - Avoid high learning rates (> 0.001 often causes divergence)

CONVERGENCE ISSUES TO WATCH:
   - Mode collapse: Check diversity of generated samples
   - Training collapse: One network dominates the other
   - Diverging losses: Sudden spikes in loss values
   - Checkerboard artifacts: Sign of stride misalignment or improper weight init
"""

import numpy as np
from typing import Tuple, Optional
from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.conv.conv2d import Conv2d
from python.nn_core.normalization.batchnorm import BatchNorm2d
from python.nn_core.activations.relu import ReLU, LeakyReLU


class DCGANGenerator(Module):
    """
    DCGAN Generator with fractional-strided (transposed) convolutions.

    Architecture for 64x64 RGB images:
        Input: latent vector z (batch, 100)
        -> FC: (batch, 100) -> (batch, 512*4*4)
        -> Reshape: (batch, 512, 4, 4)
        -> ConvTranspose2d: (512, 4, 4) -> (256, 8, 8), kernel=4, stride=2, padding=1
        -> ConvTranspose2d: (256, 8, 8) -> (128, 16, 16), kernel=4, stride=2, padding=1
        -> ConvTranspose2d: (128, 16, 16) -> (64, 32, 32), kernel=4, stride=2, padding=1
        -> ConvTranspose2d: (64, 32, 32) -> (3, 64, 64), kernel=4, stride=2, padding=1
        -> Output: Tanh activation for [-1, 1] range

    Key Design Choices:
    - ConvTranspose2d with stride=2 for upsampling (learned upsampling)
    - BatchNorm2d after each layer (including latent FC expansion)
    - ReLU activation throughout (except final Tanh)
    - Tanh output to match normalized image range [-1, 1]
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
            latent_dim: Dimension of input noise vector (typically 100)
            feature_maps: Number of feature maps in initial layer (typically 64)
            img_channels: Number of output image channels (1 for grayscale, 3 for RGB)
            img_size: Output image size (assumes square images: 32, 64, 128, etc.)
        """
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.img_size = img_size

        # Number of layers depends on image size
        # For 64x64: 4 ConvTranspose2d layers (after initial FC)
        # For 128x128: 5 ConvTranspose2d layers, etc.

        # TODO: Implement generator architecture
        # Hint: Use nn.Sequential with:
        #   1. nn.Linear: latent_dim -> feature_maps * 8 * 4 * 4
        #      (for 64x64: 512*4*4 = 8192)
        #   2. nn.BatchNorm1d
        #   3. nn.ReLU
        #   4. Reshape to (batch, feature_maps*8, 4, 4)
        #   5. ConvTranspose2d blocks with kernel=4, stride=2, padding=1:
        #      - (512, 4, 4) -> (256, 8, 8)
        #      - (256, 8, 8) -> (128, 16, 16)
        #      - (128, 16, 16) -> (64, 32, 32)
        #      - (64, 32, 32) -> (3, 64, 64)
        #   6. Include BatchNorm2d and ReLU after each ConvTranspose2d
        #   7. Final layer: ConvTranspose2d with Tanh activation (no batch norm)
        raise NotImplementedError(
            "DCGAN Generator architecture not implemented. "
            "Hint: Stack ConvTranspose2d layers with BatchNorm2d and ReLU. "
            "Use stride=2, padding=1 for proper upsampling. Final output should "
            "be passed through Tanh for [-1, 1] normalization."
        )

    def forward(self, z):
        """
        Args:
            z: Latent vector (batch_size, latent_dim)

        Returns:
            Generated image (batch_size, img_channels, img_size, img_size)
        """
        raise NotImplementedError("Forward pass not implemented")


class DCGANDiscriminator(Module):
    """
    DCGAN Discriminator with strided convolutions (no max pooling).

    Architecture for 64x64 RGB images:
        Input: Image (batch, 3, 64, 64)
        -> Conv2d: (3, 64, 64) -> (64, 32, 32), kernel=4, stride=2, padding=1
        -> Conv2d: (64, 32, 32) -> (128, 16, 16), kernel=4, stride=2, padding=1
        -> Conv2d: (128, 16, 16) -> (256, 8, 8), kernel=4, stride=2, padding=1
        -> Conv2d: (256, 8, 8) -> (512, 4, 4), kernel=4, stride=2, padding=1
        -> Flatten: (batch, 512*4*4)
        -> Linear: -> 1 (logit output)

    Key Design Choices:
    - Conv2d with stride=2 for downsampling (learned downsampling)
    - LeakyReLU(0.2) activation throughout (except output)
    - BatchNorm2d after Conv2d (but NOT on first layer, NOT on output layer)
    - No fully connected layers except final classification layer
    - Output: Single logit (no activation, will use BCEWithLogitsLoss)

    Why no BatchNorm on first layer?
        - Input distribution is not normalized (raw pixel values)
        - BatchNorm on first layer can cause training instability
        - Input-dependent features should not be normalized

    Why no BatchNorm on output layer?
        - Output should be unconstrained logit
        - BatchNorm would normalize the classification score
        - Can cause gradient flow issues
    """

    def __init__(
        self,
        feature_maps: int = 64,
        img_channels: int = 3,
        img_size: int = 64,
    ):
        """
        Args:
            feature_maps: Number of feature maps in first layer (typically 64)
            img_channels: Number of input image channels (1 for grayscale, 3 for RGB)
            img_size: Input image size (assumes square images)
        """
        super(DCGANDiscriminator, self).__init__()
        self.feature_maps = feature_maps
        self.img_size = img_size

        # TODO: Implement discriminator architecture
        # Hint: Use nn.Sequential with:
        #   1. Conv2d: (img_channels, 64, 64) -> (64, 32, 32)
        #      kernel=4, stride=2, padding=1
        #      NO BatchNorm on first layer
        #   2. LeakyReLU(0.2)
        #   3. Conv2d blocks (with BatchNorm after each):
        #      - (64, 32, 32) -> (128, 16, 16)
        #      - (128, 16, 16) -> (256, 8, 8)
        #      - (256, 8, 8) -> (512, 4, 4)
        #   4. After each Conv2d (except first): BatchNorm2d + LeakyReLU(0.2)
        #   5. Flatten
        #   6. Linear: (512*4*4) -> 1
        #      NO activation on output layer
        raise NotImplementedError(
            "DCGAN Discriminator architecture not implemented. "
            "Hint: Use Conv2d with stride=2 for downsampling, LeakyReLU(0.2) "
            "activation, and BatchNorm2d after each conv layer (except first and "
            "last). Output should be a single logit with no activation."
        )

    def forward(self, x):
        """
        Args:
            x: Image batch (batch_size, img_channels, img_size, img_size)

        Returns:
            Classification logit (batch_size, 1)
        """
        raise NotImplementedError("Forward pass not implemented")


class DCGAN:
    """
    DCGAN trainer implementing stable adversarial training.

    Improvements over Vanilla GAN:
    1. Architectural guidelines for stability
    2. Better gradient flow via strided convolutions
    3. Batch normalization for reduced internal covariate shift
    4. More stable convergence behavior
    """

    def __init__(
        self,
        generator: DCGANGenerator,
        discriminator: DCGANDiscriminator,
        latent_dim: int = 100,
        device: str = "cpu",
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        """
        Args:
            generator: DCGAN Generator
            discriminator: DCGAN Discriminator
            latent_dim: Latent dimension
            device: Device to train on
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            beta1: Beta1 for Adam (0.5 typically works better than 0.9)
            beta2: Beta2 for Adam
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        # TODO: Initialize optimizers with specified betas
        raise NotImplementedError(
            "Optimizer initialization not implemented. "
            "Hint: Create Adam-like optimizers with specified betas."
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
        Single DCGAN training iteration.

        Args:
            real_data: Batch of real images
            n_critic: Number of discriminator updates per generator update

        Returns:
            (d_loss, g_loss)

        DCGAN Training Specifics:
        - Similar to Vanilla GAN but with better gradient flow
        - Architectural improvements make training more stable
        - Typically converges faster than Vanilla GAN
        - Mode collapse still possible but less severe
        """
        batch_size = real_data.shape[0]

        # Labels
        real_label = 0.9  # One-sided label smoothing
        fake_label = 0.0

        # TODO: Implement training step (similar to Vanilla GAN but with D/G architecture)
        raise NotImplementedError(
            "Training step not implemented. "
            "Hint: Follow the same adversarial training loop as Vanilla GAN."
        )

    def generate_samples(self, num_samples: int = 16):
        """Generate samples from the generator."""
        self.generator.eval()
        z = self.sample_noise(num_samples)
        fake_data = self.generator(z)
        self.generator.train()
        return fake_data

    def save_checkpoint(self, path: str):
        """Save generator and discriminator weights."""
        raise NotImplementedError(
            "Checkpoint saving not implemented. "
            "Hint: Save state_dicts for both networks."
        )

    def load_checkpoint(self, path: str):
        """Load generator and discriminator weights."""
        raise NotImplementedError(
            "Checkpoint loading not implemented."
        )


def train_dcgan(
    generator: DCGANGenerator,
    discriminator: DCGANDiscriminator,
    train_loader,
    num_epochs: int = 30,
    latent_dim: int = 100,
) -> Tuple[list, list]:
    """
    Training loop for DCGAN.

    Args:
        generator: DCGAN Generator
        discriminator: DCGAN Discriminator
        train_loader: DataLoader for training data
        num_epochs: Number of epochs to train
        device: Device to train on
        latent_dim: Latent dimension

    Returns:
        (g_losses, d_losses): Loss lists

    DCGAN Training Benefits:
    - More stable convergence than Vanilla GAN
    - Fewer training steps needed
    - Better sample quality
    - Architectural guidelines ensure consistent behavior
    """
    gan = DCGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=latent_dim,
    )

    g_losses = []
    d_losses = []

    # TODO: Implement training loop
    raise NotImplementedError(
        "Training loop not implemented. "
        "Hint: Similar to Vanilla GAN training loop."
    )

    return g_losses, d_losses


# DCGAN ARCHITECTURAL SUMMARY:
# ============================
#
# Generator:
#   Input: z ~ N(0, I)
#   1. Linear: 100 -> 512*4*4
#   2. Reshape: -> (512, 4, 4)
#   3. ConvTranspose2d blocks: upsampling from 4x4 to 64x64
#      - Each block: Conv2d, BatchNorm2d, ReLU
#      - Stride=2, padding=1 for proper upsampling
#   4. Output: Tanh for [-1, 1] normalization
#
# Discriminator:
#   Input: x (3, 64, 64) or fake (3, 64, 64)
#   1. Conv2d blocks: downsampling from 64x64 to 4x4
#      - First layer: NO BatchNorm
#      - Other layers: Conv2d, BatchNorm2d, LeakyReLU(0.2)
#      - Stride=2, padding=1 for downsampling
#   2. Flatten to (512*4*4,)
#   3. Output: Linear to 1 logit (no activation)
#
# Key Improvements:
#   - Learned upsampling (ConvTranspose2d) instead of bilinear
#   - Learned downsampling (strided Conv2d) instead of max pooling
#   - Batch normalization stabilizes training
#   - Better gradient flow enables deeper networks
#   - Reduced mode collapse compared to Vanilla GAN
#
# Hyperparameter Notes:
#   - Beta1=0.5 works better than 0.9 (less momentum is better)
#   - Learning rate 0.0002 standard
#   - Batch size typically 128 or larger
#   - Converges in ~30 epochs on CIFAR-10 (vs ~50 for Vanilla GAN)
