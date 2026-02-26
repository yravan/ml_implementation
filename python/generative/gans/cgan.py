"""
Conditional GAN (CGAN)

Original paper: "Conditional Generative Adversarial Nets" (Mirza & Osindski, 2014)
https://arxiv.org/abs/1411.1784

CGAN extends the GAN framework to generate samples conditioned on auxiliary information.
This enables controllable generation of specific classes, attributes, or styles.

THEORY:
--------
Vanilla GAN generates unconditional samples from p_g(x). CGAN generates conditional
samples from p_g(x|y) where y is the conditioning variable (e.g., class label).

Extension of adversarial objective:

    max_D min_G V(D, G) = E_{x~p_data, y}[log D(x|y)] + E_{z~p(z), y}[log(1 - D(G(z|y)|y))]

Key Changes:
1. Generator: Takes both noise z AND condition y as input
   G(z, y) instead of G(z)

2. Discriminator: Takes both sample AND condition as input
   D(x, y) instead of D(x)
   - Determines: Is this real data from class y? Or fake?

3. Condition Integration Methods:
   a) Concatenation: Concatenate y to input (simple but not always effective)
   b) Fully-connected projection: FC(y) then concatenate
   c) Embedding: Learned embedding e(y) then concatenate
   d) Residual addition: Add projected condition to intermediate features
   e) Feature-wise linear modulation (FiLM): Scale and shift features

ADVANTAGES:
- Controllable generation (generate specific classes or attributes)
- Better mode coverage (each mode can be controlled separately)
- Enables class-specific Inception Scores (IS per class)
- Natural application to image-to-image translation precursor

TRAINING DYNAMICS:
- Condition provides additional discriminator signal
- Generator gets clearer gradient (harder to fool discriminator)
- Mode collapse less severe (each class is a separate mode)
- Training typically more stable than unconditional GAN

ARCHITECTURAL CONSIDERATIONS:
1. Condition must be informative but not make problem trivial
   - Too weak: Condition is ignored
   - Too strong: Discriminator always knows the condition is fake
   - Balance: Condition should help discriminator but not fully determine realness

2. Embedding vs One-Hot:
   - One-hot: Good for discrete classes (10 classes MNIST)
   - Embedding: Better for continuous or high-dimensional conditions
   - Embedding allows sharing of condition information across dimensions

3. Concatenation Points:
   - Generator: After first FC layer or spatial reshape
   - Discriminator: After first conv layer (mix with input features)

CHALLENGES:
- Discriminator overfits to condition (ignores image features)
  - Remedy: Use projection discriminator or drop condition occasionally
- Mode collapse within classes (subset of class generated)
  - Remedy: Larger batch sizes, better optimization
- Unbalanced class representation in training data
  - Remedy: Class-balanced sampling or class weights

APPLICATIONS:
- Class-conditional image generation (MNIST digits 0-9)
- Attribute-based generation (hair color, age, gender)
- Image inpainting (condition on known pixels)
- Style transfer (condition on style image)
- Text-to-image synthesis (condition on text description)
"""

import numpy as np
from typing import Tuple, Optional
from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.conv.conv2d import Conv2d
from python.nn_core.normalization.batchnorm import BatchNorm2d
from python.nn_core.activations.relu import ReLU, LeakyReLU


class ConditionalGenerator(Module):
    """
    Conditional Generator that generates samples conditioned on auxiliary information.

    Architecture:
        Input: noise z (batch, latent_dim) + condition y (batch, num_classes)
        -> Concatenate z and embedded y
        -> FC layers with ReLU
        -> Reshape to spatial dimensions
        -> ConvTranspose2d blocks (like DCGAN)
        -> Output: Generated image conditioned on y

    Condition Integration:
        1. Simple concatenation: Direct concat of z and one-hot y
        2. Embedding: Learned embedding of condition (better for continuous conditions)
        3. Projected concatenation: FC(y) -> concat with z

    For MNIST:
        - latent_dim=100
        - num_classes=10
        - Condition: One-hot (10) or embedded (e.g., 50-dim)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        feature_maps: int = 64,
        img_channels: int = 1,
        img_size: int = 28,
        embedding_dim: int = 50,
    ):
        """
        Args:
            latent_dim: Dimension of noise vector
            num_classes: Number of classes (for condition)
            feature_maps: Number of feature maps in first layer
            img_channels: Number of output channels
            img_size: Output image size
            embedding_dim: Dimension of class embedding
        """
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.embedding_dim = embedding_dim

        # TODO: Implement conditional generator
        # Hint: Architecture structure:
        #   1. Class embedding layer: nn.Embedding(num_classes, embedding_dim)
        #   2. Concatenate z (latent_dim) with embedded y (embedding_dim)
        #   3. FC layer: (latent_dim + embedding_dim) -> 256
        #   4. BatchNorm1d and ReLU
        #   5. FC layer: 256 -> 512 * 4 * 4 (for 28x28 output)
        #   6. BatchNorm1d and ReLU
        #   7. Reshape to (512, 4, 4)
        #   8. ConvTranspose2d blocks (similar to DCGAN):
        #      - (512, 4, 4) -> (256, 7, 7), kernel=4, stride=2, padding=1
        #      - (256, 7, 7) -> (128, 14, 14), kernel=4, stride=2, padding=1
        #      - (128, 14, 14) -> (1, 28, 28), kernel=4, stride=2, padding=1
        #   9. Tanh activation on output
        raise NotImplementedError(
            "Conditional Generator not implemented. "
            "Hint: Create embedding for condition, concatenate with noise, "
            "expand through FC layers, reshape, then use ConvTranspose2d blocks."
        )

    def forward(self, z, y):
        """
        Args:
            z: Noise vector (batch_size, latent_dim)
            y: Class label (batch_size,) or one-hot (batch_size, num_classes)

        Returns:
            Generated image (batch_size, img_channels, img_size, img_size)
        """
        raise NotImplementedError("Forward pass not implemented")


class ConditionalDiscriminator(Module):
    """
    Conditional Discriminator that classifies realness given both sample and condition.

    Architecture:
        Input: image x (batch, channels, h, w) + condition y (batch, num_classes)
        -> Conv2d layers to extract features
        -> Concatenate features with embedded condition
        -> FC layer to output logit
        -> Output: Realness score conditioned on y

    Condition Integration at Multiple Levels:
        1. Early fusion: Concatenate at input (not typical for images)
        2. Intermediate fusion: Concatenate after first few conv layers
        3. Late fusion: Concatenate with flattened features before FC
        4. Feature-wise modulation: Condition modulates intermediate features

    For this implementation: Intermediate fusion (after initial feature extraction)
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_maps: int = 64,
        img_channels: int = 1,
        img_size: int = 28,
        embedding_dim: int = 50,
    ):
        """
        Args:
            num_classes: Number of classes
            feature_maps: Feature maps in first conv layer
            img_channels: Number of input channels
            img_size: Input image size
            embedding_dim: Dimension of class embedding
        """
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.embedding_dim = embedding_dim

        # TODO: Implement conditional discriminator
        # Hint: Architecture structure:
        #   1. Class embedding: nn.Embedding(num_classes, embedding_dim)
        #   2. Conv layer: img_channels -> feature_maps
        #   3. LeakyReLU(0.2)
        #   4. Conv layer: feature_maps -> 2*feature_maps, stride=2
        #   5. LeakyReLU(0.2)
        #   6. Flatten features
        #   7. Concatenate flattened features with embedded condition
        #   8. FC layer: (flattened + embedding_dim) -> 256
        #   9. LeakyReLU(0.2)
        #  10. FC layer: 256 -> 1 (logit output, no activation)
        raise NotImplementedError(
            "Conditional Discriminator not implemented. "
            "Hint: Extract image features with conv layers, concatenate with "
            "embedded condition, then output logit via FC layer."
        )

    def forward(self, x, y):
        """
        Args:
            x: Image batch (batch_size, img_channels, img_size, img_size)
            y: Class label (batch_size,) or one-hot (batch_size, num_classes)

        Returns:
            Realness logit (batch_size, 1)
        """
        raise NotImplementedError("Forward pass not implemented")


class CGAN:
    """
    Conditional GAN trainer for class-conditional image generation.
    """

    def __init__(
        self,
        generator: ConditionalGenerator,
        discriminator: ConditionalDiscriminator,
        latent_dim: int = 100,
        num_classes: int = 10,
        device: str = "cpu",
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        beta1: float = 0.5,
    ):
        """
        Args:
            generator: Conditional Generator
            discriminator: Conditional Discriminator
            latent_dim: Latent dimension
            num_classes: Number of classes
            device: Device to train on
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            beta1: Beta1 for Adam optimizer
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # TODO: Initialize optimizers
        raise NotImplementedError(
            "Optimizer initialization not implemented. "
            "Hint: Create optimizers for G and D."
        )

    def sample_noise(self, batch_size: int):
        """Sample latent vectors."""
        return np.random.randn(batch_size, self.latent_dim).astype(np.float32)

    def sample_classes(self, batch_size: int):
        """Sample random class labels."""
        return np.random.randint(0, self.num_classes, (batch_size,))

    def train_step(
        self,
        real_data,
        real_labels,
        n_critic: int = 1,
    ) -> Tuple[float, float]:
        """
        Single training iteration for CGAN.

        Args:
            real_data: Batch of real images (batch_size, channels, h, w)
            real_labels: Batch of class labels (batch_size,)
            n_critic: Number of discriminator updates per generator update

        Returns:
            (d_loss, g_loss)

        CGAN Training Algorithm:
        -------------------------
        1. Discriminator Step (n_critic times):
           a. Sample real batch with labels from dataset
           b. Compute D loss on (real_data, real_labels) -> should output high
           c. Sample z and y (random classes)
           d. Generate fake: G(z, y)
           e. Compute D loss on (fake, y) -> should output low
           f. Total D loss = real loss + fake loss
           g. Backprop and update D

        2. Generator Step:
           a. Sample z and y
           b. Generate fake: G(z, y)
           c. Compute D(fake, y)
           d. G loss = BCE(D(fake, y), 1) [non-saturating: use real_label]
           e. Backprop and update G

        Key Differences from Vanilla GAN:
        - Must pass condition y to both G and D
        - Labels y must be sampled or provided from dataset
        - Training typically more stable due to condition signal
        """
        batch_size = real_data.shape[0]

        real_label = 0.9
        fake_label = 0.0

        # TODO: Implement discriminator training
        # For each critic iteration:
        #   1. Zero D gradients
        #   2. Forward real data: D(real_data, real_labels)
        #   3. Compute real loss
        #   4. Sample z, y_fake
        #   5. Generate fake: G(z, y_fake)
        #   6. Forward fake: D(fake.detach(), y_fake)
        #   7. Compute fake loss
        #   8. Total D loss and backprop
        raise NotImplementedError(
            "Discriminator training step not implemented. "
            "Hint: Similar to Vanilla GAN but pass condition y to D."
        )

        # TODO: Implement generator training
        # 1. Zero G gradients
        # 2. Sample z, y_fake
        # 3. Generate fake: G(z, y_fake)
        # 4. Forward through D: D(fake, y_fake)
        # 5. G loss = BCE(D(fake), real_label)
        # 6. Backprop and update G
        raise NotImplementedError(
            "Generator training step not implemented. "
            "Hint: Sample condition, generate with condition, "
            "compute loss with non-saturating objective."
        )

    def generate_class(self, class_idx: int, num_samples: int = 16):
        """
        Generate samples from a specific class.

        Args:
            class_idx: Class index to generate
            num_samples: Number of samples to generate

        Returns:
            Generated samples all from class_idx
        """
        self.generator.eval()
        z = self.sample_noise(num_samples)
        y = np.full((num_samples,), class_idx, dtype=np.long)
        fake_data = self.generator(z, y)
        self.generator.train()
        return fake_data

    def generate_samples(self, num_samples: int = 16):
        """
        Generate samples from all classes.

        Returns:
            (samples, labels) where labels are the class indices
        """
        self.generator.eval()
        z = self.sample_noise(num_samples)
        y = self.sample_classes(num_samples)
        fake_data = self.generator(z, y)
        self.generator.train()
        return fake_data, y

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        raise NotImplementedError("Checkpoint saving not implemented.")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        raise NotImplementedError("Checkpoint loading not implemented.")


def train_cgan(
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    train_loader,
    num_epochs: int = 50,
    latent_dim: int = 100,
    num_classes: int = 10,
) -> Tuple[list, list]:
    """
    Training loop for CGAN.

    Args:
        generator: Conditional Generator
        discriminator: Conditional Discriminator
        train_loader: DataLoader providing (images, labels)
        num_epochs: Number of training epochs
        device: Device to train on
        latent_dim: Latent dimension
        num_classes: Number of classes

    Returns:
        (g_losses, d_losses): Loss lists

    CGAN Training Benefits:
    - Per-class generation control
    - Better mode coverage (each class is separate)
    - More stable training than unconditional GAN
    - Enables class-conditional Inception Score
    """
    gan = CGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=latent_dim,
        num_classes=num_classes,
    )

    g_losses = []
    d_losses = []

    # TODO: Implement training loop
    # For each epoch:
    #   Iterate through (images, labels) in train_loader
    #   Call gan.train_step(images, labels)
    #   Accumulate losses
    #   Log epoch averages
    raise NotImplementedError(
        "Training loop not implemented. "
        "Hint: Similar to Vanilla GAN but train_loader provides (x, y) tuples."
    )

    return g_losses, d_losses


# MATHEMATICAL FORMULATION:
# =========================
#
# Value Function (Conditioned):
#   V(D, G) = E_x,y[log D(x|y)] + E_z,y[log(1 - D(G(z|y)|y))]
#
# Generator: G: (z, y) -> x_fake
#   min_G E_z,y[log(1 - D(G(z|y)|y))]
#   (Non-saturating: max_G E_z,y[log D(G(z|y)|y)])
#
# Discriminator: D: (x, y) -> [0,1]
#   max_D E_x,y[log D(x|y)] + E_z,y[log(1 - D(G(z|y)|y))]
#
# Condition Integration:
#   1. Generator: Concatenate embedded y with z
#   2. Discriminator: Concatenate embedded y with image features
#
# Expected Improvements:
#   - Mode collapse reduced (each class is separate mode)
#   - Training stability improved
#   - Generator gradients stronger (condition helps discriminator)
#   - Inception Score per class possible
#
# Architectural Patterns:
#   - Embedding layer: nn.Embedding(num_classes, embedding_dim)
#   - Concatenation: torch.cat([z, embedded_y], dim=1)
#   - Typical: embedding_dim = 50-100, latent_dim = 100
#
# Training Dynamics:
#   - Similar to Vanilla GAN but with condition signal
#   - Condition prevents discriminator collapse
#   - More stable convergence
#   - Fewer mode collapse issues
