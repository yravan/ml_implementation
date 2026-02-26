"""
CycleGAN (Unpaired Image-to-Image Translation)

Original paper: "Unpaired Image-to-Image Translation using Cycle-Consistent
Generative Adversarial Networks" (Zhu et al., 2017)
https://arxiv.org/abs/1703.10593

CycleGAN enables image-to-image translation WITHOUT paired training data.
Instead of requiring input-output pairs, it uses two unpaired image collections
and enforces cycle consistency: X → Y → X should recover original X.

MOTIVATION:
-----------
Pix2Pix requires paired training data (input-output pairs), which is:
- Expensive to collect (often requires manual alignment)
- Unavailable for many domains
- Difficult to scale to large datasets

CycleGAN solves this with cycle consistency loss, enabling translation between
unpaired image collections. Key innovation: F(G(x)) ≈ x (cycle reconstruction)

THEORY:
--------
Two generators:
  G: X → Y (maps from domain X to domain Y)
  F: Y → X (maps from domain Y to domain X)

Two discriminators:
  D_X: Discriminates real X vs fake X (from F)
  D_Y: Discriminates real Y vs fake Y (from G)

Three loss components:

1. ADVERSARIAL LOSS (for each generator-discriminator pair):
   L_GAN(G, D_Y, X, Y) = E_y[log D_Y(y)] + E_x[log(1 - D_Y(G(x)))]
   L_GAN(F, D_X, Y, X) = E_x[log D_X(x)] + E_y[log(1 - D_X(F(y)))]

   Ensures generated images look realistic in target domain.

2. CYCLE CONSISTENCY LOSS (key innovation):
   L_cyc = E_x[||F(G(x)) - x||_1] + E_y[||G(F(y)) - y||_1]

   Forces forward and backward mappings to be inverses:
   - Forward cycle: x → G(x) → F(G(x)) ≈ x (reconstruction)
   - Backward cycle: y → F(y) → G(F(y)) ≈ y (reconstruction)

   This prevents:
   - Mode collapse (both generators mapping to same image)
   - Meaningless translations (G ignoring input, F mapping back randomly)
   - Complete information loss

3. IDENTITY LOSS (optional, for preserving colors):
   L_identity = E_x[||G(x) - x||_1] + E_y[||F(y) - y||_1]

   For style transfer (color/texture), we want G and F to preserve images
   of their target domain (e.g., photo should stay photo).

   Useful for domain-specific attributes but can reduce translation capability.

COMBINED LOSS:
  L_total = L_GAN(G, D_Y) + L_GAN(F, D_X) + λ*L_cyc [+ γ*L_identity]

  Typical weights: λ = 10, γ = 0.5 (if using identity loss)

WHY CYCLE CONSISTENCY WORKS:
  - Without cycle loss: G could be arbitrary mapping, F could ignore input
  - With cycle loss: Meaningful structure must be preserved through cycles
  - L1 loss on cycle reconstruction enforces pixel-level fidelity
  - No paired data needed: Only need two unpaired image collections

ADVANTAGES:
  1. No paired training data required
  2. Works for many unpaired translation tasks
  3. Cycle consistency prevents mode collapse
  4. Full information preservation (original recoverable)
  5. Scalable to unpaired datasets (easier to collect)

DISADVANTAGES:
  1. Less direct control over specific attributes
  2. May produce less sharp outputs than pix2pix
  3. Requires careful λ tuning
  4. Training can be unstable (4 networks instead of 2)
  5. Cannot handle extreme domain shifts

APPLICATIONS:
  - Style transfer (monet → photo)
  - Season transfer (summer → winter)
  - Object transfiguration (horse → zebra)
  - Domain adaptation (photo ↔ painting)
  - Artistic style (realistic → cartoon)
  - Photo enhancement (low-light → normal)

ARCHITECTURE:
  Generators: ResNet-based (9-block residual blocks)
    - Preserve structure through residual connections
    - Symmetric encoder-decoder with skip connections
    - More stable than U-Net for unpaired translation

  Discriminators: PatchGAN (same as pix2pix)
    - Discriminates local patches
    - Focuses on texture and local structure

TRAINING DYNAMICS:
  - Four networks (2 generators, 2 discriminators) make training complex
  - Must balance multiple loss terms (adversarial + cycle consistency)
  - Cycle consistency loss dominates training
  - Oscillating loss values possible (stable training harder)

CONVERGENCE:
  - Typically 100-200 epochs for convergence
  - Loss curves less smooth than WGAN-GP
  - Requires careful hyperparameter tuning
  - Works best with batch size 1 (original paper)

COMPARISON TO PIX2PIX:
  ┌─────────────────────────────────────────────────────┐
  │              Pix2Pix      │      CycleGAN            │
  ├─────────────────────────────────────────────────────┤
  │ Data           Paired     │ Unpaired                 │
  │ Quality        Sharp      │ Slightly blurry          │
  │ Stability      Stable     │ Less stable              │
  │ Convergence    ~30-50 ep  │ ~100-200 epochs          │
  │ Loss           L_GAN+L_L1 │ L_GAN+λ*L_cyc            │
  │ Control        Precise    │ Less precise             │
  └─────────────────────────────────────────────────────┘

DATA COLLECTION:
  CycleGAN only needs:
  1. Unpaired images from domain X (no labels needed)
  2. Unpaired images from domain Y (no labels needed)
  3. No alignment or correspondence required

  Example:
    Domain X: 100 horse photos (any horses, any backgrounds)
    Domain Y: 100 zebra photos (any zebras, any backgrounds)
    Result: horse → zebra translation (and vice versa)

KEY HYPERPARAMETERS:
  1. λ (Cycle loss weight): Typically 10
     - Higher: Stronger cycle constraint, less translation freedom
     - Lower: More creative translation, less fidelity

  2. γ (Identity loss weight): Typically 0.5 (or 0)
     - Higher: Preserve colors, less transformation
     - 0: Disable identity loss

  3. Learning rate: 0.0002 (same as pix2pix)

  4. Batch size: 1 (original paper)

  5. Number of residual blocks: 9 for 256×256, 6 for 128×128

STABILITY CONSIDERATIONS:
  - Four networks can cause training instability
  - Cycle loss can conflict with adversarial loss
  - Requires good balance between G and D learning
  - Consider using learning rate scheduling

MATHEMATICAL INSIGHT:
  Cycle consistency creates a constraint that makes the problem well-defined
  even without paired data. Instead of learning a specific mapping, we learn
  mappings that preserve information through round-trip translation.

  This is elegant because:
  1. No paired labels needed (unsupervised)
  2. Information preservation guaranteed (cycle loss)
  3. Works across diverse domains
  4. Theoretically justified (cycle constraints reduce solution space)
"""

import numpy as np
from typing import Tuple, Optional
from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.conv.conv2d import Conv2d
from python.nn_core.normalization.batchnorm import BatchNorm2d
from python.nn_core.activations.relu import ReLU, LeakyReLU


class ResidualBlock(Module):
    """
    Residual block for CycleGAN generator.

    Preserves information through skip connections:
    Output = Input + Conv(Conv(Input))

    Benefits:
    - Enables deep networks without vanishing gradients
    - Preserves input information
    - Faster convergence
    """

    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input/output channels
        """
        super(ResidualBlock, self).__init__()

        # TODO: Implement residual block
        # Hint: Architecture pattern:
        #   1. Conv2d: channels -> channels, kernel=3, padding=1
        #   2. InstanceNorm2d (NOT BatchNorm for unpaired translation)
        #   3. ReLU
        #   4. Conv2d: channels -> channels, kernel=3, padding=1
        #   5. InstanceNorm2d
        #   6. In forward: return input + self.conv_layers(input)
        raise NotImplementedError(
            "Residual block not implemented. "
            "Hint: Two Conv-InstanceNorm-ReLU blocks with skip connection."
        )

    def forward(self, x):
        """Forward pass with residual connection."""
        raise NotImplementedError("Forward pass not implemented")


class CycleGANGenerator(Module):
    """
    CycleGAN generator with residual blocks.

    Architecture:
    1. Encoder: Conv layers downsampling input
    2. Residual blocks: 9 residual blocks at bottleneck
    3. Decoder: ConvTranspose layers upsampling output

    Uses InstanceNorm instead of BatchNorm (better for unpaired translation).

    Example (256×256 input):
    256x256 -> 128x128 -> 64x64 (encoder)
    64x64 (9 residual blocks)
    64x64 -> 128x128 -> 256x256 (decoder)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        num_residual_blocks: int = 9,
    ):
        """
        Args:
            in_channels: Input channels (typically 3)
            out_channels: Output channels (typically 3)
            num_filters: Base number of filters
            num_residual_blocks: Number of residual blocks (9 for 256×256)
        """
        super(CycleGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks

        # TODO: Implement CycleGAN generator
        # Hint: Architecture pattern:
        #   1. Initial Conv: in_channels -> num_filters, kernel=7, padding=3
        #      + InstanceNorm2d + ReLU
        #   2. Encoder blocks (downsampling with Conv stride=2):
        #      Block 1: num_filters -> 2*num_filters
        #      Block 2: 2*num_filters -> 4*num_filters
        #      Each: Conv stride=2, InstanceNorm, ReLU
        #   3. Residual blocks (no spatial change):
        #      9 blocks of ResidualBlock at (4*num_filters, 4*num_filters)
        #   4. Decoder blocks (upsampling with ConvTranspose stride=2):
        #      Block 1: 4*num_filters -> 2*num_filters
        #      Block 2: 2*num_filters -> num_filters
        #      Each: ConvTranspose stride=2, InstanceNorm, ReLU
        #   5. Final Conv: num_filters -> out_channels, kernel=7, padding=3
        #      + Tanh activation
        raise NotImplementedError(
            "CycleGAN Generator not implemented. "
            "Hint: Encoder -> Residual blocks -> Decoder, use InstanceNorm."
        )

    def forward(self, x):
        """
        Args:
            x: Input image (batch_size, in_channels, height, width)

        Returns:
            Translated image (batch_size, out_channels, height, width)
        """
        raise NotImplementedError("Forward pass not implemented")


class CycleGANDiscriminator(Module):
    """
    PatchGAN discriminator for CycleGAN (same as pix2pix).

    Discriminates N×N patches instead of whole image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 64,
    ):
        """
        Args:
            in_channels: Input channels (typically 3)
            num_filters: Base number of filters
        """
        super(CycleGANDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters

        # TODO: Implement discriminator
        # Hint: PatchGAN architecture:
        #   1. Conv2d: in_channels -> num_filters, kernel=4, stride=2, padding=1
        #      + LeakyReLU(0.2)
        #   2. Conv2d blocks (stride=2, doubling filters):
        #      num_filters -> 2*num_filters
        #      2*num_filters -> 4*num_filters
        #      Each: Conv stride=2, InstanceNorm, LeakyReLU(0.2)
        #   3. Conv2d: 4*num_filters -> 8*num_filters, kernel=4, stride=1, padding=1
        #      + InstanceNorm + LeakyReLU(0.2)
        #   4. Conv2d: 8*num_filters -> 1, kernel=4, stride=1, padding=1
        #      Output logit (no activation)
        raise NotImplementedError(
            "Discriminator not implemented. "
            "Hint: PatchGAN with InstanceNorm, output spatial logit map."
        )

    def forward(self, x):
        """
        Args:
            x: Image (batch_size, in_channels, height, width)

        Returns:
            Patch logits (batch_size, 1, h_out, w_out)
        """
        raise NotImplementedError("Forward pass not implemented")


class CycleGAN:
    """
    CycleGAN trainer for unpaired image-to-image translation.

    Combines two generators, two discriminators, and cycle consistency loss
    to enable translation between unpaired image domains.
    """

    def __init__(
        self,
        gen_x2y: CycleGANGenerator,
        gen_y2x: CycleGANGenerator,
        disc_x: CycleGANDiscriminator,
        disc_y: CycleGANDiscriminator,
        device: str = "cpu",
        lr: float = 0.0002,
        beta1: float = 0.5,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
    ):
        """
        Args:
            gen_x2y: Generator from X to Y
            gen_y2x: Generator from Y to X
            disc_x: Discriminator for X
            disc_y: Discriminator for Y
            device: Device to train on
            lr: Learning rate for all networks
            beta1: Adam beta1
            lambda_cycle: Cycle consistency loss weight (typical: 10)
            lambda_identity: Identity loss weight (typical: 0.5 or 0)
        """
        self.gen_x2y = gen_x2y
        self.gen_y2x = gen_y2x
        self.disc_x = disc_x
        self.disc_y = disc_y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        # TODO: Initialize optimizers for all 4 networks
        # Hint: Create optimizers for each network with specified lr and beta1
        raise NotImplementedError(
            "Optimizer initialization not implemented. "
            "Hint: Create optimizers for all 4 networks."
        )

    def train_step(
        self,
        real_x,
        real_y,
    ) -> Tuple[float, float, float]:
        """
        Single CycleGAN training iteration.

        Args:
            real_x: Batch of real images from domain X
            real_y: Batch of real images from domain Y

        Returns:
            (loss_gen, loss_disc, loss_cycle)

        CYCLEGAN TRAINING ALGORITHM:
        ============================

        1. GENERATOR STEP:
           a. Generate fake: y_fake = G_X2Y(x_real), x_fake = G_Y2X(y_real)
           b. Cycle reconstruction:
              x_cycle = G_Y2X(y_fake)
              y_cycle = G_X2Y(x_fake)
           c. Adversarial loss:
              L_GAN_X2Y = E[-log D_Y(y_fake)]
              L_GAN_Y2X = E[-log D_X(x_fake)]
           d. Cycle consistency loss:
              L_cyc = E[||x_cycle - x||_1] + E[||y_cycle - y||_1]
           e. Identity loss (optional):
              L_id = E[||G_X2Y(y) - y||_1] + E[||G_Y2X(x) - x||_1]
           f. Total: L_G = L_GAN_X2Y + L_GAN_Y2X + λ*L_cyc + γ*L_id
           g. Backprop and update both generators

        2. DISCRIMINATOR STEPS:
           a. Discriminator X:
              L_DX_real = E[-log D_X(x_real)]
              L_DX_fake = E[-log(1 - D_X(x_fake.detach()))]
              L_DX = L_DX_real + L_DX_fake
           b. Discriminator Y:
              L_DY_real = E[-log D_Y(y_real)]
              L_DY_fake = E[-log(1 - D_Y(y_fake.detach()))]
              L_DY = L_DY_real + L_DY_fake
           c. Backprop and update discriminators

        LOSS COMPONENTS:
        ===============
        Generator Loss (L_G):
          1. Adversarial: Make fakes look real to discriminators
          2. Cycle: Reconstruct original after round-trip
          3. Identity: Preserve if no translation needed

        Discriminator Loss (L_D):
          1. Real X: Recognize real domain X images
          2. Fake X: Recognize fake X images (from cycle)
          3. Same for domain Y

        CYCLE CONSISTENCY KEY:
        The cycle consistency loss prevents:
        - Mode collapse (all images map to same output)
        - Information loss (original must be recoverable)
        - Meaningless translation (must respect input structure)

        Without cycle loss, X2Y could ignore input and Y2X could map back randomly.
        """
        batch_size = real_x.shape[0]

        # TODO: Implement generator training step
        # 1. Generate fakes:
        #    fake_y = self.gen_x2y(real_x)
        #    fake_x = self.gen_y2x(real_y)
        # 2. Cycle reconstruction:
        #    cycle_x = self.gen_y2x(fake_y)
        #    cycle_y = self.gen_x2y(fake_x)
        # 3. Identity (optional):
        #    id_x = self.gen_y2x(real_x)
        #    id_y = self.gen_x2y(real_y)
        # 4. Compute adversarial losses, cycle loss, identity loss
        # 5. Total loss and backward
        raise NotImplementedError(
            "Generator training step not implemented. "
            "Hint: Generate fakes, compute cycle reconstruction, "
            "combine adversarial + cycle + identity losses."
        )

        # TODO: Implement discriminator training step
        # 1. Discriminate real X and fake X
        # 2. Discriminate real Y and fake Y
        # 3. Compute losses for both discriminators
        # 4. Backward and update
        raise NotImplementedError(
            "Discriminator training step not implemented. "
            "Hint: Distinguish real and fake for both domains."
        )

    def translate_x2y(self, x):
        """Translate from domain X to domain Y."""
        self.gen_x2y.eval()
        output = self.gen_x2y(x)
        self.gen_x2y.train()
        return output

    def translate_y2x(self, y):
        """Translate from domain Y to domain X."""
        self.gen_y2x.eval()
        output = self.gen_y2x(y)
        self.gen_y2x.train()
        return output

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        raise NotImplementedError("Checkpoint saving not implemented.")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        raise NotImplementedError("Checkpoint loading not implemented.")


def train_cyclegan(
    gen_x2y: CycleGANGenerator,
    gen_y2x: CycleGANGenerator,
    disc_x: CycleGANDiscriminator,
    disc_y: CycleGANDiscriminator,
    train_loader_x,
    train_loader_y,
    num_epochs: int = 200,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 0.5,
) -> Tuple[list, list, list]:
    """
    Training loop for CycleGAN.

    Args:
        gen_x2y: Generator X→Y
        gen_y2x: Generator Y→X
        disc_x: Discriminator X
        disc_y: Discriminator Y
        train_loader_x: DataLoader for domain X images
        train_loader_y: DataLoader for domain Y images
        num_epochs: Number of epochs
        device: Device to train on
        lambda_cycle: Cycle loss weight
        lambda_identity: Identity loss weight

    Returns:
        (gen_losses, disc_losses, cycle_losses): Loss lists

    CYCLEGAN TRAINING:
    ==================
    Key difference from pix2pix:
    - Unpaired datasets: Two separate dataloaders
    - Cycle consistency: Reconstruct original through round-trip
    - Four networks: 2 generators + 2 discriminators
    - More complex training dynamics

    CONVERGENCE:
    - Typically 100-200 epochs (longer than pix2pix)
    - Losses can oscillate more
    - Requires patience and careful monitoring

    TYPICAL RESULTS:
    - Style transfer (monet → photo)
    - Season transfer (summer ↔ winter)
    - Object transfiguration (horse → zebra)
    - Domain adaptation (photo ↔ sketch)
    """
    cyclegan = CycleGAN(
        gen_x2y=gen_x2y,
        gen_y2x=gen_y2x,
        disc_x=disc_x,
        disc_y=disc_y,
        lambda_cycle=lambda_cycle,
        lambda_identity=lambda_identity,
    )

    gen_losses = []
    disc_losses = []
    cycle_losses = []

    # TODO: Implement training loop
    # For each epoch:
    #   Iterate through batches from both dataloaders (zip or cycle)
    #   For each pair (x_batch, y_batch):
    #     gen_loss, disc_loss, cycle_loss = cyclegan.train_step(x_batch, y_batch)
    #     Accumulate losses
    #   Log epoch averages
    raise NotImplementedError(
        "Training loop not implemented. "
        "Hint: Iterate through paired batches from two dataloaders, "
        "call train_step, accumulate losses."
    )

    return gen_losses, disc_losses, cycle_losses


# LOSS FUNCTION SUMMARY:
# ======================
#
# TOTAL LOSS:
# L = L_GAN(G_X2Y) + L_GAN(G_Y2X) + L_GAN(D_X) + L_GAN(D_Y)
#     + λ * L_cyc(G_X2Y, G_Y2X)
#     + γ * L_id(G_X2Y, G_Y2X)
#
# ADVERSARIAL LOSS:
# L_GAN(G_X2Y, D_Y) = E_x[-log D_Y(G_X2Y(x))]
# L_GAN(G_Y2X, D_X) = E_y[-log D_X(G_Y2X(y))]
#
# L_GAN(D_Y) = E_y[-log D_Y(y)] + E_x[-log(1 - D_Y(G_X2Y(x)))]
# L_GAN(D_X) = E_x[-log D_X(x)] + E_y[-log(1 - D_X(G_Y2X(y)))]
#
# CYCLE CONSISTENCY LOSS:
# L_cyc = E_x[||G_Y2X(G_X2Y(x)) - x||_1]
#         + E_y[||G_X2Y(G_Y2X(y)) - y||_1]
#
# IDENTITY LOSS (Optional):
# L_id = E_x[||G_Y2X(x) - x||_1] + E_y[||G_X2Y(y) - y||_1]
#
# TYPICAL WEIGHTS:
# λ = 10 (cycle consistency heavily weighted)
# γ = 0.5 (identity loss, can be 0 to disable)
#
# INSTANCE NORMALIZATION:
# Used instead of BatchNorm for unpaired translation:
# - BatchNorm removes style information (uses batch statistics)
# - InstanceNorm preserves instance style (per-example normalization)
# - Better for style transfer applications
#
# RESIDUAL BLOCKS:
# Preserve information through skip connections:
# x_out = x_in + f(x_in)
# Enables deep networks without vanishing gradients
# Typically 9 blocks for 256×256 images
#
# UNPAIRED DATA ADVANTAGE:
# - No need for aligned pairs
# - Easier to collect large datasets
# - Applicable to many real-world scenarios
# - No manual annotation required
#
# TRAINING CHALLENGES:
# - Four networks make training complex
# - Mode collapse possible despite cycle loss
# - Oscillating losses (need careful monitoring)
# - Longer convergence time (100-200 epochs)
# - Hyperparameter tuning critical (λ and γ)
