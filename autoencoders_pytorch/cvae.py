"""
Conditional Variational Autoencoder (CVAE) in PyTorch.

CVAE extends VAE by conditioning on additional information (e.g., class labels).
Both encoder and decoder receive the condition as input.

Architecture:
    Encoder: q(z|x, c) - encode x given condition c
    Decoder: p(x|z, c) - decode z given condition c

This enables:
    1. Conditional generation: generate samples of a specific class
    2. Attribute manipulation: change attributes while preserving others
    3. Class interpolation: smoothly transition between classes

Applications:
    - Generate specific digits (condition on digit class)
    - Generate faces with specific attributes (condition on age, gender, etc.)
    - Goal-conditioned policies in RL (condition on goal state)

Reference: Sohn et al., "Learning Structured Output Representation using
            Deep Conditional Generative Models" (NeurIPS 2015)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders_pytorch.vae import kl_divergence, reparameterize


def one_hot_encode(labels, num_classes):
    """
    Convert integer labels to one-hot encoding.

    Parameters:
        labels: Tensor of shape (batch,) - Integer class labels
        num_classes: int - Total number of classes

    Returns:
        one_hot: Tensor of shape (batch, num_classes)
    """
    B = labels.size(0)
    one_hot = torch.zeros((B, num_classes))
    one_hot[torch.arange(B),labels] = 1
    return one_hot


class CVAEEncoder(nn.Module):
    """
    Conditional VAE Encoder: q(z|x, c).

    Concatenates input x with condition c before encoding.

    Architecture: Linear([x, c]) -> ReLU -> (Linear_mu, Linear_logvar)
    """

    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        """
        Parameters:
            input_dim: int - Input dimension
            condition_dim: int - Condition dimension (e.g., num_classes for one-hot)
            hidden_dim: int - Hidden layer dimension
            latent_dim: int - Latent space dimension
        """
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim + condition_dim, hidden_dim),
                                    nn.Linear(hidden_dim, latent_dim * 2),)
        self.latent_dim = latent_dim
        # TODO: Define layers

    def forward(self, x, c):
        """
        Encode input conditioned on c.

        Parameters:
            x: Tensor of shape (batch, input_dim)
            c: Tensor of shape (batch, condition_dim)

        Returns:
            mu: Tensor of shape (batch, latent_dim)
            logvar: Tensor of shape (batch, latent_dim)
        """
        params = self.layers(torch.concat([x, c], dim=-1))
        mu = params[:, :self.latent_dim]
        logvar = params[:, self.latent_dim:]
        return mu, logvar


class CVAEDecoder(nn.Module):
    """
    Conditional VAE Decoder: p(x|z, c).

    Concatenates latent z with condition c before decoding.

    Architecture: Linear([z, c]) -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        """
        Parameters:
            latent_dim: int - Latent dimension
            condition_dim: int - Condition dimension
            hidden_dim: int - Hidden layer dimension
            output_dim: int - Output dimension (same as input)
        """
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent_dim + condition_dim, hidden_dim),
                                    nn.Linear(hidden_dim, output_dim),)
        # TODO: Define layers

    def forward(self, z, c):
        """
        Decode latent conditioned on c.

        Parameters:
            z: Tensor of shape (batch, latent_dim)
            c: Tensor of shape (batch, condition_dim)

        Returns:
            x_recon: Tensor of shape (batch, output_dim)
        """
        x_recon = self.layers(torch.cat([z, c], dim=-1))
        return x_recon


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.

    Encodes and decodes conditioned on additional information.
    Enables conditional generation and attribute manipulation.
    """

    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = CVAEEncoder(input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = CVAEDecoder(latent_dim, condition_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

    def forward(self, x, c):
        """
        Full forward pass.

        Parameters:
            x: Tensor of shape (batch, input_dim)
            c: Tensor of shape (batch, condition_dim)

        Returns:
            x_recon, mu, logvar, z
        """
        mu, logvar = self.encoder(x, c)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar, z

    def compute_loss(self, x, x_recon, mu, logvar):
        """
        Compute CVAE loss (same as VAE: reconstruction + KL).

        Returns:
            total_loss, recon_loss, kl_loss
        """
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = kl_divergence(mu, logvar)
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, c):
        """
        Generate samples conditioned on c.

        Parameters:
            c: Tensor of shape (n_samples, condition_dim)

        Returns:
            samples: Tensor of shape (n_samples, input_dim)
        """
        B = c.shape[0]
        z = torch.randn(B, self.latent_dim)
        samples = self.decoder(z, c)
        return samples

    @torch.no_grad()
    def sample_class(self, class_label, num_classes, n_samples=1, device="cpu"):
        """
        Generate samples of a specific class.

        Parameters:
            class_label: int - Class to generate
            num_classes: int - Total number of classes
            n_samples: int - Number of samples

        Returns:
            samples: Tensor of shape (n_samples, input_dim)
        """
        c = one_hot_encode(torch.tensor([class_label]), num_classes)
        c = c.repeat(n_samples, 1)
        samples = self.sample(c)
        return samples

    @torch.no_grad()
    def reconstruct(self, x, c):
        """Reconstruct with condition (uses mean, not sampled z)."""
        mu, _ = self.encoder(x, c)
        return self.decoder(mu, c)

    @torch.no_grad()
    def interpolate_class(self, x, c1, c2, n_steps=10):
        """
        Interpolate between two conditions for the same input.

        Encodes x with c1, then decodes with interpolated conditions.
        Useful for smooth attribute transitions.

        Parameters:
            x: Tensor of shape (1, input_dim)
            c1, c2: Tensor of shape (1, condition_dim) - Start/end conditions
            n_steps: int

        Returns:
            interpolations: Tensor of shape (n_steps, input_dim)
        """
        steps = torch.linspace(0, 1, n_steps).unsqueeze(1)
        c = steps @ c1 + (1 - steps) @ c2
        mu, logvar = self.encoder(x, c1)
        z = reparameterize(mu, logvar)
        z = z.repeat(n_steps, 1)
        interpolations = self.decoder(z, c)
        return interpolations


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, input_dim, num_classes, hidden_dim, latent_dim = 32, 784, 10, 256, 20

    cvae = CVAE(input_dim, num_classes, hidden_dim, latent_dim)

    x = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    c = one_hot_encode(labels, num_classes)

    x_recon, mu, logvar, z = cvae(x, c)
    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {c.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")

    samples = cvae.sample_class(5, num_classes, n_samples=10)
    print(f"Generated samples shape: {samples.shape}")