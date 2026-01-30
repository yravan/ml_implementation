"""
Variational Autoencoder (VAE) in PyTorch.

Unlike standard autoencoders, VAEs learn a probabilistic latent space.
The encoder outputs parameters (mu, sigma) of a Gaussian distribution,
and we sample from it using the reparameterization trick.

Key Components:
    1. Encoder outputs mu and log(sigma^2) for the latent distribution
    2. Reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I)
    3. Loss = Reconstruction Loss + KL Divergence

ELBO (Evidence Lower Bound):
    L = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
      = Reconstruction           - Regularization

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(mu, logvar):
    """
    KL divergence between q(z|x) = N(mu, sigma^2) and p(z) = N(0, I).

    Closed-form solution:
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters:
        mu: Tensor of shape (batch, latent_dim) - Mean of q(z|x)
        logvar: Tensor of shape (batch, latent_dim) - Log variance of q(z|x)

    Returns:
        kl: Tensor (scalar) - KL divergence averaged over batch
    """
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl


def reparameterize(mu, logvar):
    """
    Reparameterization trick for backpropagation through stochastic sampling.

    Instead of sampling z ~ N(mu, sigma^2) directly (non-differentiable),
    we sample eps ~ N(0, I) and compute z = mu + sigma * eps (differentiable).

    Parameters:
        mu: Tensor of shape (batch, latent_dim) - Mean
        logvar: Tensor of shape (batch, latent_dim) - Log variance

    Returns:
        z: Tensor of shape (batch, latent_dim) - Sampled latent vector
    """
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
    return z


class VAEEncoder(nn.Module):
    """
    VAE Encoder: maps input x to distribution parameters (mu, log_var).

    Unlike deterministic encoder, outputs parameters of q(z|x).

    Architecture: Linear -> ReLU -> (Linear_mu, Linear_logvar)
    Two separate heads share a common hidden layer.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Parameters:
            input_dim: int - Input dimension
            hidden_dim: int - Hidden layer dimension
            latent_dim: int - Latent space dimension
        """
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.Linear(hidden_dim, latent_dim * 2),)
        self.latent_dim = latent_dim
        # TODO: Define shared layer and separate mu/logvar heads

    def forward(self, x):
        """
        Encode input to latent distribution parameters.

        Parameters:
            x: Tensor of shape (batch, input_dim)

        Returns:
            mu: Tensor of shape (batch, latent_dim)
            logvar: Tensor of shape (batch, latent_dim)
        """
        params = self.layers(x)
        mu = params[..., :self.latent_dim]
        logvar = params[..., self.latent_dim:]
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder: maps sampled z to reconstruction p(x|z).

    Architecture: Linear -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                    nn.Linear(hidden_dim, output_dim),)
        # TODO: Define layers

    def forward(self, z):
        """
        Decode latent to reconstruction.

        Parameters:
            z: Tensor of shape (batch, latent_dim)

        Returns:
            x_recon: Tensor of shape (batch, output_dim)
        """
        x_recon = self.layers(z)
        return x_recon


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Loss = Reconstruction + KL Divergence (ELBO)

    Capabilities:
        1. Reconstruction: encode -> sample -> decode
        2. Generation: sample z ~ N(0,I) -> decode
        3. Interpolation: interpolate in latent space -> decode
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        """
        Full VAE forward pass.

        Parameters:
            x: Tensor of shape (batch, input_dim)

        Returns:
            x_recon: Tensor - Reconstruction
            mu: Tensor - Latent mean
            logvar: Tensor - Latent log variance
            z: Tensor - Sampled latent
        """
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def compute_loss(self, x, x_recon, mu, logvar):
        """
        Compute VAE loss: reconstruction + KL divergence.

        Parameters:
            x: Tensor - Original input
            x_recon: Tensor - Reconstruction
            mu, logvar: Tensor - Latent distribution parameters

        Returns:
            total_loss: Tensor (scalar)
            recon_loss: Tensor (scalar)
            kl_loss: Tensor (scalar)
        """
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = kl_divergence(mu, logvar)
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, n_samples, device="cpu"):
        """
        Generate new samples by sampling from prior p(z) = N(0, I).

        Parameters:
            n_samples: int - Number of samples to generate
            device: str - Device for tensors

        Returns:
            samples: Tensor of shape (n_samples, input_dim)
        """
        z = torch.randn(n_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples

    @torch.no_grad()
    def reconstruct(self, x):
        """Reconstruct input (uses mean, not sampled z)."""
        mu, _ = self.encoder(x)
        return self.decoder(mu)

    @torch.no_grad()
    def interpolate(self, x1, x2, n_steps=10):
        """
        Interpolate between two inputs in latent space.

        Parameters:
            x1, x2: Tensor of shape (1, input_dim) - Two inputs
            n_steps: int - Number of interpolation steps

        Returns:
            interpolations: Tensor of shape (n_steps, input_dim)
        """
        mu, logvar = self.encoder(x1)
        z1 = reparameterize(mu, logvar)
        mu, logvar = self.encoder(x2)
        z2 = reparameterize(mu, logvar)
        steps = torch.linspace(0, 1, n_steps).unsqueeze(1)
        interpolations = steps @ z1 + (1 - steps) @ z2
        interpolations = self.decoder(interpolations)
        return interpolations


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, input_dim, hidden_dim, latent_dim = 32, 784, 256, 20

    vae = VAE(input_dim, hidden_dim, latent_dim)
    x = torch.randn(batch_size, input_dim)

    x_recon, mu, logvar, z = vae(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Sampled z shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")