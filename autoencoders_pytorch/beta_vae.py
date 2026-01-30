"""
Beta-VAE in PyTorch.

Beta-VAE weights the KL divergence term by a hyperparameter beta,
encouraging disentangled latent representations.

Loss = Reconstruction + beta * KL Divergence

beta > 1: stronger regularization, encourages disentanglement
beta = 1: standard VAE
beta < 1: better reconstruction, more entangled

Disentanglement: each latent dimension captures an independent factor
of variation (e.g., for faces: age, glasses, smile as separate dims).

Reference: Higgins et al., "beta-VAE: Learning Basic Visual Concepts
            with a Constrained Variational Framework" (ICLR 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.graph import rag_boundary

from .vae import VAEEncoder, VAEDecoder, reparameterize, kl_divergence


class BetaVAE(nn.Module):
    """
    Beta-VAE with controllable KL weight.

    Loss = Reconstruction + beta * KL(q(z|x) || p(z))

    Higher beta encourages disentanglement at the cost of reconstruction quality.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, beta=4.0):
        """
        Parameters:
            input_dim: int - Input dimension
            hidden_dim: int - Hidden layer dimension
            latent_dim: int - Latent space dimension
            beta: float - KL divergence weight (beta > 1 for disentanglement)
        """
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.beta = beta
        self.latent_dim = latent_dim

    def forward(self, x):
        """
        Forward pass.

        Returns:
            x_recon, mu, logvar, z
        """
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def compute_loss(self, x, x_recon, mu, logvar):
        """
        Compute beta-VAE loss with weighted KL term.

        Loss = Reconstruction + beta * KL

        Returns:
            total_loss, recon_loss, kl_loss (unweighted), weighted_kl
        """
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = kl_divergence(mu, logvar)
        weighted_kl = self.beta * kl_loss
        total_loss = recon_loss + weighted_kl
        return total_loss, recon_loss, kl_loss, weighted_kl

    @torch.no_grad()
    def traverse_latent(self, x, dim, range_vals=(-3, 3), n_steps=10):
        """
        Traverse a single latent dimension to visualize what it encodes.

        Fix all latent dims except one, vary that dim, see what changes.

        Parameters:
            x: Tensor of shape (1, input_dim) - Input to encode
            dim: int - Which latent dimension to traverse
            range_vals: tuple - Range of values to traverse
            n_steps: int - Number of steps

        Returns:
            traversals: Tensor of shape (n_steps, input_dim)
        """
        steps = torch.linspace(*range_vals, steps=n_steps)
        traversals = x.repeat(n_steps, 1)
        traversals[:, dim] = steps
        return traversals


class AnnealedBetaVAE(nn.Module):
    """
    Beta-VAE with KL annealing (warm-up).

    During training, beta gradually increases from 0 to target value.
    This prevents posterior collapse (latent dimensions being ignored).

    Schedule: beta(t) = min(beta_max, beta_max * t / warmup_steps)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 beta_max=4.0, warmup_steps=10000):
        """
        Parameters:
            beta_max: float - Final beta value
            warmup_steps: int - Steps to reach full beta
        """
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.beta_max = beta_max
        self.warmup_steps = warmup_steps
        self.step = 0
        self.latent_dim = latent_dim

    def get_beta(self):
        """Get current beta value based on training step."""
        beta = min(self.beta_max / self.warmup_steps * self.step, self.beta_max)
        return beta

    def forward(self, x):
        """Forward pass. Returns: x_recon, mu, logvar, z"""
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def train_step(self, x, optimizer):
        """
        Single training step with annealed beta.

        Should increment self.step after each call.
        """
        optimizer.zero_grad()
        x_recon, mu, logvar, z = self.forward(x)
        recon_loss = torch.nn.functional.mse_loss(x_recon, x)
        kl_loss = kl_divergence(mu, logvar) * self.get_beta()
        total_loss = recon_loss + kl_loss
        total_loss.backward()
        optimizer.step()
        self.step += 1


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, input_dim, hidden_dim, latent_dim = 32, 784, 256, 10

    vae = BetaVAE(input_dim, hidden_dim, latent_dim, beta=1.0)
    beta_vae = BetaVAE(input_dim, hidden_dim, latent_dim, beta=4.0)

    x = torch.randn(batch_size, input_dim)

    print(f"Standard VAE (beta=1)")
    print(f"Beta-VAE (beta=4) - encourages disentanglement")