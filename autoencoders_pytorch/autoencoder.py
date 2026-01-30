"""
Basic Autoencoder in PyTorch.

An autoencoder learns compressed representations by training an encoder-decoder
pair to reconstruct its input through a bottleneck layer.

Architecture:
    Input (d) -> Encoder -> Latent (k) -> Decoder -> Output (d)

Loss: Reconstruction loss (MSE or BCE)

Reference: MIT 6.390 Chapter 10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder: maps input x in R^d to latent z in R^k.

    Architecture: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Parameters:
            input_dim: int (d) - Input dimension
            hidden_dim: int - Hidden layer dimension
            latent_dim: int (k) - Latent space dimension
        """
        super().__init__()
        # TODO: Define layers
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.Linear(hidden_dim, latent_dim),)

    def forward(self, x):
        """
        Forward pass through encoder.

        Parameters:
            x: Tensor of shape (batch, input_dim)

        Returns:
            z: Tensor of shape (batch, latent_dim)
        """
        z = self.layers(x)
        return z


class Decoder(nn.Module):
    """
    Decoder: maps latent z in R^k to reconstruction x_hat in R^d.

    Architecture: Linear -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        Parameters:
            latent_dim: int (k) - Latent space dimension
            hidden_dim: int - Hidden layer dimension
            output_dim: int (d) - Output dimension (same as input)
        """
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                    nn.Linear(hidden_dim, output_dim),)
        # TODO: Define layers

    def forward(self, z):
        """
        Forward pass through decoder.

        Parameters:
            z: Tensor of shape (batch, latent_dim)

        Returns:
            x_recon: Tensor of shape (batch, output_dim)
        """
        x_recon = self.layers(z)
        return x_recon


class Autoencoder(nn.Module):
    """
    Complete Autoencoder combining encoder and decoder.

    Training objective: minimize reconstruction loss
        L = ||x - decode(encode(x))||^2
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        """
        Full forward pass: encode then decode.

        Parameters:
            x: Tensor of shape (batch, input_dim)

        Returns:
            x_recon: Tensor of shape (batch, input_dim) - Reconstruction
            z: Tensor of shape (batch, latent_dim) - Latent representation
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_loss(self, x, x_recon, loss_type="mse"):
        """
        Compute reconstruction loss.

        Parameters:
            x: Tensor - Original input
            x_recon: Tensor - Reconstruction
            loss_type: str - "mse" or "bce"

        Returns:
            loss: Tensor (scalar)
        """
        if loss_type == "mse":
            loss = torch.nn.functional.mse_loss(x, x_recon)
        elif loss_type == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x, x_recon)
        return loss

    def encode(self, x):
        """Get latent representation."""
        return self.encoder(x)

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

    def reconstruct(self, x):
        """Reconstruct input through bottleneck."""
        x_recon, _ = self.forward(x)
        return x_recon


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, input_dim, hidden_dim, latent_dim = 32, 784, 256, 32

    ae = Autoencoder(input_dim, hidden_dim, latent_dim)
    x = torch.randn(batch_size, input_dim)

    x_recon, z = ae(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")