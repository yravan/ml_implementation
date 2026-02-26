"""
Vanilla Autoencoder (AE) Implementation

Theory:
--------
A Vanilla Autoencoder is the simplest form of autoencoder, consisting of an encoder that maps
input data to a latent representation, and a decoder that reconstructs the input from this
latent code. The model is trained end-to-end to minimize reconstruction error. Unlike VAEs,
vanilla autoencoders do not impose probabilistic constraints on the latent space and do not
have an explicit regularization term. The learned latent representations capture the essential
features of the data in an unsupervised manner. This architecture forms the foundation for
more sophisticated autoencoders like denoising AEs and VAEs.

Key Papers:
-----------
- Hinton & Zemel (1994): "Reducing the Dimensionality of Data with Neural Networks"
  https://www.cs.toronto.edu/~hinton/science.pdf
- Goodfellow et al. (2016): "Deep Learning" - Chapter on Autoencoders
  https://www.deeplearningbook.org/

Loss Function:
---------------
L(x) = MSE(x, x_reconstructed) = (1/N) * sum((x_i - x_reconstructed_i)^2)

Alternative: L2 reconstruction loss (per element)
    L = ||x - decoder(encoder(x))||^2_2

Mathematical Notation:
----------------------
- Input: x in R^D (data space)
- Encoder: z = f_encoder(x, theta_encoder) where z in R^d (d << D)
- Decoder: x_recon = f_decoder(z, theta_decoder) where x_recon in R^D
- Loss: L = E[||x - f_decoder(f_encoder(x))||^2]
"""

import numpy as np
from typing import Tuple, Optional

from python.nn_core import Module, Parameter
from python.nn_core.layers import Linear
from python.nn_core.module import Sequential


class VanillaAutoencoder(Module):
    """
    Vanilla Autoencoder with configurable encoder and decoder architectures.

    This model learns to reconstruct input data through a bottleneck latent representation,
    forcing the model to capture the most salient features of the data.

    Args:
        input_dim (int): Dimension of input data
        latent_dim (int): Dimension of latent representation
        hidden_dims (list): List of hidden dimensions for encoder/decoder layers
        activation: Activation function to use (default: ReLU)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
        activation=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.activation = activation or lambda x: np.maximum(x, 0)  # ReLU

        # Initialize encoder
        self.encoder = self._build_encoder()

        # Initialize decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> Sequential:
        """
        Build encoder network: input_dim -> hidden_dims -> latent_dim

        Returns:
            Sequential: Sequential encoder network
        """
        raise NotImplementedError(
            "TODO: Implement encoder architecture. "
            "Build a sequential network that projects from input_dim through hidden_dims "
            "to latent_dim using self.hidden_dims and self.activation. "
            "Hint: Use Sequential with Linear layers and activation functions."
        )

    def _build_decoder(self) -> Sequential:
        """
        Build decoder network: latent_dim -> hidden_dims (reversed) -> input_dim

        Returns:
            Sequential: Sequential decoder network
        """
        raise NotImplementedError(
            "TODO: Implement decoder architecture. "
            "Build a sequential network that projects from latent_dim through "
            "reversed hidden_dims back to input_dim. "
            "Note: No activation function on final output layer (linear reconstruction)."
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input data to latent representation.

        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dim)

        Returns:
            np.ndarray: Latent representation of shape (batch_size, latent_dim)
        """
        raise NotImplementedError(
            "TODO: Implement encode method. "
            "Simply pass input x through self.encoder and return the latent code z."
        )

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation to reconstructed data.

        Args:
            z (np.ndarray): Latent array of shape (batch_size, latent_dim)

        Returns:
            np.ndarray: Reconstructed data of shape (batch_size, input_dim)
        """
        raise NotImplementedError(
            "TODO: Implement decode method. "
            "Simply pass latent code z through self.decoder and return reconstructed x."
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: encode input and decode latent representation.

        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dim)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (reconstructed_x, latent_z)
        """
        raise NotImplementedError(
            "TODO: Implement forward pass. "
            "Call encode(x) to get latent code z, then decode(z) to get reconstruction. "
            "Return tuple of (reconstructed_x, latent_z)."
        )

    def compute_loss(self, x: np.ndarray) -> float:
        """
        Compute reconstruction loss for training.

        L = MSE(x, decoder(encoder(x)))

        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dim)

        Returns:
            float: Scalar loss value
        """
        raise NotImplementedError(
            "TODO: Implement loss computation. "
            "Forward pass through the model to get reconstruction x_recon, "
            "then compute MSE loss between x and x_recon. Return the loss scalar."
        )

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        Reconstruct input data.

        Args:
            x (np.ndarray): Input array

        Returns:
            np.ndarray: Reconstructed data
        """
        raise NotImplementedError(
            "TODO: Implement reconstruct method. "
            "Call forward(x) and return just the reconstructed array (first element of tuple)."
        )


# Alias for common naming
Autoencoder = VanillaAutoencoder

