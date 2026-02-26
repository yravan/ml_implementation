"""
β-VAE (Beta-Variational Autoencoder) Implementation

Theory:
--------
β-VAE is an extension of the standard VAE that learns disentangled representations by
weighting the KL divergence term with a hyperparameter β > 1. This encourages the latent
factors to be statistically independent and interpretable, with each dimension capturing
a single meaningful variation in the data.

In standard VAE, the loss is:
L = Reconstruction + KL_divergence

In β-VAE, the loss becomes:
L = Reconstruction + β * KL_divergence, where β ≥ 1

Interpretation of β:
- β = 1: Standard VAE (balance between reconstruction and regularization)
- β > 1: Stronger regularization, more emphasis on KL term, leading to more disentangled factors
- Larger β values force the latent space to be more structured and match the prior distribution

Disentanglement:
-----------------
A disentangled representation is one where individual latent dimensions correspond to
semantically meaningful factors of variation in the data. For example, in a generative
model of faces, one dimension might control expression, another controls lighting, etc.

β > 1 encourages disentanglement by:
1. Making KL loss dominant, forcing q(z|x) to match p(z) = N(0,I)
2. Preventing different factors from correlating in latent space
3. Encouraging independence between latent dimensions

Trade-off:
β-VAE exhibits a trade-off between reconstruction quality and disentanglement:
- Low β: Better reconstruction, less disentanglement
- High β: Better disentanglement, but poorer reconstruction

The β parameter allows users to control this trade-off based on their application.

Key Papers:
-----------
- Higgins et al. (2017): "β-VAE: Learning Basic Visual Concepts with a Constrained..."
  https://arxiv.org/abs/1804.03599
- Burgess et al. (2018): "Understanding disentangling in β-VAE"
  https://arxiv.org/abs/1804.03599

Loss Function:
---------------
L_β-VAE = L_recon + β * L_KL
        = MSE(x, x_recon) + β * (-1/2 * sum(1 + log_var - μ^2 - exp(log_var)))

The β weighting is the core innovation: it changes the relative importance of the
regularization term during training.
"""

import numpy as np
from typing import Tuple, Optional, Dict

from python.nn_core import Module, Parameter
from python.nn_core.layers import Linear
from python.nn_core.module import Sequential


class BetaVAE(Module):
    """
    β-VAE for learning disentangled representations.

    β-VAE extends standard VAE by introducing a hyperparameter β that weights the KL
    divergence term. Setting β > 1 encourages the learned latent factors to be more
    disentangled and statistically independent.

    Args:
        input_dim (int): Dimension of input data
        latent_dim (int): Dimension of latent representation
        hidden_dims (list): List of hidden dimensions for encoder/decoder layers
        beta (float): Weight for KL divergence (default: 4.0). Values > 1 encourage disentanglement
        activation: Activation function to use (default: ReLU)
        reconstruction_loss_type (str): Type of reconstruction loss ('mse' or 'bce')
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
        beta: float = 4.0,
        activation=None,
        reconstruction_loss_type: str = 'mse',
    ):
        super().__init__()

        if beta < 1.0:
            raise ValueError(
                f"β must be >= 1.0 for β-VAE (got {beta}). "
                "Use standard VAE for β = 1.0 or lower values."
            )

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.beta = beta
        self.activation = activation or (lambda x: np.maximum(x, 0))  # ReLU
        self.reconstruction_loss_type = reconstruction_loss_type

        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.fc_mu = None      # Will map to mean μ
        self.fc_logvar = None  # Will map to log variance

        self.decoder = self._build_decoder()

        if reconstruction_loss_type not in ['mse', 'bce']:
            raise ValueError(f"Unknown reconstruction loss type: {reconstruction_loss_type}")

        # Track loss components for analysis
        self.last_recon_loss = None
        self.last_kl_loss = None

    def _build_encoder(self) -> Sequential:
        """
        Build encoder network: input_dim -> hidden_dims -> output_hidden

        Returns:
            Sequential: Sequential encoder network
        """
        raise NotImplementedError(
            "TODO: Implement encoder architecture. "
            "Build Sequential from input_dim through hidden_dims. "
            "Output dimension should be hidden_dims[-1]."
        )

    def _build_mu_logvar_heads(self):
        """
        Build output heads for mean and log variance from encoder output.

        Returns:
            Tuple of (mu_head, logvar_head) Modules
        """
        raise NotImplementedError(
            "TODO: Implement mean and log_var output heads. "
            "Create self.fc_mu and self.fc_logvar as Linear layers "
            "mapping from hidden_dims[-1] to latent_dim."
        )

    def _build_decoder(self) -> Sequential:
        """
        Build decoder network: latent_dim -> hidden_dims (reversed) -> input_dim

        Returns:
            Sequential: Sequential decoder network
        """
        raise NotImplementedError(
            "TODO: Implement decoder architecture. "
            "Build Sequential from latent_dim through reversed hidden_dims to input_dim."
        )

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input to mean and log_variance of latent distribution.

        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dim)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (mean, log_var)
        """
        raise NotImplementedError(
            "TODO: Implement encode method. "
            "Pass x through encoder, then through fc_mu and fc_logvar. "
            "Return (mu, logvar) tuple."
        )

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = μ + σ * ε.

        Args:
            mu (np.ndarray): Mean of latent distribution
            logvar (np.ndarray): Log variance of latent distribution

        Returns:
            np.ndarray: Sampled latent vector z
        """
        raise NotImplementedError(
            "TODO: Implement reparameterization. "
            "Sample ε ~ N(0,I) and compute z = mu + exp(0.5 * logvar) * epsilon."
        )

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to reconstructed input.

        Args:
            z (np.ndarray): Latent vector of shape (batch_size, latent_dim)

        Returns:
            np.ndarray: Reconstructed data of shape (batch_size, input_dim)
        """
        raise NotImplementedError(
            "TODO: Implement decode method. "
            "Pass z through self.decoder and return reconstruction."
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through β-VAE: encode, reparameterize, decode.

        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dim)

        Returns:
            Tuple containing (x_recon, mu, logvar, z)
        """
        raise NotImplementedError(
            "TODO: Implement forward pass. "
            "1. Encode x to get mu and logvar "
            "2. Reparameterize to get z "
            "3. Decode z to get x_recon "
            "Return (x_recon, mu, logvar, z)"
        )

    def compute_reconstruction_loss(self, x: np.ndarray, x_recon: np.ndarray) -> float:
        """
        Compute reconstruction loss.

        Args:
            x (np.ndarray): Original input
            x_recon (np.ndarray): Reconstructed input

        Returns:
            float: Scalar reconstruction loss
        """
        raise NotImplementedError(
            "TODO: Implement reconstruction loss. "
            "For MSE: use np.mean((x - x_recon) ** 2). "
            "For BCE: use appropriate binary cross-entropy formula."
        )

    def compute_kl_loss(self, mu: np.ndarray, logvar: np.ndarray) -> float:
        """
        Compute KL divergence: KL(q(z|x) || p(z)).

        KL_loss = -0.5 * sum_d (1 + log_var_d - μ_d^2 - exp(log_var_d))

        Args:
            mu (np.ndarray): Mean of q(z|x)
            logvar (np.ndarray): Log variance of q(z|x)

        Returns:
            float: Scalar KL loss
        """
        raise NotImplementedError(
            "TODO: Implement KL divergence. "
            "Use formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)). "
            "Sum over latent dim, mean over batch."
        )

    def compute_beta_vae_loss(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute β-VAE loss: reconstruction + β * KL.

        This is the key difference from standard VAE: the β weighting on KL term.
        L_β-VAE = L_recon + β * L_KL

        Args:
            x (np.ndarray): Original input
            x_recon (np.ndarray): Reconstructed input
            mu (np.ndarray): Latent mean
            logvar (np.ndarray): Latent log variance

        Returns:
            Dict[str, float]: Dictionary with keys:
                - 'loss': Total β-VAE loss
                - 'recon': Reconstruction loss component
                - 'kl': KL divergence loss component (before β weighting)
        """
        raise NotImplementedError(
            "TODO: Implement β-VAE loss. "
            "1. Compute reconstruction loss "
            "2. Compute KL loss "
            "3. Compute total loss: loss = recon + self.beta * kl "
            "Return dict with 'loss', 'recon', 'kl' keys. "
            "Store loss components in self.last_recon_loss and self.last_kl_loss."
        )

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate samples from learned distribution by sampling from prior.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            np.ndarray: Generated samples of shape (num_samples, input_dim)
        """
        raise NotImplementedError(
            "TODO: Implement sampling. "
            "Sample z ~ N(0, I) and decode: x = decoder(z). "
            "Return x."
        )

    def get_disentanglement_loss_component(self) -> float:
        """
        Get the relative contribution of KL term (disentanglement objective).

        Returns:
            float: Ratio of (β * KL loss) to total loss
        """
        raise NotImplementedError(
            "TODO: Implement disentanglement ratio calculation. "
            "Return: (self.beta * self.last_kl_loss) / (self.last_recon_loss + self.beta * self.last_kl_loss)"
        )

    def analyze_disentanglement(self, x_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze properties of disentangled representation.

        Computes statistics on latent codes to assess disentanglement quality:
        - Mean and std of each latent dimension
        - Correlation matrix between dimensions
        - Effective dimensionality (number of used dimensions)

        Args:
            x_batch (np.ndarray): Batch of input data

        Returns:
            Dict with analysis metrics:
                - 'latent_means': Mean of each latent dimension
                - 'latent_stds': Std dev of each latent dimension
                - 'latent_correlation': Correlation matrix between dimensions
                - 'active_dims': Number of significantly active dimensions (std > 0.1)
        """
        raise NotImplementedError(
            "TODO: Implement disentanglement analysis. "
            "1. Encode batch to get z samples "
            "2. Compute mean and std per dimension "
            "3. Compute correlation matrix between dimensions "
            "4. Count 'active' dimensions (std > 0.1) "
            "Return dict with these metrics."
        )
