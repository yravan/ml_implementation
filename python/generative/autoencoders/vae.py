"""
Variational Autoencoder (VAE) Implementation

COMPREHENSIVE THEORY:
======================

Probabilistic Formulation:
--------------------------
A Variational Autoencoder (VAE) is a deep generative model that combines ideas from variational
inference and neural networks. Unlike vanilla autoencoders, VAEs define a probabilistic model
over data by introducing a latent variable z and optimizing a principled probabilistic objective.

The model assumes data x is generated from latent variable z through a generative process:
1. Sample z from prior p(z) = N(0, I)
2. Generate x from p(x|z) defined by decoder network with parameters θ

Inference Problem:
-----------------
Given observed data x, we want to infer the posterior p(z|x). This is intractable to compute
directly. VAE solves this by introducing a variational approximation q_φ(z|x) (encoder) that
learns to approximate the true posterior.

The Evidence Lower Bound (ELBO):
---------------------------------
We cannot directly maximize log p(x) (the data likelihood) because it requires computing the
intractable integral ∫ p(x|z)p(z) dz. Instead, VAE maximizes the Evidence Lower Bound (ELBO):

log p(x) ≥ E_q[log p(x|z)] - KL(q_φ(z|x) || p(z))
           \_________________/     \____________________/
          Reconstruction Term      Regularization (KL Divergence)

This can be rewritten as:
ELBO = E_q_φ(z|x)[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))

The first term encourages good reconstruction of x from z (like vanilla AE).
The second term regularizes the learned posterior to stay close to the prior.

Why This Formulation?
---------------------
- Maximizing ELBO is equivalent to minimizing: -log p_θ(x|z) + KL(q_φ(z|x) || p(z))
- First term: Reconstruction loss (e.g., MSE or cross-entropy depending on likelihood model)
- Second term: KL regularization pushing q_φ closer to standard normal prior
- The KL divergence encourages meaningful, compact latent space (disentanglement)
- Sampling z from p(z) after training enables generation of new samples

The Reparameterization Trick:
-----------------------------
The key innovation enabling gradient-based optimization of VAEs. Standard sampling is not
differentiable (can't backprop through randomness). Solution: reparameterize the sampling:

Instead of: z ~ q_φ(z|x) = N(μ_φ(x), Σ_φ(x))
We use:     z = μ_φ(x) + Σ_φ(x)^(1/2) ⊙ ε,  where ε ~ N(0, I)

where ⊙ denotes element-wise multiplication. This separates stochastic and deterministic
components, allowing gradients to flow through the encoder parameters φ.

Computational Perspective:
- Encoder outputs two vectors: μ (mean) and log_var (log variance)
- For each sample, we sample noise ε ~ N(0, I)
- Compute z = μ + exp(0.5 * log_var) * ε
- This is differentiable w.r.t. μ and log_var!

Why log_var Instead of σ^2?
- Stability: log_var prevents variance from becoming negative or too small
- Numerical stability in KL divergence computation

Mathematical Details of KL Divergence for Gaussians:
----------------------------------------------------
For q_φ(z|x) = N(μ, Σ) with diagonal covariance and p(z) = N(0, I):

KL(N(μ, Σ) || N(0, I)) = -1/2 * sum_d (1 + log(σ_d^2) - μ_d^2 - σ_d^2)
                        = -1/2 * sum_d (1 + log_var_d - μ_d^2 - exp(log_var_d))

This has a closed-form solution, making VAE efficient to train.

Reconstruction Loss:
--------------------
Depends on the likelihood model p(x|z):

1. For continuous data with Gaussian likelihood:
   L_recon = MSE(x, x_recon) = ||x - μ_decoder(z)||^2 / (2σ_x^2)
   where σ_x is reconstruction variance (often fixed)

2. For binary data with Bernoulli likelihood:
   L_recon = -E[x * log(p) + (1-x) * log(1-p)]
   where p = sigmoid(decoder_output)

Complete Loss Function:
-----------------------
L_ELBO = -Reconstruction_Loss + KL_Divergence
       = MSE(x, x_recon) + (-1/2 * sum(1 + log_var - μ^2 - exp(log_var)))
       = MSE(x, x_recon) + KL_Loss

The negative sign on KL is because we want to MINIMIZE the final loss, and KL divergence
is positive, so we MINIMIZE KL which is equivalent to MAXimizing the ELBO.

Interpretation:
- Without KL term: Model collapses to vanilla AE with no prior constraints
- Without reconstruction term: Latent space becomes useless (random noise)
- Together: Trade-off between accurate reconstruction and staying close to prior

Sampling for Generation:
------------------------
After training:
1. Sample z ~ p(z) = N(0, I) from standard normal
2. Generate x = decoder(z)
3. This produces samples from the learned data distribution p(x)

Key Papers:
===========
- Kingma & Welling (2013): "Auto-Encoding Variational Bayes"
  https://arxiv.org/abs/1312.6114
- Doersch (2016): "Tutorial on Variational Autoencoders"
  https://arxiv.org/abs/1606.05908
- Rezende et al. (2014): "Stochastic Backpropagation and Approximate Inference..."
  https://arxiv.org/abs/1401.4082
"""

import numpy as np
from typing import Tuple, Optional, Dict

from python.nn_core import Module, Parameter
from python.nn_core.layers import Linear
from python.nn_core.module import Sequential


class VAE(Module):
    """
    Variational Autoencoder (VAE) with diagonal Gaussian encoder and decoder.

    Implements the full VAE framework including:
    - Reparameterization trick for differentiable sampling
    - ELBO loss computation (reconstruction + KL divergence)
    - Generation from prior
    - Latent space interpolation

    Args:
        input_dim (int): Dimension of input data
        latent_dim (int): Dimension of latent representation
        hidden_dims (list): List of hidden dimensions for encoder/decoder layers
        beta (float): Weight for KL divergence term (default: 1.0). Values > 1 encourage more regularization
        activation: Activation function to use (default: ReLU)
        reconstruction_loss_type (str): Type of reconstruction loss ('mse' or 'bce')
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
        beta: float = 1.0,
        activation=None,
        reconstruction_loss_type: str = 'mse',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.beta = beta
        self.activation = activation or (lambda x: np.maximum(x, 0))  # ReLU
        self.reconstruction_loss_type = reconstruction_loss_type

        # Build encoder that outputs mean and log variance
        self.encoder = self._build_encoder()
        self.fc_mu = None      # Will map to mean μ
        self.fc_logvar = None  # Will map to log variance (log σ^2)

        # Build decoder
        self.decoder = self._build_decoder()

        if reconstruction_loss_type not in ['mse', 'bce']:
            raise ValueError(f"Unknown reconstruction loss type: {reconstruction_loss_type}")

    def _build_encoder(self) -> Sequential:
        """
        Build encoder network: input_dim -> hidden_dims -> output_hidden

        The output will then be split into mean and log_var.

        Returns:
            Sequential: Sequential encoder network
        """
        raise NotImplementedError(
            "TODO: Implement encoder architecture. "
            "Build Sequential from input_dim through hidden_dims. "
            "Output should be a hidden representation that will be split into μ and log_var. "
            "Hint: Last layer should output to hidden_dims[-1]."
        )

    def _build_mu_logvar_heads(self):
        """
        Build output heads for mean and log variance from encoder output.

        Creates two separate linear layers:
        - fc_mu: maps encoder_output_dim -> latent_dim
        - fc_logvar: maps encoder_output_dim -> latent_dim

        This is typically called after encoder is built to know output dimension.

        Returns:
            Tuple of (mu_head, logvar_head) Modules
        """
        raise NotImplementedError(
            "TODO: Implement mean and log_var output heads. "
            "Create self.fc_mu and self.fc_logvar as Linear layers. "
            "Both map from the encoder output dimension to latent_dim. "
            "Hint: encoder_output_dim = hidden_dims[-1]"
        )

    def _build_decoder(self) -> Sequential:
        """
        Build decoder network: latent_dim -> hidden_dims (reversed) -> input_dim

        Returns:
            Sequential: Sequential decoder network
        """
        raise NotImplementedError(
            "TODO: Implement decoder architecture. "
            "Build Sequential from latent_dim through reversed hidden_dims to input_dim. "
            "Note: No activation on final layer for linear reconstruction."
        )

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input to mean and log_variance of latent distribution.

        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dim)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (mean, log_var) each of shape (batch_size, latent_dim)
        """
        raise NotImplementedError(
            "TODO: Implement encode method. "
            "Pass x through self.encoder, then through self.fc_mu and self.fc_logvar. "
            "Return (mu, logvar) tuple."
        )

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: sample z = μ + σ * ε, where ε ~ N(0,I).

        This enables gradient flow through the stochastic sampling process.

        Mathematical formulation:
            z = μ + exp(0.5 * log_var) * ε
            where ε ~ N(0, I) is standard normal noise

        Args:
            mu (np.ndarray): Mean of latent distribution, shape (batch_size, latent_dim)
            logvar (np.ndarray): Log variance of latent distribution, shape (batch_size, latent_dim)

        Returns:
            np.ndarray: Sampled latent vector z, shape (batch_size, latent_dim)
        """
        raise NotImplementedError(
            "TODO: Implement reparameterization trick. "
            "Sample standard normal noise ε from np.random.randn(*mu.shape). "
            "Compute std = exp(0.5 * logvar). "
            "Return z = mu + std * epsilon. "
            "This implements: z = μ + σ * ε where σ = exp(0.5 * log_var)"
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
        Forward pass through VAE: encode, reparameterize, decode.

        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dim)

        Returns:
            Tuple containing:
                - x_recon (np.ndarray): Reconstructed input, shape (batch_size, input_dim)
                - mu (np.ndarray): Latent mean, shape (batch_size, latent_dim)
                - logvar (np.ndarray): Latent log variance, shape (batch_size, latent_dim)
                - z (np.ndarray): Sampled latent vector, shape (batch_size, latent_dim)
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
        Compute reconstruction loss between input and reconstruction.

        For MSE: L_recon = MSE(x, x_recon)
        For BCE: L_recon = -E[x * log(x_recon) + (1-x) * log(1-x_recon)]

        Args:
            x (np.ndarray): Original input
            x_recon (np.ndarray): Reconstructed input

        Returns:
            float: Scalar reconstruction loss
        """
        raise NotImplementedError(
            "TODO: Implement reconstruction loss. "
            "For MSE: use np.mean((x - x_recon) ** 2). "
            "For BCE: use appropriate binary cross-entropy formula. "
            "Return loss scalar."
        )

    def compute_kl_loss(self, mu: np.ndarray, logvar: np.ndarray) -> float:
        """
        Compute Kullback-Leibler divergence between q(z|x) and p(z).

        For diagonal Gaussian with q(z|x) = N(μ, Σ) and p(z) = N(0, I):

        KL(q || p) = -0.5 * sum_d (1 + log_var_d - μ_d^2 - exp(log_var_d))
                   = -0.5 * sum_d (1 + log(σ_d^2) - μ_d^2 - σ_d^2)

        Averaged over batch:
        KL_loss = (1/batch_size) * sum_batch KL(...)

        Args:
            mu (np.ndarray): Mean of q(z|x), shape (batch_size, latent_dim)
            logvar (np.ndarray): Log variance of q(z|x), shape (batch_size, latent_dim)

        Returns:
            float: Scalar KL divergence loss
        """
        raise NotImplementedError(
            "TODO: Implement KL divergence loss. "
            "Use the closed-form formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) "
            "Sum over latent dimension, mean over batch dimension. "
            "Return scalar loss. "
            "Hint: np.sum() and np.mean() will be useful."
        )

    def compute_elbo_loss(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute complete ELBO loss: reconstruction loss + β * KL loss.

        L_ELBO = L_recon + β * L_KL

        The β parameter allows weighting the KL term (β-VAE). When β > 1, more emphasis
        is placed on the regularization, leading to more disentangled representations.

        Args:
            x (np.ndarray): Original input
            x_recon (np.ndarray): Reconstructed input
            mu (np.ndarray): Latent mean
            logvar (np.ndarray): Latent log variance

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'elbo': Total ELBO loss
                - 'recon': Reconstruction loss component
                - 'kl': KL divergence loss component
        """
        raise NotImplementedError(
            "TODO: Implement ELBO loss computation. "
            "1. Compute reconstruction loss using compute_reconstruction_loss() "
            "2. Compute KL loss using compute_kl_loss() "
            "3. Compute total: elbo = recon + self.beta * kl "
            "Return dict with 'elbo', 'recon', 'kl' keys."
        )

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate samples from the learned generative model.

        Samples from prior p(z) = N(0, I) and decodes through decoder:
        1. Sample z ~ N(0, I)
        2. Compute x = decoder(z)

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            np.ndarray: Generated samples of shape (num_samples, input_dim)
        """
        raise NotImplementedError(
            "TODO: Implement sampling from prior. "
            "1. Sample z from standard normal: z = np.random.randn(num_samples, self.latent_dim) "
            "2. Decode z to get samples: x = self.decode(z) "
            "3. Return x."
        )

    def interpolate(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        steps: int = 10,
    ) -> np.ndarray:
        """
        Interpolate between two data points in latent space.

        Linearly interpolate between z1 and z2:
        z_t = (1-t) * z1 + t * z2, where t in [0, 1]

        Args:
            x1 (np.ndarray): First data point
            x2 (np.ndarray): Second data point
            steps (int): Number of interpolation steps

        Returns:
            np.ndarray: Interpolated samples of shape (steps, input_dim)
        """
        raise NotImplementedError(
            "TODO: Implement latent space interpolation. "
            "1. Encode x1 and x2 to get mu1, mu2 "
            "2. Create interpolation weights: t = np.linspace(0, 1, steps) "
            "3. For each t, compute z_t = (1-t) * mu1 + t * mu2 "
            "4. Decode all z_t and return interpolated samples."
        )
