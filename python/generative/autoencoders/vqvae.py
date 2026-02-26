"""
Vector Quantized Variational Autoencoder (VQ-VAE) Implementation

COMPREHENSIVE THEORY:
======================

Motivation and Problem Statement:
----------------------------------
Standard VAEs face the "posterior collapse" problem where the latent variables become
uninformative (the encoder ignores z and the KL divergence becomes negligible). This
occurs because the powerful decoder can reconstruct data without using the latent code,
making the z variables redundant.

VQ-VAE solves this by:
1. Replacing continuous latent distributions with discrete vectors (codes)
2. Using a learnable codebook/dictionary of quantized embeddings
3. Removing the KL divergence term entirely
4. Forcing the encoder output to be matched (quantized) to nearest codebook vector

The Discrete Latent Space:
--------------------------
Instead of z ~ q_φ(z|x) where z is continuous, VQ-VAE:
1. Encodes x to continuous vectors e = encoder(x)
2. Quantizes e to nearest codebook vector: e_q = codebook[argmin_k ||e - c_k||^2]
3. Uses quantized representation e_q for decoding

This gives discrete, interpretable latent codes while maintaining gradients.

The Codebook (Dictionary):
--------------------------
The codebook is a learned matrix C of shape (num_embeddings, embedding_dim):
C = [c_1, c_2, ..., c_K]^T  where each c_i is embedding_dim-dimensional

The model learns which codebook entries are most useful for representing data.
Codebook entries that are frequently used are updated; rarely-used entries can be
reinitialized to ensure all codebook vectors are utilized.

VQ Loss Function:
------------------
L_VQ = L_recon + L_codebook + L_commitment

Where:
1. L_recon = ||x - decoder(sg[e_q])||^2
   - Reconstruction loss from quantized embedding
   - sg[...] is stop-gradient operator (no gradients through e_q to encoder)

2. L_codebook = ||e - sg[e_q]||^2
   - Updates codebook to move towards encoder outputs
   - Uses stop-gradient on encoder output (prevents gradient flow)

3. L_commitment = β * ||sg[e_q] - e||^2
   - Encourages encoder outputs to stay close to codebook
   - β is typically 0.25, controls strength of commitment
   - Uses stop-gradient on codebook (prevents gradient flow)

The stop-gradient operators (sg) are critical:
- They prevent gradients from flowing in directions that would make the problem trivial
- Encoder gradients: only through L_commitment term
- Codebook gradients: only through L_codebook term
- This asymmetry prevents collapse and ensures useful learning

Why This Works:
----------------
- L_reconstruction: Decoder learns to reconstruct from quantized codes
- L_codebook: Codebook vectors move towards encoder outputs
- L_commitment: Encoder learns to predict good codebook indices

The asymmetry of stop-gradients prevents the encoder from just learning to output constant
vectors or the codebook from collapsing. Each component has a specific learning objective.

The Straight-Through Estimator:
--------------------------------
Quantization is not differentiable: argmin_k ||e - c_k||^2 has zero gradient almost everywhere.

Solution: Use the "straight-through" gradient approximation:
- Forward pass: e_q = quantize(e) = codebook[argmin_k ||e - c_k||^2]
- Backward pass: ∂L/∂e ≈ ∂L/∂e_q  (copy gradients from e_q to e)

This is NOT the true gradient, but an approximation that works well in practice.
Implementation: During backprop, treat quantization as identity operation:
    def straight_through_quantize(e):
        # Forward: quantize
        # Backward: pass through gradients as if identity
        e_q = quantize(e)
        return e + (e_q - e).detach()

The .detach() operation ensures:
- Forward: e_q is used (true quantization)
- Backward: gradients flow as if through e (straight-through)

Codebook Collapse Prevention:
------------------------------
Without proper updates, some codebook entries become unused. VQ-VAE uses:

1. Exponential Moving Average (EMA) update (recommended):
   - n_k ← γ * n_k + (1-γ) * sum_batch(1_{e~quantize(e)=c_k})
   - c_k ← (sum_{e~c_k} e) / n_k  (normalized sum of encoder outputs)
   - γ is decay rate (typically 0.99)
   - Only updates used codebook entries

2. L2 normalization of codebook

3. Periodic reinitialization of unused codes

Perplexity Metric:
-------------------
Measures the effective number of codebook entries used:
Perplexity = exp(-sum_k p_k * log(p_k))

where p_k = count(e_q == c_k) / batch_size

- Perplexity = num_embeddings: All codes equally used (ideal)
- Perplexity << num_embeddings: Many codes unused (codebook collapse)
- Perplexity ≈ 1: Only one code used (total collapse)

Advantages of VQ-VAE:
---------------------
1. Discrete latent codes → interpretable representations
2. No posterior collapse
3. Codebook acts as learned discrete representation
4. Can be combined with autoregressive models (e.g., PixelCNN) for generation
5. Lower dimensional discrete codes → better for downstream tasks
6. Strong empirical results on image and speech generation

Disadvantages:
----------------
1. Codebook collapse: some embeddings unused
2. Discrete latent space less smooth than continuous VAEs
3. More complex training (needs careful tuning of β and learning rates)
4. Straight-through estimator is biased gradient approximation

Applications:
--------------
1. Speech synthesis: VQ-VAE learns discrete codes for speech
2. Image generation: VQ-VAE + PixelCNN for high-quality image generation
3. Video generation: Hierarchical VQ-VAE for video sequences
4. Robotics: Learn discrete action codes
5. Text representation: Quantized encodings for NLP

Key Papers:
===========
- van den Oord et al. (2017): "Neural Discrete Representation Learning"
  https://arxiv.org/abs/1711.00937
- Razavi et al. (2019): "Generating Diverse High-Fidelity Images with VQ-VAE-2"
  https://arxiv.org/abs/1906.00446
- Tjandra et al. (2017): "An Unsupervised Autoregressive Model for Speech Representation Learning"
  https://arxiv.org/abs/1703.02143
"""

import numpy as np
from typing import Tuple, Optional, Dict

from python.nn_core import Module, Parameter
from python.nn_core.layers import Linear
from python.nn_core.module import Sequential


class VectorQuantizer(Module):
    """
    Vector Quantizer: Maps continuous vectors to nearest codebook embeddings.

    The core component of VQ-VAE that discretizes encoder outputs.

    Args:
        num_embeddings (int): Number of codebook vectors (K in theory)
        embedding_dim (int): Dimension of each codebook vector
        beta (float): Commitment loss coefficient (default: 0.25)
        use_ema (bool): Use exponential moving average for codebook updates
        ema_decay (float): EMA decay rate (default: 0.99)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Initialize codebook: shape (num_embeddings, embedding_dim)
        # Use Parameter instead of nn.Embedding
        codebook_init = np.random.uniform(-1 / num_embeddings, 1 / num_embeddings,
                                          (num_embeddings, embedding_dim))
        self.embedding = Parameter(codebook_init.astype(np.float32))

        # For tracking perplexity
        self.cluster_size = np.zeros(num_embeddings)
        self.w = self.embedding.data.copy()

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Quantize encoder outputs using codebook.

        Applies straight-through estimator for gradient flow:
        - Forward: Use true quantization
        - Backward: Copy gradients as if through identity

        Args:
            z (np.ndarray): Encoder outputs of shape (batch_size, ..., embedding_dim)

        Returns:
            Tuple containing:
                - z_q (np.ndarray): Quantized outputs, same shape as z
                - loss_dict (dict): Contains 'quantize_loss', 'commitment_loss', 'perplexity'
        """
        raise NotImplementedError(
            "TODO: Implement vector quantization forward pass. "
            "1. Flatten z to shape (N, embedding_dim) "
            "2. Compute distances to all codebook entries: ||z - c_k||^2 "
            "3. Find nearest codebook index per sample: argmin_k distance "
            "4. Quantize: z_q = codebook[indices] "
            "5. Compute straight-through gradient approximation for backprop "
            "6. Compute loss components (see compute_loss) "
            "7. Update codebook if use_ema=True "
            "8. Reshape z_q back to original shape "
            "Return (z_q, loss_dict)"
        )

    def compute_loss(
        self,
        z: np.ndarray,
        z_q: np.ndarray,
        perplexity: float,
    ) -> Dict[str, float]:
        """
        Compute VQ loss components:
        - L_commit: encoder commitment to codes
        - L_codebook: codebook updates

        Args:
            z (np.ndarray): Encoder outputs (original continuous)
            z_q (np.ndarray): Quantized outputs
            perplexity (float): Perplexity metric

        Returns:
            Dict with keys:
                - 'commit_loss': β * ||z_q_stopped - z||^2
                - 'quantize_loss': ||z_stopped - z_q||^2
                - 'perplexity': Effective number of used codes
        """
        raise NotImplementedError(
            "TODO: Implement VQ loss computation. "
            "Commitment loss: self.beta * ||z_q - z||^2 (stop gradients on z_q) "
            "Quantize loss: ||z - z_q||^2 (stop gradients on z) "
            "Both losses should be mean-reduced over batch. "
            "Return dict with these components."
        )

    def update_codebook_ema(self, z: np.ndarray, indices: np.ndarray):
        """
        Update codebook using exponential moving average (recommended for training stability).

        For each codebook entry k:
        - n_k ← γ * n_k + (1-γ) * count(indices == k)
        - c_k ← (γ * c_k + (1-γ) * sum_{e_i~c_k} z_i) / (n_k + epsilon)

        Args:
            z (np.ndarray): Flattened encoder outputs of shape (N, embedding_dim)
            indices (np.ndarray): Indices of nearest codebook entries, shape (N,)
        """
        raise NotImplementedError(
            "TODO: Implement EMA update for codebook. "
            "1. Count how many times each embedding is used "
            "2. Update cluster_size: cluster_size ← γ * cluster_size + (1-γ) * counts "
            "3. Sum encoder outputs for each cluster "
            "4. Update embeddings: embedding_weight ← sums / (cluster_size + epsilon) "
            "Use self.ema_decay as γ."
        )

    def get_perplexity(self, indices: np.ndarray) -> float:
        """
        Compute perplexity: exp(-sum_k p_k * log(p_k)).

        Measures effective number of codebook entries used.

        Args:
            indices (np.ndarray): Quantized indices

        Returns:
            float: Scalar perplexity value
        """
        raise NotImplementedError(
            "TODO: Compute perplexity metric. "
            "1. Count usage of each codebook entry "
            "2. Compute probability p_k = count / total "
            "3. Return exp(-sum(p_k * log(p_k + epsilon))). "
            "High perplexity means codes are used evenly. "
            "Low perplexity means codebook collapse."
        )


class VQVAE(Module):
    """
    Vector Quantized Variational Autoencoder.

    Combines discretization through vector quantization with VAE-style reconstruction loss.
    Learns a discrete latent representation that captures data variation in interpretable codes.

    Args:
        input_dim (int): Dimension of input data
        latent_dim (int): Dimension of latent vectors (after quantization)
        hidden_dims (list): Hidden dimensions for encoder/decoder
        num_embeddings (int): Number of codebook vectors (dictionary size)
        beta (float): Commitment loss weight (default: 0.25)
        use_ema (bool): Use EMA for codebook updates (default: True)
        activation: Activation function (default: ReLU)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
        num_embeddings: int = 512,
        beta: float = 0.25,
        use_ema: bool = True,
        activation=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.activation = activation or (lambda x: np.maximum(x, 0))  # ReLU

        # Build encoder
        self.encoder = self._build_encoder()

        # Vector quantizer with codebook
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            beta=beta,
            use_ema=use_ema,
        )

        # Build decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> Sequential:
        """
        Build encoder: input_dim -> hidden_dims -> latent_dim

        Returns:
            Sequential: Sequential encoder
        """
        raise NotImplementedError(
            "TODO: Implement encoder. "
            "Build Sequential from input_dim through hidden_dims to latent_dim."
        )

    def _build_decoder(self) -> Sequential:
        """
        Build decoder: latent_dim -> hidden_dims (reversed) -> input_dim

        Returns:
            Sequential: Sequential decoder
        """
        raise NotImplementedError(
            "TODO: Implement decoder. "
            "Build Sequential from latent_dim through reversed hidden_dims to input_dim."
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to latent representation.

        Args:
            x (np.ndarray): Input of shape (batch_size, input_dim)

        Returns:
            np.ndarray: Encoded latent of shape (batch_size, latent_dim)
        """
        raise NotImplementedError(
            "TODO: Implement encode. "
            "Pass x through self.encoder and return latent representation."
        )

    def quantize(self, z: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Quantize latent representation using codebook.

        Args:
            z (np.ndarray): Latent from encoder

        Returns:
            Tuple of (z_q, loss_dict)
        """
        raise NotImplementedError(
            "TODO: Implement quantize. "
            "Pass z through self.quantizer and return (z_q, loss_dict)."
        )

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        """
        Decode quantized latent to reconstruction.

        Args:
            z_q (np.ndarray): Quantized latent of shape (batch_size, latent_dim)

        Returns:
            np.ndarray: Reconstruction of shape (batch_size, input_dim)
        """
        raise NotImplementedError(
            "TODO: Implement decode. "
            "Pass z_q through self.decoder and return reconstruction."
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass: encode -> quantize -> decode.

        Args:
            x (np.ndarray): Input array

        Returns:
            Tuple containing:
                - x_recon (np.ndarray): Reconstructed input
                - z_q (np.ndarray): Quantized latent codes
                - loss_dict (dict): Loss components
        """
        raise NotImplementedError(
            "TODO: Implement forward pass. "
            "1. Encode x to get z "
            "2. Quantize z to get z_q and loss_dict "
            "3. Decode z_q to get x_recon "
            "Return (x_recon, z_q, loss_dict)."
        )

    def compute_loss(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        loss_dict: Dict,
    ) -> Dict[str, float]:
        """
        Compute total VQ-VAE loss:
        L = L_recon + L_quantize + L_commit

        Args:
            x (np.ndarray): Original input
            x_recon (np.ndarray): Reconstructed input
            loss_dict (dict): Loss components from quantizer

        Returns:
            Dict with keys:
                - 'loss': Total loss
                - 'recon': Reconstruction loss
                - 'quantize': Codebook loss (from loss_dict)
                - 'commit': Commitment loss (from loss_dict)
                - 'perplexity': Effective number of used codes
        """
        raise NotImplementedError(
            "TODO: Compute total VQ-VAE loss. "
            "recon = np.mean((x - x_recon) ** 2) "
            "Extract quantize and commit losses from loss_dict. "
            "total_loss = recon + quantize + commit "
            "Return dict with all components."
        )

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate samples by randomly sampling from codebook.

        Args:
            num_samples (int): Number of samples

        Returns:
            np.ndarray: Generated samples of shape (num_samples, input_dim)
        """
        raise NotImplementedError(
            "TODO: Implement sampling. "
            "1. Randomly sample embedding indices: indices ~ Uniform(0, num_embeddings) "
            "2. Get embeddings from codebook using indices "
            "3. Decode embeddings to samples "
            "Return samples."
        )

    def encode_to_codes(self, x: np.ndarray) -> np.ndarray:
        """
        Encode data to discrete codebook indices.

        This is useful for downstream tasks like autoregressive modeling.

        Args:
            x (np.ndarray): Input data

        Returns:
            np.ndarray: Codebook indices for each sample
        """
        raise NotImplementedError(
            "TODO: Implement encode_to_codes. "
            "1. Encode x to get z "
            "2. Find nearest codebook index for each z "
            "Return indices array of shape (batch_size,)."
        )

    def codes_to_samples(self, codes: np.ndarray) -> np.ndarray:
        """
        Convert discrete codes back to samples.

        Args:
            codes (np.ndarray): Codebook indices

        Returns:
            np.ndarray: Reconstructed samples
        """
        raise NotImplementedError(
            "TODO: Implement codes_to_samples. "
            "1. Look up embeddings for codes in codebook "
            "2. Decode embeddings "
            "Return samples."
        )

    def analyze_codebook_usage(self) -> Dict:
        """
        Analyze codebook health and usage statistics.

        Returns:
            Dict with metrics:
                - 'perplexity': Current effective usage
                - 'active_codes': Number of used codes
                - 'inactive_codes': Number of unused codes
                - 'usage_ratio': Fraction of codebook used
        """
        raise NotImplementedError(
            "TODO: Implement codebook analysis. "
            "Track which codes were used in recent batches. "
            "Compute perplexity, count active/inactive codes. "
            "Return metrics dict."
        )
