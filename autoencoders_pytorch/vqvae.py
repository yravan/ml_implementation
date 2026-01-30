"""
Vector Quantized VAE (VQ-VAE) in PyTorch.

VQ-VAE learns DISCRETE latent representations using a learned codebook
of embedding vectors, instead of continuous Gaussian latents.

Key Components:
    1. Encoder: maps input to continuous embeddings z_e
    2. Vector Quantization: snaps z_e to nearest codebook vector z_q
    3. Decoder: reconstructs from quantized embeddings z_q

Loss = Reconstruction + Codebook Loss + Commitment Loss
     = ||x - decode(z_q)||^2
       + ||sg[z_e] - z_q||^2        (move codebook toward encoder)
       + beta * ||z_e - sg[z_q]||^2  (commit encoder to codebook)

Straight-Through Estimator:
    Forward: z_q (quantized)
    Backward: gradient copies through to z_e
    Implemented as: z_e + (z_q - z_e).detach()

Applications: DALL-E, audio synthesis, video prediction

Reference: van den Oord et al., "Neural Discrete Representation Learning" (2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.TiffTags import lookup


class VQVAEEncoder(nn.Module):
    """
    VQ-VAE Encoder: maps input to continuous latent embeddings.

    Unlike VAE, outputs deterministic embeddings (no mu, sigma).

    Architecture: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        """
        Encode input to continuous embeddings.

        Parameters:
            x: Tensor of shape (batch, input_dim)

        Returns:
            z_e: Tensor of shape (batch, latent_dim) - Pre-quantization embeddings
        """
        z_e = self.layers(x)
        return z_e


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer.

    Maintains a codebook of K embedding vectors and quantizes inputs
    to their nearest neighbor in the codebook.

    Use nn.Embedding for the codebook.
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        """
        Parameters:
            num_embeddings: int (K) - Number of vectors in codebook
            embedding_dim: int - Dimension of each embedding vector
            beta: float - Commitment loss weight
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, z_e):
        """
        Quantize encoder outputs using codebook.

        Steps:
            1. Compute distances ||z_e - e_k||^2 for all k
               = ||z_e||^2 - 2*z_e*e_k^T + ||e_k||^2
            2. Find nearest codebook vector (argmin distance)
            3. Look up quantized vectors z_q
            4. Compute VQ loss = codebook_loss + beta * commitment_loss
               codebook_loss  = F.mse_loss(z_q, z_e.detach())
               commitment_loss = F.mse_loss(z_e, z_q.detach())
            5. Apply straight-through estimator:
               z_q_st = z_e + (z_q - z_e).detach()

        Parameters:
            z_e: Tensor of shape (batch, embedding_dim)

        Returns:
            z_q: Tensor of shape (batch, embedding_dim) - With straight-through grad
            indices: Tensor of shape (batch,) - Codebook indices used
            vq_loss: Tensor (scalar) - Vector quantization loss
        """

        distances = torch.linalg.norm(z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0), dim=2)
        indices = torch.argmin(distances, dim=1)
        z_q = self.lookup(indices)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
        z_q = z_e + (z_q - z_e).detach()
        return z_q, indices, vq_loss

    def lookup(self, indices):
        """
        Look up embeddings by index.

        Parameters:
            indices: Tensor of shape (batch,) - Codebook indices

        Returns:
            embeddings: Tensor of shape (batch, embedding_dim)
        """
        embeddings = self.codebook(indices)
        return embeddings


class VQVAEDecoder(nn.Module):
    """
    VQ-VAE Decoder: reconstructs from quantized embeddings.

    Architecture: Linear -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_dim),)

    def forward(self, z_q):
        """
        Decode from quantized embeddings.

        Parameters:
            z_q: Tensor of shape (batch, latent_dim)

        Returns:
            x_recon: Tensor of shape (batch, output_dim)
        """
        x_recon = self.layers(z_q)
        return x_recon


class VQVAE(nn.Module):
    """
    Vector Quantized VAE.

    Uses discrete latent codes from a learned codebook instead of
    continuous Gaussian latents.

    Total Loss = Reconstruction + VQ Loss
               = ||x - x_hat||^2 + ||sg[z_e] - z_q||^2 + beta*||z_e - sg[z_q]||^2
    """

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 num_embeddings=512, beta=0.25):
        super().__init__()
        self.encoder = VQVAEEncoder(input_dim, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, beta)
        self.decoder = VQVAEDecoder(latent_dim, hidden_dim, input_dim)
        self.num_embeddings = num_embeddings

    def forward(self, x):
        """
        Full VQ-VAE forward pass.

        Parameters:
            x: Tensor of shape (batch, input_dim)

        Returns:
            x_recon: Tensor - Reconstruction
            z_e: Tensor - Pre-quantization embeddings
            z_q: Tensor - Quantized embeddings (with straight-through grad)
            indices: Tensor - Codebook indices used
            vq_loss: Tensor - VQ loss
        """
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q, indices, vq_loss

    def compute_loss(self, x, x_recon, vq_loss):
        """
        Compute total loss = reconstruction + VQ loss.

        Returns:
            total_loss, recon_loss, vq_loss
        """
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = vq_loss + recon_loss
        return total_loss, recon_loss, vq_loss

    @torch.no_grad()
    def encode(self, x):
        """
        Encode to discrete codes.

        Returns:
            indices: Tensor - Codebook indices (the discrete codes)
        """
        z_e = self.encoder(x)
        _, indices, _ = self.quantizer(z_e)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices):
        """
        Decode from discrete codes (indices).

        Parameters:
            indices: Tensor of shape (batch,) - Codebook indices

        Returns:
            x_recon: Tensor
        """
        z_q = self.quantizer.lookup(indices)
        x_recon = self.decoder(z_q)
        return x_recon

    @torch.no_grad()
    def reconstruct(self, x):
        """Reconstruct through quantization."""
        x_recon, _, _, _, _ = self.forward(x)
        return x_recon

    @torch.no_grad()
    def get_codebook_usage(self, data):
        """
        Compute codebook usage statistics.

        Useful for diagnosing codebook collapse (when only few codes are used).

        Parameters:
            data: Tensor of shape (n_samples, input_dim)

        Returns:
            usage: Tensor of shape (num_embeddings,) - Usage count per code
            perplexity: Tensor (scalar) - Effective codebook size (exp(entropy))
        """
        z_q = self.encoder(data)
        _, indices, _ = self.quantizer(z_q)
        usage = torch.bincount(indices, minlength=self.num_embeddings)
        prob = usage / torch.sum(usage)
        perplexity = -(prob[prob > 0] * torch.log2(prob[prob > 0])).sum()
        return usage, perplexity


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, input_dim, hidden_dim, latent_dim = 32, 784, 256, 64
    num_embeddings = 512

    vqvae = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings)
    x = torch.randn(batch_size, input_dim)

    x_recon, z_e, z_q, indices, vq_loss = vqvae(x)
    print(f"Input shape: {x.shape}")
    print(f"Pre-quantization z_e shape: {z_e.shape}")
    print(f"Quantized z_q shape: {z_q.shape}")
    print(f"Codebook indices shape: {indices.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")