import numpy as np
from .attention import multi_head_attention


def adaln(x, gamma, beta, eps=1e-5):
    """
    Adaptive Layer Normalization (AdaLN).

    Unlike standard LayerNorm where gamma/beta are learned parameters,
    AdaLN receives gamma/beta as inputs (generated from conditioning).

    Parameters:
        x: np.ndarray of shape (n, d) - Input
        gamma: np.ndarray of shape (d,) or (n, d) - Scale (from conditioning)
        beta: np.ndarray of shape (d,) or (n, d) - Shift (from conditioning)
        eps: float - Numerical stability

    Returns:
        out: np.ndarray of shape (n, d) - Normalized and modulated output

    Formula: AdaLN(x, γ, β) = γ * ((x - μ) / σ) + β
    """

    mu, sigma = np.mean(x, axis = -1, keepdims=True), np.std(x, axis = -1, keepdims=True)
    if len(gamma.shape) == 1:
        gamma = np.expand_dims(gamma, axis = 0)
    if len(beta.shape) == 1:
        beta = np.expand_dims(beta, axis = 0)
    out = gamma * (x - mu) / (sigma + eps) + beta
    return out


def adaln_zero(x, gamma, beta, alpha, eps=1e-5):
    """
    AdaLN-Zero: Adaptive LayerNorm with additional gating parameter.

    Used in DiT to enable zero-initialization of residual blocks.
    The alpha parameter gates the output, initialized to zero so the
    block initially acts as identity.

    Parameters:
        x: np.ndarray of shape (n, d) - Input
        gamma: np.ndarray of shape (d,) - Scale
        beta: np.ndarray of shape (d,) - Shift
        alpha: np.ndarray of shape (d,) - Gate/scale for residual (initialized to 0)
        eps: float - Numerical stability

    Returns:
        out: np.ndarray of shape (n, d) - Output scaled by alpha

    Formula: AdaLN-Zero(x) = α * AdaLN(x, γ, β)

    When α=0 (initialization), block output is 0, so residual connection
    passes input unchanged: x + 0 = x
    """
    adaln_out = adaln(x, gamma, beta, eps)
    out = adaln_out * alpha[None,...]
    return out


def timestep_embedding(timesteps, d, max_period=10000):
    """
    Sinusoidal timestep embeddings (similar to positional encoding).

    Used in diffusion models to encode the noise level/timestep.

    Parameters:
        timesteps: np.ndarray of shape (batch,) - Timestep values (can be float)
        d: int - Embedding dimension
        max_period: int - Maximum period for sinusoids

    Returns:
        emb: np.ndarray of shape (batch, d) - Timestep embeddings
    """
    half = d // 2
    frequencies = 1 / np.power([max_period], np.arange(half) / half)
    inputs = timesteps[..., None] * frequencies[None, :]
    emb = np.concatenate([np.sin(inputs), np.cos(inputs)], axis = -1)
    return emb


def conditioning_mlp(c, W1, b1, W2, b2):
    """
    MLP to project conditioning (timestep + class) to AdaLN parameters.

    Parameters:
        c: np.ndarray of shape (batch, d_cond) - Conditioning vector
        W1: np.ndarray of shape (d_cond, d_hidden) - First layer weights
        b1: np.ndarray of shape (d_hidden,) - First layer bias
        W2: np.ndarray of shape (d_hidden, d_out) - Second layer weights
        b2: np.ndarray of shape (d_out,) - Second layer bias

    Returns:
        out: np.ndarray of shape (batch, d_out) - AdaLN parameters
    """
    out = c @ W1 + b1
    out = np.maximum(out, 0)
    out = out @ W2 + b2
    return out

def feed_forward(x,  W1_ffn, b1_ffn, W2_ffn, b2_ffn):
    out = x @ W1_ffn + b1_ffn
    out = np.maximum(out, 0)
    out = out @ W2_ffn + b2_ffn
    return out


def dit_block(x, c, params, n_heads):
    """
    Diffusion Transformer (DiT) Block with AdaLN-Zero.

    Structure:
        1. AdaLN → Multi-head Self-Attention → Scale by α1 → Residual
        2. AdaLN → FFN → Scale by α2 → Residual

    The conditioning c is used to generate (γ1, β1, α1, γ2, β2, α2).

    Parameters:
        x: np.ndarray of shape (n, d) - Input patches/tokens
        c: np.ndarray of shape (d,) - Conditioning vector (timestep + class embedding)
        params: dict containing:
            - Attention weights: W_q, W_k, W_v, W_o
            - FFN weights: W1_ffn, b1_ffn, W2_ffn, b2_ffn
            - Conditioning MLP: W1_cond, b1_cond, W2_cond, b2_cond
        n_heads: int - Number of attention heads

    Returns:
        z: np.ndarray of shape (n, d) - Block output

    Key insight: Unlike standard transformer where LayerNorm params are static,
    DiT generates them dynamically from conditioning, allowing the model to
    adapt its normalization based on timestep and class.
    """
    adaln_params = conditioning_mlp(c[None, :], params['W1_cond'], params['b1_cond'],
                                     params['W2_cond'], params['b2_cond'])
    adaln_params = adaln_params.squeeze(0)  # (6*d,)
    gamma1, beta1, alpha1, gamma2, beta2, alpha2 = np.split(adaln_params, 6)
    x_norm = adaln(x, gamma1, beta1)
    attn_out, _ = multi_head_attention(x_norm, params['W_q'], params['W_k'],
                                        params['W_v'], params['W_o'], n_heads)
    x = x + alpha1 * attn_out
    x_norm = adaln(x, gamma2, beta2)
    ffn_out = feed_forward(x_norm, params['W1_ffn'], params['b1_ffn'], params['W2_ffn'], params['b2_ffn'])
    z = x + alpha2 * ffn_out
    return z


def dit(x, t, y, blocks_params, embed_params, n_heads):
    """
    Full Diffusion Transformer (DiT).

    Used for image generation in latent diffusion models.

    Parameters:
        x: np.ndarray of shape (n_patches, d) - Input patches (patchified latent)
        t: float or int - Timestep (noise level)
        y: int - Class label
        blocks_params: list of parameter dicts for each DiT block
        embed_params: dict containing:
            - t_embed_W1, t_embed_b1, t_embed_W2, t_embed_b2: Timestep MLP
            - y_embed: np.ndarray of shape (n_classes, d) - Class embeddings
        n_heads: int - Number of attention heads

    Returns:
        out: np.ndarray of shape (n_patches, d) - Denoised output

    Architecture:
        1. Embed timestep with sinusoidal + MLP
        2. Embed class label
        3. Combine: c = t_emb + y_emb
        4. Pass through DiT blocks with conditioning c
        5. Final layer (AdaLN + linear projection)
    """
    n_patches, d = x.shape

    # TODO: Implement full DiT
    #
    # Step 1: Timestep embedding
    # t_sinusoidal = timestep_embedding(np.array([t]), d)  # (1, d)
    # t_emb = conditioning_mlp(t_sinusoidal, embed_params['t_embed_W1'], ...)
    # t_emb = t_emb.squeeze(0)  # (d,)
    #
    # Step 2: Class embedding
    # y_emb = embed_params['y_embed'][y]  # (d,)
    #
    # Step 3: Combine conditioning
    # c = t_emb + y_emb  # (d,)
    #
    # Step 4: Apply DiT blocks
    # z = x
    # for params in blocks_params:
    #     z = dit_block(z, c, params, n_heads)
    #
    # Step 5: Final layer (AdaLN + linear to output channels)
    # ...

    out = np.zeros_like(x)
    return out


def patchify(image, patch_size):
    """
    Convert image to sequence of patches.

    Parameters:
        image: np.ndarray of shape (H, W, C) - Input image
        patch_size: int - Size of each square patch

    Returns:
        patches: np.ndarray of shape (n_patches, patch_size * patch_size * C)

    n_patches = (H // patch_size) * (W // patch_size)
    """
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0

    # TODO: Implement patchify
    #
    n_h = H // patch_size
    n_w = W // patch_size
    patches = image.reshape(n_h, patch_size, n_w, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)  # (n_h, n_w, p, p, C)
    patches = patches.reshape(n_h * n_w, patch_size * patch_size * C)

    return patches


def unpatchify(patches, patch_size, H, W, C):
    """
    Convert patches back to image.

    Parameters:
        patches: np.ndarray of shape (n_patches, patch_dim)
        patch_size: int
        H, W, C: int - Original image dimensions

    Returns:
        image: np.ndarray of shape (H, W, C)
    """
    n_H = H // patch_size
    n_W = W // patch_size
    patches = patches.reshape((n_H, n_W, patch_size, patch_size, C))
    patches = patches.transpose(0, 2, 1, 3, 4)
    image = patches.reshape((H, W, C))
    return image


def init_dit_block_params(d, d_ff, n_heads, seed=None):
    """Initialize parameters for a DiT block."""
    if seed is not None:
        np.random.seed(seed)

    scale = 0.02
    return {
        # Self-attention
        'W_q': np.random.randn(d, d) * scale,
        'W_k': np.random.randn(d, d) * scale,
        'W_v': np.random.randn(d, d) * scale,
        'W_o': np.random.randn(d, d) * scale,
        # FFN
        'W1_ffn': np.random.randn(d, d_ff) * scale,
        'b1_ffn': np.zeros(d_ff),
        'W2_ffn': np.random.randn(d_ff, d) * scale,
        'b2_ffn': np.zeros(d),
        # Conditioning MLP: outputs (γ1, β1, α1, γ2, β2, α2) = 6*d params
        'W1_cond': np.random.randn(d, d * 4) * scale,
        'b1_cond': np.zeros(d * 4),
        'W2_cond': np.random.randn(d * 4, d * 6) * scale,
        'b2_cond': np.zeros(d * 6),
    }


if __name__ == "__main__":
    # Example: DiT for image generation
    np.random.seed(42)

    # Image parameters
    H, W, C = 32, 32, 4  # Latent image (e.g., from VAE encoder)
    patch_size = 4
    n_patches = (H // patch_size) * (W // patch_size)  # 64 patches

    # Model parameters
    d = 64  # Hidden dimension
    d_ff = 256
    n_heads = 8
    n_blocks = 4
    n_classes = 10

    # Create fake latent image
    image = np.random.randn(H, W, C)
    patches = patchify(image, patch_size)
    print(f"Image shape: {image.shape}")
    print(f"Patches shape: {patches.shape}")

    # Project patches to hidden dimension
    patch_dim = patch_size * patch_size * C
    patch_proj = np.random.randn(patch_dim, d) * 0.02
    x = patches @ patch_proj  # (n_patches, d)

    # Initialize parameters
    blocks_params = [init_dit_block_params(d, d_ff, n_heads, seed=i)
                     for i in range(n_blocks)]

    # Run single DiT block
    c = np.random.randn(d)  # Conditioning vector
    z = dit_block(x, c, blocks_params[0], n_heads)
    print(f"DiT block input shape: {x.shape}")
    print(f"DiT block output shape: {z.shape}")
