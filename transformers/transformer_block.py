import numpy as np
from .attention import multi_head_attention, scaled_dot_product_attention


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer Normalization.

    Normalizes each token embedding independently (across the feature dimension).

    Parameters:
        x: np.ndarray of shape (n, d) - Input (n tokens, d dimensions)
        gamma: np.ndarray of shape (d,) - Scale parameter
        beta: np.ndarray of shape (d,) - Shift parameter
        eps: float - Small constant for numerical stability

    Returns:
        out: np.ndarray of shape (n, d) - Normalized output

    Reference equation from notes [Eq 9.15]:
        LayerNorm(z; γ, β) = γ * (z - μ_z) / σ_z + β

    Where μ_z and σ_z are computed per-token across the d dimensions [Eq 9.16, 9.17].
    """
    n, d = x.shape
    out = np.zeros_like(x)

    # TODO: Implement layer normalization
    #
    # For each token (row):
    # Step 1: Compute mean across features
    # mu = x.mean(axis=-1, keepdims=True)  # shape: (n, 1)
    #
    # Step 2: Compute standard deviation across features
    # sigma = x.std(axis=-1, keepdims=True)  # shape: (n, 1)
    #
    # Step 3: Normalize and apply scale/shift
    # out = gamma * (x - mu) / (sigma + eps) + beta

    return out


def feed_forward_network(x, W1, b1, W2, b2):
    """
    Position-wise Feed-Forward Network (FFN).

    Two linear transformations with a ReLU activation in between.
    Applied independently to each token.

    Parameters:
        x: np.ndarray of shape (n, d) - Input
        W1: np.ndarray of shape (d, d_ff) - First layer weights
        b1: np.ndarray of shape (d_ff,) - First layer bias
        W2: np.ndarray of shape (d_ff, d) - Second layer weights
        b2: np.ndarray of shape (d,) - Second layer bias

    Returns:
        out: np.ndarray of shape (n, d) - FFN output

    Reference equation from notes [Eq 9.13]:
        z' = W2^T ReLU(W1^T u)
    """
    # TODO: Implement feed-forward network
    #
    # Step 1: First linear layer
    # hidden = x @ W1 + b1  # shape: (n, d_ff)
    #
    # Step 2: ReLU activation
    # hidden = np.maximum(0, hidden)
    #
    # Step 3: Second linear layer
    # out = hidden @ W2 + b2  # shape: (n, d)

    out = np.zeros_like(x)
    return out


def transformer_block(x, params, n_heads, mask=None):
    """
    Single Transformer Block.

    Consists of:
        1. Multi-head self-attention with residual connection and layer norm
        2. Feed-forward network with residual connection and layer norm

    Parameters:
        x: np.ndarray of shape (n, d) - Input sequence
        params: dict containing:
            - W_q, W_k, W_v, W_o: Attention projection matrices
            - gamma1, beta1: LayerNorm parameters after attention
            - W1, b1, W2, b2: FFN parameters
            - gamma2, beta2: LayerNorm parameters after FFN
        n_heads: int - Number of attention heads
        mask: optional attention mask

    Returns:
        z: np.ndarray of shape (n, d) - Block output

    Reference equations from notes:
        u = LayerNorm(x + MultiHeadAttention(x))  [Eq 9.12]
        z = LayerNorm(u + FFN(u))  [Eq 9.14]

    Note: This implements "Post-LN" (original Transformer). "Pre-LN" variant
    applies LayerNorm before attention/FFN instead of after.
    """
    n, d = x.shape

    # TODO: Implement transformer block
    #
    # Step 1: Multi-head self-attention
    # attn_output, _ = multi_head_attention(x, params['W_q'], params['W_k'],
    #                                        params['W_v'], params['W_o'], n_heads, mask)
    #
    # Step 2: Residual connection + LayerNorm
    # u = layer_norm(x + attn_output, params['gamma1'], params['beta1'])
    #
    # Step 3: Feed-forward network
    # ffn_output = feed_forward_network(u, params['W1'], params['b1'],
    #                                    params['W2'], params['b2'])
    #
    # Step 4: Residual connection + LayerNorm
    # z = layer_norm(u + ffn_output, params['gamma2'], params['beta2'])

    z = np.zeros_like(x)
    return z


def transformer_block_pre_ln(x, params, n_heads, mask=None):
    """
    Transformer Block with Pre-LayerNorm (Pre-LN variant).

    This variant applies LayerNorm BEFORE attention/FFN, which improves
    training stability and is used in GPT-2, GPT-3, etc.

    Parameters: Same as transformer_block

    Returns:
        z: np.ndarray of shape (n, d) - Block output

    Structure:
        u = x + MultiHeadAttention(LayerNorm(x))
        z = u + FFN(LayerNorm(u))
    """
    n, d = x.shape

    # TODO: Implement Pre-LN transformer block
    #
    # Step 1: LayerNorm before attention
    # x_norm = layer_norm(x, params['gamma1'], params['beta1'])
    #
    # Step 2: Multi-head self-attention + residual
    # attn_output, _ = multi_head_attention(x_norm, ...)
    # u = x + attn_output
    #
    # Step 3: LayerNorm before FFN
    # u_norm = layer_norm(u, params['gamma2'], params['beta2'])
    #
    # Step 4: FFN + residual
    # ffn_output = feed_forward_network(u_norm, ...)
    # z = u + ffn_output

    z = np.zeros_like(x)
    return z


def init_transformer_block_params(d, d_ff, n_heads, seed=None):
    """
    Initialize parameters for a transformer block.

    Parameters:
        d: int - Model dimension
        d_ff: int - Feed-forward hidden dimension (typically 4*d)
        n_heads: int - Number of attention heads
        seed: optional random seed

    Returns:
        params: dict - Initialized parameters
    """
    if seed is not None:
        np.random.seed(seed)

    scale = 0.02  # Xavier-like initialization

    params = {
        # Attention projections
        'W_q': np.random.randn(d, d) * scale,
        'W_k': np.random.randn(d, d) * scale,
        'W_v': np.random.randn(d, d) * scale,
        'W_o': np.random.randn(d, d) * scale,

        # First LayerNorm
        'gamma1': np.ones(d),
        'beta1': np.zeros(d),

        # FFN
        'W1': np.random.randn(d, d_ff) * scale,
        'b1': np.zeros(d_ff),
        'W2': np.random.randn(d_ff, d) * scale,
        'b2': np.zeros(d),

        # Second LayerNorm
        'gamma2': np.ones(d),
        'beta2': np.zeros(d),
    }

    return params


def transformer(x, blocks_params, n_heads, mask=None):
    """
    Full Transformer (stack of L transformer blocks).

    Parameters:
        x: np.ndarray of shape (n, d) - Input sequence (with positional encoding added)
        blocks_params: list of dicts - Parameters for each block
        n_heads: int - Number of attention heads
        mask: optional attention mask

    Returns:
        z: np.ndarray of shape (n, d) - Transformer output

    Reference [Eq 9.18]:
        f_θL ∘ ... ∘ f_θ2 ∘ f_θ1(x)
    """
    # TODO: Implement full transformer
    #
    # z = x
    # for params in blocks_params:
    #     z = transformer_block(z, params, n_heads, mask)
    # return z

    return x


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    n, d, d_ff, n_heads = 10, 64, 256, 8

    x = np.random.randn(n, d)
    params = init_transformer_block_params(d, d_ff, n_heads, seed=42)

    z = transformer_block(x, params, n_heads)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
