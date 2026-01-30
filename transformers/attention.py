import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention mechanism.

    This is the core attention computation: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Parameters:
        Q: np.ndarray of shape (n_queries, d_k) - Query matrix
        K: np.ndarray of shape (n_keys, d_k) - Key matrix
        V: np.ndarray of shape (n_keys, d_v) - Value matrix
        mask: optional np.ndarray of shape (n_queries, n_keys) - Attention mask
              Where mask is True/1, attention is allowed; where False/0, attention is blocked
              (set to -inf before softmax)

    Returns:
        output: np.ndarray of shape (n_queries, d_v) - Attention output
        attention_weights: np.ndarray of shape (n_queries, n_keys) - Attention weights (alpha)

    Reference equations from notes:
        - Attention weights: α_ij = softmax_j(Q_i^T K_j / sqrt(d_k))  [Eq 9.10]
        - Output: y^(i) = Σ_j α_ij V_j  [Eq 9.6]
    """
    n_queries, d_k = Q.shape
    n_keys, d_v = V.shape

    attention_matrix = Q @ K.transpose(-1, -2) / np.sqrt(d_k)
    if mask is not None:
        attention_matrix[mask == 0] = -np.inf
    attention_weights = softmax(attention_matrix, axis=-1)
    attention_outputs = attention_weights @ V

    output = attention_outputs
    attention_weights = attention_weights

    return output, attention_weights


def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=None):
    """
    Multi-head attention mechanism.

    Splits the embedding dimension into multiple heads, applies attention independently
    to each head, then concatenates and projects the results.

    Parameters:
        X: np.ndarray of shape (n, d) - Input sequence (n tokens, d dimensions)
        W_q: np.ndarray of shape (d, d_k * n_heads) - Query projection weights
        W_k: np.ndarray of shape (d, d_k * n_heads) - Key projection weights
        W_v: np.ndarray of shape (d, d_k * n_heads) - Value projection weights
        W_o: np.ndarray of shape (d_k * n_heads, d) - Output projection weights
        n_heads: int - Number of attention heads (H in the notes)
        mask: optional np.ndarray - Attention mask

    Returns:
        output: np.ndarray of shape (n, d) - Multi-head attention output
        attention_weights: list of np.ndarray - Attention weights for each head

    Reference equations from notes:
        - Q^(h) = X W_{h,q}  [Eq 9.7]
        - K^(h) = X W_{h,k}  [Eq 9.8]
        - V^(h) = X W_{h,v}  [Eq 9.9]
        - Output combines all heads: u'^(i) = Σ_h W_{h,c}^T Σ_j α_ij^(h) V_j^(h)  [Eq 9.11]
    """
    n, d = X.shape
    d_total = W_q.shape[1]
    d_k = d_total // n_heads  # Dimension per head

    Q, K, V = X @ W_q, X @ W_k, X @ W_v

    Q, K, V = Q.reshape((-1, n_heads, d_k)), K.reshape((-1, n_heads, d_k)), V.reshape((-1, n_heads, d_k))

    Q, K, V = Q.transpose((1, 0, 2)), K.transpose((1, 0, 2)), V.transpose((1, 0, 2))

    attention_matrices = Q @ K.transpose((0, 2, 1)) / np.sqrt(d_k)
    if mask is not None:
        attention_matrices[np.tile(mask,(n_heads, 1, 1)) == 0] = -np.inf
    attention_matrices = softmax(attention_matrices, axis=-1)
    attention_output = attention_matrices @ V
    attention_output = attention_output.transpose((1, 0, 2)).reshape( n, d)

    output =attention_output
    attention_weights = attention_matrices

    return output, attention_weights


def create_causal_mask(n):
    """
    Create a causal (autoregressive) mask that prevents attending to future tokens.

    Parameters:
        n: int - Sequence length

    Returns:
        mask: np.ndarray of shape (n, n) - Lower triangular mask
              mask[i,j] = True if j <= i (can attend), False if j > i (cannot attend)

    This is used in decoder self-attention to prevent "looking ahead."
    """
    # TODO: Create lower triangular mask
    # mask[i, j] = True means query i can attend to key j
    # In causal attention, position i can only attend to positions 0, 1, ..., i

    mask = np.ones((n, n), dtype=bool)
    mask = np.tril(mask)
    return mask


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    n, d_k, d_v = 4, 8, 8
    Q = np.random.randn(n, d_k)
    K = np.random.randn(n, d_k)
    V = np.random.randn(n, d_v)

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Weights sum per query: {weights.sum(axis=-1)}")  # Should be all 1s
