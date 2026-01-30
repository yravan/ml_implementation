import numpy as np
from .attention import scaled_dot_product_attention, softmax


def cross_attention(X_q, X_kv, W_q, W_k, W_v, W_o, n_heads, mask=None):
    """
    Multi-head Cross-Attention mechanism.

    Unlike self-attention where Q, K, V all come from the same input,
    cross-attention generates queries from one input and keys/values from another.

    This is used in:
        - Encoder-decoder attention (decoder queries attend to encoder outputs)
        - Vision-language models (text queries attend to image features)
        - Conditioning mechanisms in diffusion models

    Parameters:
        X_q: np.ndarray of shape (n_q, d) - Input for queries (e.g., decoder states)
        X_kv: np.ndarray of shape (n_kv, d) - Input for keys/values (e.g., encoder outputs)
        W_q: np.ndarray of shape (d, d_k * n_heads) - Query projection weights
        W_k: np.ndarray of shape (d, d_k * n_heads) - Key projection weights
        W_v: np.ndarray of shape (d, d_k * n_heads) - Value projection weights
        W_o: np.ndarray of shape (d_k * n_heads, d) - Output projection weights
        n_heads: int - Number of attention heads
        mask: optional np.ndarray of shape (n_q, n_kv) - Cross-attention mask

    Returns:
        output: np.ndarray of shape (n_q, d) - Cross-attention output
        attention_weights: list of np.ndarray of shape (n_q, n_kv) - Weights per head

    Key difference from self-attention:
        - Q comes from X_q: Q = X_q @ W_q
        - K, V come from X_kv: K = X_kv @ W_k, V = X_kv @ W_v
        - Output has same sequence length as X_q
    """
    n_q, d = X_q.shape
    n_kv = X_kv.shape[0]
    d_total = W_q.shape[1]
    d_k = d_total // n_heads

    Q = X_q @ W_q
    K = X_kv @ W_k
    V = X_kv @ W_v

    Q, K, V = Q.reshape((n_q, n_heads, d_k)), K.reshape((n_kv, n_heads, d_k)), V.reshape((n_kv, n_heads, d_k))
    Q, K, V = Q.transpose((1, 0, 2)), K.transpose((1, 0, 2)), V.transpose((1, 0, 2))

    attention_matrix = Q @ K.transpose((0, 2, 1)) / np.sqrt(d_k)
    if mask is not None:
        attention_matrix[np.tile(mask, (n_heads, 1, 1)) == 0] = -np.inf
    attention_matrix = softmax(attention_matrix, axis=-1)
    attention_outputs = attention_matrix @ V
    attention_outputs = attention_outputs.transpose((1, 0, 2)).reshape((n_q, d_total))

    output = attention_outputs
    attention_weights = attention_matrix

    return output, attention_weights


def cross_attention_single_head(Q, K, V, mask=None):
    """
    Single-head cross-attention (for educational clarity).

    Parameters:
        Q: np.ndarray of shape (n_q, d_k) - Queries
        K: np.ndarray of shape (n_kv, d_k) - Keys
        V: np.ndarray of shape (n_kv, d_v) - Values
        mask: optional attention mask

    Returns:
        output: np.ndarray of shape (n_q, d_v)
        attention_weights: np.ndarray of shape (n_q, n_kv)

    This is essentially scaled_dot_product_attention but with different
    sequence lengths for Q vs K/V.
    """
    # TODO: Implement (this is the same as scaled_dot_product_attention)
    # The key insight is that n_q can differ from n_kv

    return scaled_dot_product_attention(Q, K, V, mask)


def create_cross_attention_mask(n_q, n_kv, padding_mask_kv=None):
    """
    Create a cross-attention mask.

    Parameters:
        n_q: int - Number of query positions
        n_kv: int - Number of key/value positions
        padding_mask_kv: optional np.ndarray of shape (n_kv,) - True where kv is valid

    Returns:
        mask: np.ndarray of shape (n_q, n_kv) - Attention mask

    Use cases:
        - Masking out padding tokens in the encoder output
        - Masking specific key-value pairs from being attended to
    """
    if padding_mask_kv is None:
        mask = np.ones((n_q, n_kv), dtype=bool)
    else:
        mask = np.tile(padding_mask_kv, (n_q, 1))  # Broadcast to all queries
    return mask


def visualize_cross_attention(attention_weights, query_labels=None, kv_labels=None):
    """
    Visualize cross-attention weights as a heatmap.

    Parameters:
        attention_weights: np.ndarray of shape (n_q, n_kv)
        query_labels: optional list of labels for query positions
        kv_labels: optional list of labels for key/value positions
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')

    if query_labels is not None:
        plt.yticks(range(len(query_labels)), query_labels)
    if kv_labels is not None:
        plt.xticks(range(len(kv_labels)), kv_labels, rotation=45, ha='right')

    plt.xlabel('Key/Value Positions (Source)')
    plt.ylabel('Query Positions (Target)')
    plt.title('Cross-Attention Weights')
    plt.tight_layout()
    plt.savefig('cross_attention_viz.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Example: Decoder attending to encoder outputs
    np.random.seed(42)

    # Encoder output (e.g., source sentence with 8 tokens)
    n_encoder, d = 8, 32
    encoder_output = np.random.randn(n_encoder, d)

    # Decoder state (e.g., target sentence being generated, 5 tokens so far)
    n_decoder = 5
    decoder_state = np.random.randn(n_decoder, d)

    # Initialize projections
    n_heads = 4
    W_q = np.random.randn(d, d) * 0.02
    W_k = np.random.randn(d, d) * 0.02
    W_v = np.random.randn(d, d) * 0.02
    W_o = np.random.randn(d, d) * 0.02

    output, weights = cross_attention(
        decoder_state,  # Queries from decoder
        encoder_output,  # Keys/values from encoder
        W_q, W_k, W_v, W_o, n_heads
    )

    print(f"Decoder state shape: {decoder_state.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Cross-attention output shape: {output.shape}")
    print(f"Number of attention heads: {len(weights)}")
    if weights:
        print(f"Attention weights shape per head: {weights[0].shape}")
