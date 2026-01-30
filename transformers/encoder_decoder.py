import numpy as np
from .attention import multi_head_attention, create_causal_mask
from .cross_attention import cross_attention
from .transformer_block import layer_norm, feed_forward_network
from .positional_encoding import sinusoidal_positional_encoding


def encoder_block(x, params, n_heads):
    """
    Single Transformer Encoder Block.

    Structure:
        1. Multi-head self-attention + residual + LayerNorm
        2. Feed-forward network + residual + LayerNorm

    Parameters:
        x: np.ndarray of shape (n, d) - Input sequence
        params: dict containing attention and FFN parameters
        n_heads: int - Number of attention heads

    Returns:
        z: np.ndarray of shape (n, d) - Encoder block output

    Note: Encoder uses bidirectional attention (no causal mask).
    """
    n, d = x.shape

    # TODO: Implement encoder block
    #
    # Step 1: Self-attention (no mask - bidirectional)
    # attn_out, _ = multi_head_attention(x, params['W_q'], params['W_k'],
    #                                     params['W_v'], params['W_o'], n_heads)
    #
    # Step 2: Residual + LayerNorm
    # u = layer_norm(x + attn_out, params['gamma1'], params['beta1'])
    #
    # Step 3: FFN
    # ffn_out = feed_forward_network(u, params['W1'], params['b1'],
    #                                 params['W2'], params['b2'])
    #
    # Step 4: Residual + LayerNorm
    # z = layer_norm(u + ffn_out, params['gamma2'], params['beta2'])

    z = np.zeros_like(x)
    return z


def decoder_block(x, encoder_output, params, n_heads, causal_mask=None):
    """
    Single Transformer Decoder Block.

    Structure:
        1. Masked multi-head self-attention + residual + LayerNorm
        2. Multi-head cross-attention (to encoder) + residual + LayerNorm
        3. Feed-forward network + residual + LayerNorm

    Parameters:
        x: np.ndarray of shape (n_dec, d) - Decoder input sequence
        encoder_output: np.ndarray of shape (n_enc, d) - Encoder output to attend to
        params: dict containing all attention, cross-attention, and FFN parameters
        n_heads: int - Number of attention heads
        causal_mask: optional mask for self-attention (prevents attending to future)

    Returns:
        z: np.ndarray of shape (n_dec, d) - Decoder block output
    """
    n_dec, d = x.shape

    # TODO: Implement decoder block
    #
    # Step 1: Masked self-attention
    # self_attn_out, _ = multi_head_attention(x, params['W_q'], params['W_k'],
    #                                          params['W_v'], params['W_o'],
    #                                          n_heads, mask=causal_mask)
    #
    # Step 2: Residual + LayerNorm
    # u = layer_norm(x + self_attn_out, params['gamma1'], params['beta1'])
    #
    # Step 3: Cross-attention to encoder output
    # cross_attn_out, _ = cross_attention(u, encoder_output,
    #                                      params['W_q_cross'], params['W_k_cross'],
    #                                      params['W_v_cross'], params['W_o_cross'],
    #                                      n_heads)
    #
    # Step 4: Residual + LayerNorm
    # v = layer_norm(u + cross_attn_out, params['gamma2'], params['beta2'])
    #
    # Step 5: FFN
    # ffn_out = feed_forward_network(v, params['W1'], params['b1'],
    #                                 params['W2'], params['b2'])
    #
    # Step 6: Residual + LayerNorm
    # z = layer_norm(v + ffn_out, params['gamma3'], params['beta3'])

    z = np.zeros_like(x)
    return z


def encoder(x, blocks_params, n_heads):
    """
    Full Transformer Encoder (stack of encoder blocks).

    Parameters:
        x: np.ndarray of shape (n, d) - Input sequence (with positional encoding)
        blocks_params: list of parameter dicts for each encoder block
        n_heads: int - Number of attention heads

    Returns:
        encoder_output: np.ndarray of shape (n, d) - Encoded representation
    """
    # TODO: Implement encoder
    #
    # z = x
    # for params in blocks_params:
    #     z = encoder_block(z, params, n_heads)
    # return z

    return x


def decoder(x, encoder_output, blocks_params, n_heads):
    """
    Full Transformer Decoder (stack of decoder blocks).

    Parameters:
        x: np.ndarray of shape (n_dec, d) - Decoder input (with positional encoding)
        encoder_output: np.ndarray of shape (n_enc, d) - Encoder output
        blocks_params: list of parameter dicts for each decoder block
        n_heads: int - Number of attention heads

    Returns:
        decoder_output: np.ndarray of shape (n_dec, d) - Decoded representation
    """
    n_dec = x.shape[0]
    causal_mask = create_causal_mask(n_dec)

    # TODO: Implement decoder
    #
    # z = x
    # for params in blocks_params:
    #     z = decoder_block(z, encoder_output, params, n_heads, causal_mask)
    # return z

    return x


def encoder_decoder_transformer(src, tgt, encoder_params, decoder_params,
                                 embed_params, n_heads, max_len=512):
    """
    Full Encoder-Decoder Transformer (e.g., for sequence-to-sequence tasks).

    Parameters:
        src: np.ndarray of shape (n_src,) - Source token indices
        tgt: np.ndarray of shape (n_tgt,) - Target token indices (for teacher forcing)
        encoder_params: list of parameter dicts for encoder blocks
        decoder_params: list of parameter dicts for decoder blocks
        embed_params: dict containing:
            - src_embed: np.ndarray of shape (vocab_src, d) - Source embedding matrix
            - tgt_embed: np.ndarray of shape (vocab_tgt, d) - Target embedding matrix
            - output_proj: np.ndarray of shape (d, vocab_tgt) - Output projection
        n_heads: int - Number of attention heads
        max_len: int - Maximum sequence length for positional encoding

    Returns:
        logits: np.ndarray of shape (n_tgt, vocab_tgt) - Output logits

    Architecture:
        1. Embed source tokens + positional encoding
        2. Pass through encoder stack
        3. Embed target tokens + positional encoding
        4. Pass through decoder stack (with cross-attention to encoder output)
        5. Project to vocabulary size
    """
    d = embed_params['src_embed'].shape[1]

    # TODO: Implement full encoder-decoder transformer
    #
    # Step 1: Embed source and add positional encoding
    # src_embedded = embed_params['src_embed'][src]  # (n_src, d)
    # pe = sinusoidal_positional_encoding(max_len, d)
    # src_embedded = src_embedded + pe[:len(src)]
    #
    # Step 2: Encode
    # encoder_output = encoder(src_embedded, encoder_params, n_heads)
    #
    # Step 3: Embed target and add positional encoding
    # tgt_embedded = embed_params['tgt_embed'][tgt]  # (n_tgt, d)
    # tgt_embedded = tgt_embedded + pe[:len(tgt)]
    #
    # Step 4: Decode
    # decoder_output = decoder(tgt_embedded, encoder_output, decoder_params, n_heads)
    #
    # Step 5: Project to vocabulary
    # logits = decoder_output @ embed_params['output_proj']  # (n_tgt, vocab_tgt)

    n_tgt = len(tgt)
    vocab_tgt = embed_params['output_proj'].shape[1]
    logits = np.zeros((n_tgt, vocab_tgt))

    return logits


def init_encoder_block_params(d, d_ff, n_heads, seed=None):
    """Initialize parameters for an encoder block."""
    if seed is not None:
        np.random.seed(seed)

    scale = 0.02
    return {
        # Self-attention
        'W_q': np.random.randn(d, d) * scale,
        'W_k': np.random.randn(d, d) * scale,
        'W_v': np.random.randn(d, d) * scale,
        'W_o': np.random.randn(d, d) * scale,
        'gamma1': np.ones(d),
        'beta1': np.zeros(d),
        # FFN
        'W1': np.random.randn(d, d_ff) * scale,
        'b1': np.zeros(d_ff),
        'W2': np.random.randn(d_ff, d) * scale,
        'b2': np.zeros(d),
        'gamma2': np.ones(d),
        'beta2': np.zeros(d),
    }


def init_decoder_block_params(d, d_ff, n_heads, seed=None):
    """Initialize parameters for a decoder block."""
    if seed is not None:
        np.random.seed(seed)

    scale = 0.02
    return {
        # Self-attention
        'W_q': np.random.randn(d, d) * scale,
        'W_k': np.random.randn(d, d) * scale,
        'W_v': np.random.randn(d, d) * scale,
        'W_o': np.random.randn(d, d) * scale,
        'gamma1': np.ones(d),
        'beta1': np.zeros(d),
        # Cross-attention
        'W_q_cross': np.random.randn(d, d) * scale,
        'W_k_cross': np.random.randn(d, d) * scale,
        'W_v_cross': np.random.randn(d, d) * scale,
        'W_o_cross': np.random.randn(d, d) * scale,
        'gamma2': np.ones(d),
        'beta2': np.zeros(d),
        # FFN
        'W1': np.random.randn(d, d_ff) * scale,
        'b1': np.zeros(d_ff),
        'W2': np.random.randn(d_ff, d) * scale,
        'b2': np.zeros(d),
        'gamma3': np.ones(d),
        'beta3': np.zeros(d),
    }


if __name__ == "__main__":
    # Example: Simple sequence-to-sequence setup
    np.random.seed(42)

    # Hyperparameters
    d, d_ff, n_heads = 64, 256, 8
    n_encoder_layers, n_decoder_layers = 2, 2
    vocab_src, vocab_tgt = 1000, 1000

    # Initialize parameters
    encoder_params = [init_encoder_block_params(d, d_ff, n_heads, seed=i)
                      for i in range(n_encoder_layers)]
    decoder_params = [init_decoder_block_params(d, d_ff, n_heads, seed=i + 100)
                      for i in range(n_decoder_layers)]
    embed_params = {
        'src_embed': np.random.randn(vocab_src, d) * 0.02,
        'tgt_embed': np.random.randn(vocab_tgt, d) * 0.02,
        'output_proj': np.random.randn(d, vocab_tgt) * 0.02,
    }

    # Example input
    src_tokens = np.array([5, 23, 102, 45, 67])  # Source sentence
    tgt_tokens = np.array([1, 15, 88, 32])        # Target sentence (teacher forcing)

    logits = encoder_decoder_transformer(
        src_tokens, tgt_tokens,
        encoder_params, decoder_params, embed_params, n_heads
    )

    print(f"Source length: {len(src_tokens)}")
    print(f"Target length: {len(tgt_tokens)}")
    print(f"Output logits shape: {logits.shape}")  # Should be (4, 1000)
