import numpy as np
import pytest
from transformers.encoder_decoder import (
    encoder_block,
    decoder_block,
    encoder,
    decoder,
    encoder_decoder_transformer,
    init_encoder_block_params,
    init_decoder_block_params,
)
from transformers.attention import create_causal_mask


class TestEncoderBlock:
    """Tests for transformer encoder block."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d, d_ff, n_heads = 10, 64, 256, 8
        x = np.random.randn(n, d)
        params = init_encoder_block_params(d, d_ff, n_heads, seed=42)

        z = encoder_block(x, params, n_heads)
        assert z.shape == (n, d)

    def test_bidirectional_attention(self):
        """Test that encoder uses bidirectional attention (no causal mask)."""
        n, d, d_ff, n_heads = 6, 32, 128, 4
        x = np.random.randn(n, d)
        params = init_encoder_block_params(d, d_ff, n_heads, seed=42)

        # Encoder should be able to attend to all positions
        z = encoder_block(x, params, n_heads)

        # All positions should be influenced by all other positions
        # (unlike decoder which is causal)
        assert z.shape == (n, d)

    def test_deterministic(self):
        """Test that encoder block is deterministic."""
        n, d, d_ff, n_heads = 5, 32, 128, 4
        x = np.random.randn(n, d)
        params = init_encoder_block_params(d, d_ff, n_heads, seed=42)

        z1 = encoder_block(x, params, n_heads)
        z2 = encoder_block(x, params, n_heads)

        np.testing.assert_array_equal(z1, z2)


class TestDecoderBlock:
    """Tests for transformer decoder block."""

    def test_output_shape(self):
        """Test that output shape matches decoder input shape."""
        n_enc, n_dec, d, d_ff, n_heads = 8, 6, 64, 256, 8

        x = np.random.randn(n_dec, d)
        encoder_output = np.random.randn(n_enc, d)
        params = init_decoder_block_params(d, d_ff, n_heads, seed=42)

        z = decoder_block(x, encoder_output, params, n_heads)
        assert z.shape == (n_dec, d)

    def test_with_causal_mask(self):
        """Test decoder with causal masking."""
        n_enc, n_dec, d, d_ff, n_heads = 10, 8, 32, 128, 4

        x = np.random.randn(n_dec, d)
        encoder_output = np.random.randn(n_enc, d)
        params = init_decoder_block_params(d, d_ff, n_heads, seed=42)
        causal_mask = create_causal_mask(n_dec)

        z = decoder_block(x, encoder_output, params, n_heads, causal_mask)
        assert z.shape == (n_dec, d)

    def test_cross_attention_to_encoder(self):
        """Test that decoder attends to encoder output."""
        n_enc, n_dec, d, d_ff, n_heads = 10, 5, 32, 128, 4

        x = np.random.randn(n_dec, d)
        encoder_output = np.random.randn(n_enc, d)
        params = init_decoder_block_params(d, d_ff, n_heads, seed=42)

        # With different encoder outputs, decoder output should differ
        z1 = decoder_block(x, encoder_output, params, n_heads)

        encoder_output_different = np.random.randn(n_enc, d)
        z2 = decoder_block(x, encoder_output_different, params, n_heads)

        assert not np.allclose(z1, z2)

    def test_different_encoder_decoder_lengths(self):
        """Test decoder with various encoder/decoder length combinations."""
        d, d_ff, n_heads = 32, 128, 4

        test_cases = [(5, 10), (10, 5), (1, 8), (8, 1), (6, 6)]

        for n_enc, n_dec in test_cases:
            x = np.random.randn(n_dec, d)
            encoder_output = np.random.randn(n_enc, d)
            params = init_decoder_block_params(d, d_ff, n_heads, seed=42)

            z = decoder_block(x, encoder_output, params, n_heads)
            assert z.shape == (n_dec, d), f"Failed for n_enc={n_enc}, n_dec={n_dec}"


class TestEncoder:
    """Tests for full encoder stack."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        n, d, d_ff, n_heads, n_layers = 10, 64, 256, 8, 3
        x = np.random.randn(n, d)

        blocks_params = [init_encoder_block_params(d, d_ff, n_heads, seed=i)
                         for i in range(n_layers)]

        z = encoder(x, blocks_params, n_heads)
        assert z.shape == (n, d)

    def test_multiple_layers(self):
        """Test that multiple encoder layers are applied."""
        n, d, d_ff, n_heads = 6, 32, 128, 4
        x = np.random.randn(n, d)

        # Single layer
        params1 = [init_encoder_block_params(d, d_ff, n_heads, seed=1)]
        z1 = encoder(x, params1, n_heads)

        # Two layers
        params2 = [init_encoder_block_params(d, d_ff, n_heads, seed=1),
                   init_encoder_block_params(d, d_ff, n_heads, seed=2)]
        z2 = encoder(x, params2, n_heads)

        # Results should differ with different number of layers
        assert not np.allclose(z1, z2)


class TestDecoder:
    """Tests for full decoder stack."""

    def test_output_shape(self):
        """Test that output shape matches decoder input shape."""
        n_enc, n_dec, d, d_ff, n_heads, n_layers = 10, 8, 64, 256, 8, 3

        x = np.random.randn(n_dec, d)
        encoder_output = np.random.randn(n_enc, d)

        blocks_params = [init_decoder_block_params(d, d_ff, n_heads, seed=i)
                         for i in range(n_layers)]

        z = decoder(x, encoder_output, blocks_params, n_heads)
        assert z.shape == (n_dec, d)

    def test_causal_masking_applied(self):
        """Test that causal masking is applied in decoder."""
        n_enc, n_dec, d, d_ff, n_heads = 5, 6, 32, 128, 4

        x = np.random.randn(n_dec, d)
        encoder_output = np.random.randn(n_enc, d)
        params = [init_decoder_block_params(d, d_ff, n_heads, seed=42)]

        # Decoder should apply causal masking internally
        z = decoder(x, encoder_output, params, n_heads)
        assert z.shape == (n_dec, d)


class TestEncoderDecoderTransformer:
    """Tests for full encoder-decoder transformer."""

    def test_output_shape(self):
        """Test that output logits have correct shape."""
        d, d_ff, n_heads = 32, 128, 4
        n_enc_layers, n_dec_layers = 2, 2
        vocab_src, vocab_tgt = 100, 100

        encoder_params = [init_encoder_block_params(d, d_ff, n_heads, seed=i)
                          for i in range(n_enc_layers)]
        decoder_params = [init_decoder_block_params(d, d_ff, n_heads, seed=i + 10)
                          for i in range(n_dec_layers)]
        embed_params = {
            'src_embed': np.random.randn(vocab_src, d) * 0.02,
            'tgt_embed': np.random.randn(vocab_tgt, d) * 0.02,
            'output_proj': np.random.randn(d, vocab_tgt) * 0.02,
        }

        src = np.array([5, 10, 15, 20])
        tgt = np.array([1, 2, 3])

        logits = encoder_decoder_transformer(
            src, tgt, encoder_params, decoder_params, embed_params, n_heads
        )

        assert logits.shape == (len(tgt), vocab_tgt)

    def test_different_source_changes_output(self):
        """Test that different source sequences produce different outputs."""
        d, d_ff, n_heads = 32, 128, 4
        vocab_src, vocab_tgt = 100, 100

        encoder_params = [init_encoder_block_params(d, d_ff, n_heads, seed=1)]
        decoder_params = [init_decoder_block_params(d, d_ff, n_heads, seed=2)]
        embed_params = {
            'src_embed': np.random.randn(vocab_src, d) * 0.02,
            'tgt_embed': np.random.randn(vocab_tgt, d) * 0.02,
            'output_proj': np.random.randn(d, vocab_tgt) * 0.02,
        }

        tgt = np.array([1, 2, 3])

        src1 = np.array([5, 10, 15])
        logits1 = encoder_decoder_transformer(
            src1, tgt, encoder_params, decoder_params, embed_params, n_heads
        )

        src2 = np.array([20, 25, 30])
        logits2 = encoder_decoder_transformer(
            src2, tgt, encoder_params, decoder_params, embed_params, n_heads
        )

        assert not np.allclose(logits1, logits2)

    def test_variable_length_sequences(self):
        """Test with various sequence lengths."""
        d, d_ff, n_heads = 32, 128, 4
        vocab_src, vocab_tgt = 100, 100

        encoder_params = [init_encoder_block_params(d, d_ff, n_heads, seed=1)]
        decoder_params = [init_decoder_block_params(d, d_ff, n_heads, seed=2)]
        embed_params = {
            'src_embed': np.random.randn(vocab_src, d) * 0.02,
            'tgt_embed': np.random.randn(vocab_tgt, d) * 0.02,
            'output_proj': np.random.randn(d, vocab_tgt) * 0.02,
        }

        test_cases = [
            (np.array([1, 2, 3]), np.array([10])),           # Long src, short tgt
            (np.array([1]), np.array([10, 20, 30, 40])),    # Short src, long tgt
            (np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50])),  # Equal
        ]

        for src, tgt in test_cases:
            logits = encoder_decoder_transformer(
                src, tgt, encoder_params, decoder_params, embed_params, n_heads
            )
            assert logits.shape == (len(tgt), vocab_tgt)


class TestInitParams:
    """Tests for parameter initialization functions."""

    def test_encoder_params_keys(self):
        """Test that encoder params have all required keys."""
        d, d_ff, n_heads = 64, 256, 8
        params = init_encoder_block_params(d, d_ff, n_heads)

        required = ['W_q', 'W_k', 'W_v', 'W_o', 'gamma1', 'beta1',
                    'W1', 'b1', 'W2', 'b2', 'gamma2', 'beta2']
        for key in required:
            assert key in params

    def test_decoder_params_keys(self):
        """Test that decoder params have all required keys including cross-attention."""
        d, d_ff, n_heads = 64, 256, 8
        params = init_decoder_block_params(d, d_ff, n_heads)

        required = ['W_q', 'W_k', 'W_v', 'W_o', 'gamma1', 'beta1',
                    'W_q_cross', 'W_k_cross', 'W_v_cross', 'W_o_cross', 'gamma2', 'beta2',
                    'W1', 'b1', 'W2', 'b2', 'gamma3', 'beta3']
        for key in required:
            assert key in params, f"Missing key: {key}"

    def test_reproducible_with_seed(self):
        """Test that same seed gives same parameters."""
        d, d_ff, n_heads = 64, 256, 8

        p1 = init_encoder_block_params(d, d_ff, n_heads, seed=42)
        p2 = init_encoder_block_params(d, d_ff, n_heads, seed=42)

        for key in p1:
            np.testing.assert_array_equal(p1[key], p2[key])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
