"""
Comprehensive Tests for Sequence Module (Transformers)
======================================================

Tests for all transformer architectures: Encoder, Decoder, Encoder-Decoder,
BERT, GPT, ViT, CLIP, DETR, and Switch Transformer.

Sections:
    A. Imports, Helpers, Model Registry
    B. Encoder (Bidirectional) Tests
    C. Decoder (Autoregressive) Tests
    D. Encoder-Decoder (Seq2Seq) Tests
    E. BERT Tests
    F. GPT Tests
    G. Vision Transformer (ViT) Tests
    H. CLIP Tests
    I. DETR Tests
    J. Switch Transformer Tests
    K. Cross-Architecture Tests
    L. Inference Benchmarks (slow)
"""

import pytest
import numpy as np
import time
import json
from pathlib import Path

from python.foundations import Tensor, no_grad
from python.optimization.optimizers import SGD
from python.optimization.losses import CrossEntropyLoss

# Encoder (bidirectional)
from python.sequence.transformers.encoder import (
    EncoderLayer,
    TransformerEncoder,
    BERT_CONFIG,
    BERT_LARGE_CONFIG,
    ROBERTA_CONFIG,
)

# Decoder (autoregressive)
from python.sequence.transformers.decoder import (
    DecoderLayer,
    TransformerDecoder,
    GPT2_SMALL_CONFIG,
    GPT2_MEDIUM_CONFIG,
    GPT2_LARGE_CONFIG,
    GPT2_XL_CONFIG,
)

# Encoder-Decoder (seq2seq)
from python.sequence.transformers.encoder_decoder import (
    DecoderLayerWithCrossAttention,
    TransformerEncoderDecoder,
    TRANSFORMER_BASE_CONFIG,
    T5_BASE_CONFIG,
    T5_LARGE_CONFIG,
)

# BERT
from python.sequence.transformers.bert import (
    BertEmbeddings,
    BertModel,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForMaskedLM,
    BERT_BASE_CONFIG,
    BERT_LARGE_CONFIG as BERT_LARGE_CONFIG_FULL,
    ROBERTA_BASE_CONFIG,
)

# GPT
from python.sequence.transformers.gpt import (
    GPT,
    GPT2_SMALL_CONFIG as GPT2_SMALL,
    GPT2_MEDIUM_CONFIG as GPT2_MEDIUM,
    GPT3_SMALL_CONFIG,
    GPT3_MEDIUM_CONFIG,
)

# Vision Transformer
from python.sequence.transformers.vit import (
    PatchEmbedding,
    VisionTransformer,
    VIT_BASE_CONFIG,
    VIT_LARGE_CONFIG,
    VIT_HUGE_CONFIG,
    DEIT_BASE_CONFIG,
    DEIT_SMALL_CONFIG,
)

# DETR
from python.sequence.transformers.detr import (
    DETREncoder,
    DETRDecoder,
    DETR,
    DETR_RESNET50_CONFIG,
    DETR_RESNET101_CONFIG,
)

# Switch Transformer
from python.sequence.transformers.switch_transformer import (
    SwitchTransformerLayer,
    SwitchTransformer,
    SWITCH_BASE_8_CONFIG,
    SWITCH_BASE_64_CONFIG,
    SWITCH_LARGE_128_CONFIG,
)


# =============================================================================
# Section A: Helpers and Fixtures
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


def make_token_ids(batch, seq_len, vocab_size=1000):
    """Create random token ID Tensor."""
    data = np.random.randint(0, vocab_size, (batch, seq_len))
    return Tensor(data)


def make_image(batch, channels=3, h=32, w=32):
    """Create a random image Tensor."""
    data = np.random.randn(batch, channels, h, w).astype(np.float64) * 0.1
    return Tensor(data, requires_grad=False)


def make_hidden(batch, seq_len, d_model):
    """Create a random hidden state Tensor (for layer-level tests)."""
    data = np.random.randn(batch, seq_len, d_model).astype(np.float64) * 0.1
    return Tensor(data, requires_grad=True)


def make_padding_mask(batch, seq_len, pad_fraction=0.2):
    """Create a padding mask Tensor. True=valid, False=pad."""
    mask = np.ones((batch, seq_len), dtype=bool)
    n_pad = int(seq_len * pad_fraction)
    if n_pad > 0:
        mask[:, -n_pad:] = False
    return Tensor(mask)


def make_token_type_ids(batch, seq_len, split=None):
    """Create token type IDs Tensor (0s and 1s for sentence A/B)."""
    data = np.zeros((batch, seq_len), dtype=np.int64)
    if split is not None:
        data[:, split:] = 1
    return Tensor(data)


def make_attention_mask(batch, seq_len, n_pad=0):
    """Create attention mask Tensor (1=token, 0=pad)."""
    data = np.ones((batch, seq_len), dtype=np.int64)
    if n_pad > 0:
        data[:, -n_pad:] = 0
    return Tensor(data)


# Small configs for fast testing (minimize d_model, layers, etc.)
SMALL_ENCODER_CFG = dict(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                         vocab_size=500, max_seq_len=32, dropout=0.0)
SMALL_DECODER_CFG = dict(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                         vocab_size=500, max_seq_len=32, dropout=0.0)
SMALL_ENC_DEC_CFG = dict(d_model=64, num_heads=4, num_encoder_layers=2,
                         num_decoder_layers=2, d_ff=128, src_vocab_size=500,
                         tgt_vocab_size=500, max_src_len=32, max_tgt_len=32,
                         dropout=0.0)
SMALL_BERT_CFG = dict(vocab_size=500, d_model=64, num_heads=4, num_layers=2,
                      d_ff=128, max_seq_len=32, dropout=0.0)
SMALL_GPT_CFG = dict(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                     vocab_size=500, max_seq_len=32, dropout=0.0)
SMALL_VIT_CFG = dict(img_size=32, patch_size=8, in_channels=3, num_classes=10,
                     d_model=64, num_heads=4, num_layers=2, d_ff=128, dropout=0.0)
SMALL_CLIP_VISION_CFG = dict(img_size=32, patch_size=8, d_model=64,
                             num_heads=4, num_layers=2, d_ff=128)
SMALL_CLIP_TEXT_CFG = dict(vocab_size=500, d_model=64, num_heads=4,
                          num_layers=2, d_ff=128, max_seq_len=16)
SMALL_DETR_CFG = dict(num_classes=10, d_model=64, num_heads=4,
                      num_encoder_layers=2, num_decoder_layers=2, d_ff=128,
                      num_queries=10, dropout=0.0, backbone_channels=64)
SMALL_SWITCH_CFG = dict(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                        num_experts=4, vocab_size=500, max_seq_len=32,
                        dropout=0.0)


# =============================================================================
# Section B: Encoder (Bidirectional) Tests
# =============================================================================

class TestEncoderLayer:
    """Tests for individual EncoderLayer."""

    def test_forward_shape(self):
        """Output shape should match input: [batch, seq_len, d_model]."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        x = make_hidden(2, 16, 64)
        out = layer(x)
        assert out.data.shape == (2, 16, 64)

    def test_forward_with_padding_mask(self):
        """Should accept and respect a padding mask."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        x = make_hidden(2, 16, 64)
        mask = make_padding_mask(2, 16)
        out = layer(x, padding_mask=mask)
        assert out.data.shape == (2, 16, 64)

    def test_output_finite(self):
        """All outputs should be finite."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        x = make_hidden(2, 8, 64)
        out = layer(x)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        """Backward should produce gradients for all parameters."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        x = make_hidden(2, 8, 64)
        out = layer(x)
        out.sum().backward()
        params = list(layer.parameters())
        assert len(params) > 0
        grads_exist = sum(1 for p in params if p.grad is not None)
        assert grads_exist > 0

    def test_gradient_shapes_match(self):
        """Gradient shapes must match parameter shapes."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        x = make_hidden(2, 8, 64)
        out = layer(x)
        out.sum().backward()
        for p in layer.parameters():
            if p.grad is not None:
                assert p.grad.shape == p.data.shape

    def test_gelu_activation(self):
        """Should support GELU activation."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128, activation="gelu")
        x = make_hidden(1, 8, 64)
        out = layer(x)
        assert out.data.shape == (1, 8, 64)

    def test_relu_activation(self):
        """Should support ReLU activation."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128, activation="relu")
        x = make_hidden(1, 8, 64)
        out = layer(x)
        assert out.data.shape == (1, 8, 64)

    def test_different_d_ff(self):
        """Should handle custom feed-forward dimension."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=256, dropout=0.0)
        x = make_hidden(1, 8, 64)
        out = layer(x)
        assert out.data.shape == (1, 8, 64)

    def test_bidirectional_attention(self):
        """Encoder is bidirectional: changing a later token should affect earlier outputs."""
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        layer.eval()
        x1 = make_hidden(1, 8, 64)
        out1 = layer(x1)

        # Modify the last token
        x2_data = x1.data.copy()
        x2_data[0, -1, :] += 10.0
        x2 = Tensor(x2_data, requires_grad=True)
        out2 = layer(x2)

        # First token output should differ (bidirectional can see the change)
        assert not np.allclose(out1.data[0, 0], out2.data[0, 0], atol=1e-7), \
            "Encoder should be bidirectional: changing last token must affect first token"


class TestTransformerEncoder:
    """Tests for the full TransformerEncoder stack."""

    def test_forward_shape(self):
        """Output: [batch, seq_len, d_model]."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        out = model(ids)
        assert out.data.shape == (2, 16, 64)

    def test_forward_with_padding_mask(self):
        """Should accept padding mask."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        mask = make_padding_mask(2, 16)
        out = model(ids, padding_mask=mask)
        assert out.data.shape == (2, 16, 64)

    def test_output_finite(self):
        """All outputs should be finite."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = model(ids)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        """Backward pass should produce gradients."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = model(ids)
        out.sum().backward()
        params = list(model.parameters())
        grads_exist = sum(1 for p in params if p.grad is not None)
        assert grads_exist > 0

    def test_gradients_finite(self):
        """All gradients should be finite."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = model(ids)
        out.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad))

    def test_parameter_count_positive(self):
        """Model should have parameters."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        params = list(model.parameters())
        assert len(params) > 0
        total = sum(p.data.size for p in params)
        assert total > 0

    def test_different_sequence_lengths(self):
        """Should handle varying sequence lengths."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        for seq_len in [4, 8, 16, 32]:
            ids = make_token_ids(1, seq_len, vocab_size=500)
            out = model(ids)
            assert out.data.shape == (1, seq_len, 64), f"Failed for seq_len={seq_len}"

    def test_batch_independence(self):
        """Each sample in a batch should be processed independently."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        model.eval()
        ids1 = make_token_ids(1, 8, vocab_size=500)
        ids2 = make_token_ids(1, 8, vocab_size=500)
        ids_batch = Tensor(np.concatenate([ids1.data, ids2.data], axis=0))

        out_single1 = model(ids1)
        out_single2 = model(ids2)
        out_batch = model(ids_batch)

        np.testing.assert_allclose(out_batch.data[0], out_single1.data[0], atol=1e-5)
        np.testing.assert_allclose(out_batch.data[1], out_single2.data[0], atol=1e-5)

    def test_eval_determinism(self):
        """Eval mode should produce deterministic outputs."""
        model = TransformerEncoder(**SMALL_ENCODER_CFG)
        model.eval()
        ids = make_token_ids(2, 8, vocab_size=500)
        out1 = model(ids)
        out2 = model(ids)
        np.testing.assert_array_equal(out1.data, out2.data)

    def test_sinusoidal_pos_embed(self):
        """Should support sinusoidal positional embeddings."""
        cfg = {**SMALL_ENCODER_CFG, "use_learnable_pos_embed": False}
        model = TransformerEncoder(**cfg)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = model(ids)
        assert out.data.shape == (2, 8, 64)

    def test_configs_are_valid_dicts(self):
        """All config dicts should have required keys."""
        for cfg in [BERT_CONFIG, BERT_LARGE_CONFIG, ROBERTA_CONFIG]:
            assert "d_model" in cfg
            assert "num_heads" in cfg
            assert "num_layers" in cfg
            assert "vocab_size" in cfg


# =============================================================================
# Section C: Decoder (Autoregressive) Tests
# =============================================================================

class TestDecoderLayer:
    """Tests for individual DecoderLayer."""

    def test_forward_shape(self):
        """Output shape should match input."""
        layer = DecoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        x = make_hidden(2, 16, 64)
        out = layer(x)
        assert out.data.shape == (2, 16, 64)

    def test_output_finite(self):
        layer = DecoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        x = make_hidden(2, 8, 64)
        out = layer(x)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        layer = DecoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        x = make_hidden(2, 8, 64)
        out = layer(x)
        out.sum().backward()
        grads_exist = sum(1 for p in layer.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_causal_masking(self):
        """Decoder is causal: changing a later token should NOT affect earlier outputs."""
        layer = DecoderLayer(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        layer.eval()
        x1 = make_hidden(1, 8, 64)
        out1 = layer(x1)

        # Modify the last token
        x2_data = x1.data.copy()
        x2_data[0, -1, :] += 10.0
        x2 = Tensor(x2_data, requires_grad=True)
        out2 = layer(x2)

        # First token output should be the same (causal: can't see the future)
        np.testing.assert_allclose(out1.data[0, 0], out2.data[0, 0], atol=1e-5,
            err_msg="Decoder should be causal: changing last token must not affect first token")


class TestTransformerDecoder:
    """Tests for the full TransformerDecoder stack."""

    def test_forward_shape(self):
        """Output logits: [batch, seq_len, vocab_size]."""
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        out = model(ids)
        assert out.data.shape == (2, 16, 500)

    def test_output_finite(self):
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = model(ids)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = model(ids)
        out.sum().backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_gradients_finite(self):
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = model(ids)
        out.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad))

    def test_different_sequence_lengths(self):
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        for seq_len in [4, 8, 16]:
            ids = make_token_ids(1, seq_len, vocab_size=500)
            out = model(ids)
            assert out.data.shape == (1, seq_len, 500)

    def test_causal_property(self):
        """Logits at position i should depend only on tokens 0..i."""
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        model.eval()

        ids = make_token_ids(1, 8, vocab_size=500)
        out1 = model(ids)

        # Change token at position 7 — should not affect logits at positions 0-6
        ids2_data = ids.data.copy()
        ids2_data[0, 7] = (ids2_data[0, 7] + 1) % 500
        ids2 = Tensor(ids2_data)
        out2 = model(ids2)

        np.testing.assert_allclose(out1.data[0, :7], out2.data[0, :7], atol=1e-5,
            err_msg="Decoder must be causal: future tokens should not affect past logits")

    def test_generate_shape(self):
        """Generate should produce extended Tensor sequence."""
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        model.eval()
        prompt = make_token_ids(1, 4, vocab_size=500)
        generated = model.generate(prompt, max_length=10)
        assert generated.data.shape[0] == 1
        assert generated.data.shape[1] == 4 + 10  # prompt + generated

    def test_generate_valid_token_ids(self):
        """Generated tokens should be valid vocab indices."""
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        model.eval()
        prompt = make_token_ids(1, 4, vocab_size=500)
        generated = model.generate(prompt, max_length=5)
        assert np.all(generated.data >= 0)
        assert np.all(generated.data < 500)

    def test_generate_temperature(self):
        """Temperature=0 should give near-deterministic output (greedy)."""
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        model.eval()
        prompt = make_token_ids(1, 4, vocab_size=500)
        gen1 = model.generate(prompt, max_length=5, temperature=0.01)
        gen2 = model.generate(prompt, max_length=5, temperature=0.01)
        np.testing.assert_array_equal(gen1.data, gen2.data)

    def test_eval_determinism(self):
        model = TransformerDecoder(**SMALL_DECODER_CFG)
        model.eval()
        ids = make_token_ids(2, 8, vocab_size=500)
        out1 = model(ids)
        out2 = model(ids)
        np.testing.assert_array_equal(out1.data, out2.data)

    def test_configs_are_valid_dicts(self):
        for cfg in [GPT2_SMALL_CONFIG, GPT2_MEDIUM_CONFIG, GPT2_LARGE_CONFIG, GPT2_XL_CONFIG]:
            assert "d_model" in cfg
            assert "num_heads" in cfg
            assert "vocab_size" in cfg


# =============================================================================
# Section D: Encoder-Decoder (Seq2Seq) Tests
# =============================================================================

class TestDecoderLayerWithCrossAttention:
    """Tests for DecoderLayerWithCrossAttention."""

    def test_forward_shape(self):
        layer = DecoderLayerWithCrossAttention(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        dec_in = make_hidden(2, 10, 64)
        enc_out = make_hidden(2, 16, 64)
        out = layer(dec_in, enc_out)
        assert out.data.shape == (2, 10, 64)

    def test_cross_attention_uses_encoder_output(self):
        """Output should change when encoder output changes (proves cross-attention works)."""
        layer = DecoderLayerWithCrossAttention(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        layer.eval()
        dec_in = make_hidden(1, 8, 64)

        enc_out1 = make_hidden(1, 12, 64)
        out1 = layer(dec_in, enc_out1)

        enc_out2_data = enc_out1.data.copy()
        enc_out2_data += 5.0
        enc_out2 = Tensor(enc_out2_data, requires_grad=True)
        out2 = layer(dec_in, enc_out2)

        assert not np.allclose(out1.data, out2.data, atol=1e-5), \
            "Cross-attention output must depend on encoder output"

    def test_backward_produces_gradients(self):
        layer = DecoderLayerWithCrossAttention(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        dec_in = make_hidden(2, 8, 64)
        enc_out = make_hidden(2, 12, 64)
        out = layer(dec_in, enc_out)
        out.sum().backward()
        grads_exist = sum(1 for p in layer.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_different_src_tgt_lengths(self):
        """Source and target can have different sequence lengths."""
        layer = DecoderLayerWithCrossAttention(d_model=64, num_heads=4, d_ff=128, dropout=0.0)
        for tgt_len, src_len in [(4, 16), (16, 4), (8, 8)]:
            dec_in = make_hidden(1, tgt_len, 64)
            enc_out = make_hidden(1, src_len, 64)
            out = layer(dec_in, enc_out)
            assert out.data.shape == (1, tgt_len, 64)


class TestTransformerEncoderDecoder:
    """Tests for the full Encoder-Decoder model."""

    def test_forward_shape(self):
        """Output logits: [batch, tgt_len, tgt_vocab_size]."""
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        src = make_token_ids(2, 12, vocab_size=500)
        tgt = make_token_ids(2, 8, vocab_size=500)
        out = model(src, tgt)
        assert out.data.shape == (2, 8, 500)

    def test_output_finite(self):
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        src = make_token_ids(2, 8, vocab_size=500)
        tgt = make_token_ids(2, 6, vocab_size=500)
        out = model(src, tgt)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        src = make_token_ids(2, 8, vocab_size=500)
        tgt = make_token_ids(2, 6, vocab_size=500)
        out = model(src, tgt)
        out.sum().backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_gradients_finite(self):
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        src = make_token_ids(2, 8, vocab_size=500)
        tgt = make_token_ids(2, 6, vocab_size=500)
        out = model(src, tgt)
        out.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad))

    def test_encode_shape(self):
        """Encode should return [batch, src_len, d_model]."""
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        src = make_token_ids(2, 12, vocab_size=500)
        enc_out = model.encode(src)
        assert enc_out.data.shape == (2, 12, 64)

    def test_with_padding_masks(self):
        """Should accept src and tgt padding masks."""
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        src = make_token_ids(2, 12, vocab_size=500)
        tgt = make_token_ids(2, 8, vocab_size=500)
        src_mask = make_padding_mask(2, 12)
        tgt_mask = make_padding_mask(2, 8)
        out = model(src, tgt, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask)
        assert out.data.shape == (2, 8, 500)

    def test_different_src_tgt_lengths(self):
        """Source and target can have different lengths."""
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        for src_len, tgt_len in [(16, 4), (4, 16), (8, 8)]:
            src = make_token_ids(1, src_len, vocab_size=500)
            tgt = make_token_ids(1, tgt_len, vocab_size=500)
            out = model(src, tgt)
            assert out.data.shape == (1, tgt_len, 500)

    def test_generate_shape(self):
        """Generate should produce a target sequence."""
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        model.eval()
        src = make_token_ids(1, 8, vocab_size=500)
        generated = model.generate(src, max_length=10)
        assert generated.data.shape[0] == 1
        assert generated.data.shape[1] <= 10 + 1  # up to max_length + start token

    def test_generate_valid_token_ids(self):
        model = TransformerEncoderDecoder(**SMALL_ENC_DEC_CFG)
        model.eval()
        src = make_token_ids(1, 8, vocab_size=500)
        generated = model.generate(src, max_length=5)
        assert np.all(generated.data >= 0)
        assert np.all(generated.data < 500)

    def test_shared_embeddings(self):
        """With shared_embeddings=True, encoder/decoder should share weights."""
        cfg = {**SMALL_ENC_DEC_CFG, "share_embeddings": True}
        model = TransformerEncoderDecoder(**cfg)
        src = make_token_ids(2, 8, vocab_size=500)
        tgt = make_token_ids(2, 6, vocab_size=500)
        out = model(src, tgt)
        assert out.data.shape == (2, 6, 500)

    def test_configs_are_valid_dicts(self):
        for cfg in [TRANSFORMER_BASE_CONFIG, T5_BASE_CONFIG, T5_LARGE_CONFIG]:
            assert "d_model" in cfg
            assert "num_heads" in cfg
            assert "src_vocab_size" in cfg


# =============================================================================
# Section E: BERT Tests
# =============================================================================

class TestBertEmbeddings:
    """Tests for BertEmbeddings layer."""

    def test_forward_shape(self):
        emb = BertEmbeddings(vocab_size=500, d_model=64, max_seq_len=32)
        ids = make_token_ids(2, 16, vocab_size=500)
        out = emb(ids)
        assert out.data.shape == (2, 16, 64)

    def test_with_token_type_ids(self):
        emb = BertEmbeddings(vocab_size=500, d_model=64, max_seq_len=32)
        ids = make_token_ids(2, 16, vocab_size=500)
        token_types = make_token_type_ids(2, 16, split=8)
        out = emb(ids, token_type_ids=token_types)
        assert out.data.shape == (2, 16, 64)

    def test_segment_embeddings_differ(self):
        """Different token_type_ids should produce different embeddings."""
        emb = BertEmbeddings(vocab_size=500, d_model=64, max_seq_len=32)
        emb.eval()
        ids = make_token_ids(1, 8, vocab_size=500)
        types_a = make_token_type_ids(1, 8)
        types_b = Tensor(np.ones((1, 8), dtype=np.int64))
        out_a = emb(ids, token_type_ids=types_a)
        out_b = emb(ids, token_type_ids=types_b)
        assert not np.allclose(out_a.data, out_b.data, atol=1e-5), \
            "Different segment types should produce different embeddings"

    def test_output_finite(self):
        emb = BertEmbeddings(vocab_size=500, d_model=64, max_seq_len=32)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = emb(ids)
        assert np.all(np.isfinite(out.data))


class TestBertModel:
    """Tests for BertModel."""

    def test_forward_shapes(self):
        """Should return (hidden_states, pooled_output)."""
        model = BertModel(**SMALL_BERT_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        hidden, pooled = model(ids)
        assert hidden.data.shape == (2, 16, 64)
        assert pooled.data.shape == (2, 64)

    def test_output_finite(self):
        model = BertModel(**SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        hidden, pooled = model(ids)
        assert np.all(np.isfinite(hidden.data))
        assert np.all(np.isfinite(pooled.data))

    def test_backward_produces_gradients(self):
        model = BertModel(**SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        hidden, pooled = model(ids)
        pooled.sum().backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_gradients_finite(self):
        model = BertModel(**SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        hidden, pooled = model(ids)
        pooled.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad))

    def test_with_attention_mask(self):
        model = BertModel(**SMALL_BERT_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        mask = make_attention_mask(2, 16, n_pad=4)
        hidden, pooled = model(ids, attention_mask=mask)
        assert hidden.data.shape == (2, 16, 64)

    def test_with_token_type_ids(self):
        model = BertModel(**SMALL_BERT_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        types = make_token_type_ids(2, 16, split=8)
        hidden, pooled = model(ids, token_type_ids=types)
        assert hidden.data.shape == (2, 16, 64)

    def test_pooled_output_from_cls_token(self):
        """Pooled output should be derived from the [CLS] token (position 0)."""
        model = BertModel(**SMALL_BERT_CFG)
        model.eval()
        ids = make_token_ids(1, 8, vocab_size=500)
        hidden, pooled = model(ids)
        # pooled should be a transformation of hidden[:, 0, :]
        assert pooled.data.shape == (1, 64)
        # Can't be all zeros
        assert not np.allclose(pooled.data, 0.0)


class TestBertForSequenceClassification:
    """Tests for BertForSequenceClassification."""

    def test_forward_shape(self):
        model = BertForSequenceClassification(num_classes=5, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        logits = model(ids)
        assert logits.data.shape == (2, 5)

    def test_output_finite(self):
        model = BertForSequenceClassification(num_classes=3, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        assert np.all(np.isfinite(logits.data))

    def test_backward_produces_gradients(self):
        model = BertForSequenceClassification(num_classes=3, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        logits.sum().backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_different_num_classes(self):
        for nc in [2, 5, 10, 100]:
            model = BertForSequenceClassification(num_classes=nc, **SMALL_BERT_CFG)
            ids = make_token_ids(1, 8, vocab_size=500)
            logits = model(ids)
            assert logits.data.shape == (1, nc)

    def test_gradient_shapes(self):
        model = BertForSequenceClassification(num_classes=3, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        logits.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.shape == p.data.shape


class TestBertForTokenClassification:
    """Tests for BertForTokenClassification."""

    def test_forward_shape(self):
        model = BertForTokenClassification(num_classes=9, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        logits = model(ids)
        assert logits.data.shape == (2, 16, 9)

    def test_output_finite(self):
        model = BertForTokenClassification(num_classes=5, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        assert np.all(np.isfinite(logits.data))

    def test_backward_produces_gradients(self):
        model = BertForTokenClassification(num_classes=5, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        logits.sum().backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_different_num_classes(self):
        for nc in [2, 5, 17]:
            model = BertForTokenClassification(num_classes=nc, **SMALL_BERT_CFG)
            ids = make_token_ids(1, 8, vocab_size=500)
            logits = model(ids)
            assert logits.data.shape == (1, 8, nc)


class TestBertForMaskedLM:
    """Tests for BertForMaskedLM."""

    def test_forward_shape(self):
        model = BertForMaskedLM(vocab_size=500, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        logits = model(ids)
        assert logits.data.shape == (2, 16, 500)

    def test_output_finite(self):
        model = BertForMaskedLM(vocab_size=500, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        assert np.all(np.isfinite(logits.data))

    def test_backward_produces_gradients(self):
        model = BertForMaskedLM(vocab_size=500, **SMALL_BERT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        logits.sum().backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_configs_are_valid_dicts(self):
        for cfg in [BERT_BASE_CONFIG, BERT_LARGE_CONFIG_FULL, ROBERTA_BASE_CONFIG]:
            assert "vocab_size" in cfg
            assert "d_model" in cfg
            assert "num_layers" in cfg


# =============================================================================
# Section F: GPT Tests
# =============================================================================

class TestGPT:
    """Comprehensive tests for GPT language model."""

    def test_forward_shape(self):
        """Output logits: [batch, seq_len, vocab_size]."""
        model = GPT(**SMALL_GPT_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        logits = model(ids)
        assert logits.data.shape == (2, 16, 500)

    def test_output_finite(self):
        model = GPT(**SMALL_GPT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        assert np.all(np.isfinite(logits.data))

    def test_backward_produces_gradients(self):
        model = GPT(**SMALL_GPT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        logits.sum().backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_gradients_finite(self):
        model = GPT(**SMALL_GPT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits = model(ids)
        logits.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad))

    def test_causal_property(self):
        """Logits at position i should only depend on tokens 0..i."""
        model = GPT(**SMALL_GPT_CFG)
        model.eval()
        ids = make_token_ids(1, 8, vocab_size=500)
        out1 = model(ids)

        ids2_data = ids.data.copy()
        ids2_data[0, 7] = (ids2_data[0, 7] + 1) % 500
        ids2 = Tensor(ids2_data)
        out2 = model(ids2)

        np.testing.assert_allclose(out1.data[0, :7], out2.data[0, :7], atol=1e-5,
            err_msg="GPT must be causal")

    def test_generate_shape(self):
        model = GPT(**SMALL_GPT_CFG)
        model.eval()
        prompt = make_token_ids(1, 4, vocab_size=500)
        generated = model.generate(prompt, max_length=10)
        assert generated.data.shape == (1, 14)

    def test_generate_valid_tokens(self):
        model = GPT(**SMALL_GPT_CFG)
        model.eval()
        prompt = make_token_ids(1, 4, vocab_size=500)
        generated = model.generate(prompt, max_length=5)
        assert np.all(generated.data >= 0)
        assert np.all(generated.data < 500)

    def test_generate_temperature_low(self):
        """Very low temperature should be near-deterministic."""
        model = GPT(**SMALL_GPT_CFG)
        model.eval()
        prompt = make_token_ids(1, 4, vocab_size=500)
        gen1 = model.generate(prompt, max_length=5, temperature=0.01)
        gen2 = model.generate(prompt, max_length=5, temperature=0.01)
        np.testing.assert_array_equal(gen1.data, gen2.data)

    def test_generate_top_k(self):
        """Top-k generation should produce valid tokens."""
        model = GPT(**SMALL_GPT_CFG)
        model.eval()
        prompt = make_token_ids(1, 4, vocab_size=500)
        generated = model.generate(prompt, max_length=5, top_k=10)
        assert np.all(generated.data >= 0)
        assert np.all(generated.data < 500)

    def test_generate_top_p(self):
        """Nucleus sampling should produce valid tokens."""
        model = GPT(**SMALL_GPT_CFG)
        model.eval()
        prompt = make_token_ids(1, 4, vocab_size=500)
        generated = model.generate(prompt, max_length=5, top_p=0.9)
        assert np.all(generated.data >= 0)
        assert np.all(generated.data < 500)

    def test_compute_loss(self):
        """compute_loss should return a scalar loss."""
        model = GPT(**SMALL_GPT_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        loss = model.compute_loss(ids)
        assert loss.data.shape == () or loss.data.size == 1
        assert np.isfinite(loss.data)

    def test_compute_loss_backward(self):
        """Loss should be differentiable."""
        model = GPT(**SMALL_GPT_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        loss = model.compute_loss(ids)
        loss.backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_tied_embeddings(self):
        """With tie_embeddings=True, LM head should share token embedding weights."""
        model = GPT(**SMALL_GPT_CFG, tie_embeddings=True)
        ids = make_token_ids(1, 8, vocab_size=500)
        logits = model(ids)
        assert logits.data.shape == (1, 8, 500)

    def test_untied_embeddings(self):
        """With tie_embeddings=False, should have separate LM head weights."""
        model = GPT(**SMALL_GPT_CFG, tie_embeddings=False)
        ids = make_token_ids(1, 8, vocab_size=500)
        logits = model(ids)
        assert logits.data.shape == (1, 8, 500)

    def test_different_sequence_lengths(self):
        model = GPT(**SMALL_GPT_CFG)
        for seq_len in [4, 8, 16, 32]:
            ids = make_token_ids(1, seq_len, vocab_size=500)
            logits = model(ids)
            assert logits.data.shape == (1, seq_len, 500)

    def test_eval_determinism(self):
        model = GPT(**SMALL_GPT_CFG)
        model.eval()
        ids = make_token_ids(2, 8, vocab_size=500)
        out1 = model(ids)
        out2 = model(ids)
        np.testing.assert_array_equal(out1.data, out2.data)

    def test_configs_are_valid_dicts(self):
        for cfg in [GPT2_SMALL, GPT2_MEDIUM, GPT3_SMALL_CONFIG, GPT3_MEDIUM_CONFIG]:
            assert "d_model" in cfg
            assert "num_heads" in cfg
            assert "vocab_size" in cfg


# =============================================================================
# Section G: Vision Transformer (ViT) Tests
# =============================================================================

class TestPatchEmbedding:
    """Tests for PatchEmbedding."""

    def test_forward_shape(self):
        """[batch, 3, 32, 32] with patch_size=8 -> [batch, 16, embed_dim]."""
        pe = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
        x = make_image(2, 3, 32, 32)
        out = pe(x)
        num_patches = (32 // 8) ** 2  # 16
        assert out.data.shape == (2, num_patches, 64)

    def test_different_patch_sizes(self):
        for patch_size in [4, 8, 16]:
            pe = PatchEmbedding(img_size=32, patch_size=patch_size, in_channels=3, embed_dim=64)
            x = make_image(1, 3, 32, 32)
            out = pe(x)
            num_patches = (32 // patch_size) ** 2
            assert out.data.shape == (1, num_patches, 64)

    def test_output_finite(self):
        pe = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
        x = make_image(2, 3, 32, 32)
        out = pe(x)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        pe = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float64) * 0.1, requires_grad=True)
        out = pe(x)
        out.sum().backward()
        grads_exist = sum(1 for p in pe.parameters() if p.grad is not None)
        assert grads_exist > 0


class TestVisionTransformer:
    """Comprehensive tests for VisionTransformer."""

    def test_forward_shape(self):
        """Output: [batch, num_classes]."""
        model = VisionTransformer(**SMALL_VIT_CFG)
        x = make_image(2, 3, 32, 32)
        logits = model(x)
        assert logits.data.shape == (2, 10)

    def test_output_finite(self):
        model = VisionTransformer(**SMALL_VIT_CFG)
        x = make_image(2, 3, 32, 32)
        logits = model(x)
        assert np.all(np.isfinite(logits.data))

    def test_backward_produces_gradients(self):
        model = VisionTransformer(**SMALL_VIT_CFG)
        x = make_image(2, 3, 32, 32)
        logits = model(x)
        logits.sum().backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_gradients_finite(self):
        model = VisionTransformer(**SMALL_VIT_CFG)
        x = make_image(2, 3, 32, 32)
        logits = model(x)
        logits.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad))

    def test_gradient_shapes(self):
        model = VisionTransformer(**SMALL_VIT_CFG)
        x = make_image(1, 3, 32, 32)
        logits = model(x)
        logits.sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.shape == p.data.shape

    def test_different_num_classes(self):
        for nc in [2, 10, 100, 1000]:
            cfg = {**SMALL_VIT_CFG, "num_classes": nc}
            model = VisionTransformer(**cfg)
            x = make_image(1, 3, 32, 32)
            logits = model(x)
            assert logits.data.shape == (1, nc)

    def test_different_image_sizes(self):
        """Should handle images that divide evenly by patch_size."""
        for img_size in [16, 32, 64]:
            cfg = {**SMALL_VIT_CFG, "img_size": img_size}
            model = VisionTransformer(**cfg)
            x = make_image(1, 3, img_size, img_size)
            logits = model(x)
            assert logits.data.shape == (1, 10)

    def test_cls_pool_type(self):
        cfg = {**SMALL_VIT_CFG, "pool_type": "cls"}
        model = VisionTransformer(**cfg)
        x = make_image(1, 3, 32, 32)
        logits = model(x)
        assert logits.data.shape == (1, 10)

    def test_mean_pool_type(self):
        cfg = {**SMALL_VIT_CFG, "pool_type": "mean"}
        model = VisionTransformer(**cfg)
        x = make_image(1, 3, 32, 32)
        logits = model(x)
        assert logits.data.shape == (1, 10)

    def test_forward_features_shape(self):
        """forward_features should return [batch, d_model] (before classification head)."""
        model = VisionTransformer(**SMALL_VIT_CFG)
        x = make_image(2, 3, 32, 32)
        features = model.forward_features(x)
        assert features.data.shape == (2, 64)

    def test_eval_determinism(self):
        model = VisionTransformer(**SMALL_VIT_CFG)
        model.eval()
        x = make_image(2, 3, 32, 32)
        out1 = model(x)
        out2 = model(x)
        np.testing.assert_array_equal(out1.data, out2.data)

    def test_configs_are_valid_dicts(self):
        for cfg in [VIT_BASE_CONFIG, VIT_LARGE_CONFIG, VIT_HUGE_CONFIG,
                    DEIT_BASE_CONFIG, DEIT_SMALL_CONFIG]:
            assert "d_model" in cfg
            assert "num_heads" in cfg
            assert "patch_size" in cfg


# =============================================================================
# Section H: CLIP Tests
# =============================================================================

class TestCLIPTextEncoder:
    """Tests for CLIPTextEncoder."""

    def test_forward_shape(self):
        """Output: [batch, embed_dim] L2-normalized."""
        enc = CLIPTextEncoder(**SMALL_CLIP_TEXT_CFG, embed_dim=32)
        ids = make_token_ids(2, 16, vocab_size=500)
        out = enc(ids)
        assert out.data.shape == (2, 32)

    def test_output_normalized(self):
        """Output should be L2-normalized (unit vectors)."""
        enc = CLIPTextEncoder(**SMALL_CLIP_TEXT_CFG, embed_dim=32)
        ids = make_token_ids(2, 16, vocab_size=500)
        out = enc(ids)
        norms = np.linalg.norm(out.data, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_output_finite(self):
        enc = CLIPTextEncoder(**SMALL_CLIP_TEXT_CFG, embed_dim=32)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = enc(ids)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        enc = CLIPTextEncoder(**SMALL_CLIP_TEXT_CFG, embed_dim=32)
        ids = make_token_ids(2, 8, vocab_size=500)
        out = enc(ids)
        out.sum().backward()
        grads_exist = sum(1 for p in enc.parameters() if p.grad is not None)
        assert grads_exist > 0


class TestCLIPVisionEncoder:
    """Tests for CLIPVisionEncoder."""

    def test_forward_shape(self):
        """Output: [batch, embed_dim] L2-normalized."""
        enc = CLIPVisionEncoder(**SMALL_CLIP_VISION_CFG, embed_dim=32)
        x = make_image(2, 3, 32, 32)
        out = enc(x)
        assert out.data.shape == (2, 32)

    def test_output_normalized(self):
        """Output should be L2-normalized."""
        enc = CLIPVisionEncoder(**SMALL_CLIP_VISION_CFG, embed_dim=32)
        x = make_image(2, 3, 32, 32)
        out = enc(x)
        norms = np.linalg.norm(out.data, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_output_finite(self):
        enc = CLIPVisionEncoder(**SMALL_CLIP_VISION_CFG, embed_dim=32)
        x = make_image(2, 3, 32, 32)
        out = enc(x)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        enc = CLIPVisionEncoder(**SMALL_CLIP_VISION_CFG, embed_dim=32)
        x = make_image(2, 3, 32, 32)
        out = enc(x)
        out.sum().backward()
        grads_exist = sum(1 for p in enc.parameters() if p.grad is not None)
        assert grads_exist > 0


class TestCLIP:
    """Comprehensive tests for the full CLIP model."""

    def _make_clip(self):
        return CLIP(
            embed_dim=32,
            vision_cfg=SMALL_CLIP_VISION_CFG,
            text_cfg=SMALL_CLIP_TEXT_CFG,
        )

    def test_encode_image_shape(self):
        model = self._make_clip()
        x = make_image(2, 3, 32, 32)
        embeds = model.encode_image(x)
        assert embeds.data.shape == (2, 32)

    def test_encode_text_shape(self):
        model = self._make_clip()
        ids = make_token_ids(2, 16, vocab_size=500)
        embeds = model.encode_text(ids)
        assert embeds.data.shape == (2, 32)

    def test_encode_image_normalized(self):
        model = self._make_clip()
        x = make_image(4, 3, 32, 32)
        embeds = model.encode_image(x)
        norms = np.linalg.norm(embeds.data, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_text_normalized(self):
        model = self._make_clip()
        ids = make_token_ids(4, 16, vocab_size=500)
        embeds = model.encode_text(ids)
        norms = np.linalg.norm(embeds.data, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_forward_shapes(self):
        """Forward should return (logits_per_image, logits_per_text, logit_scale)."""
        model = self._make_clip()
        images = make_image(4, 3, 32, 32)
        text = make_token_ids(4, 16, vocab_size=500)
        logits_i2t, logits_t2i, scale = model(images, text)
        assert logits_i2t.data.shape == (4, 4)
        assert logits_t2i.data.shape == (4, 4)

    def test_logits_symmetric(self):
        """logits_per_text should be transpose of logits_per_image."""
        model = self._make_clip()
        images = make_image(4, 3, 32, 32)
        text = make_token_ids(4, 16, vocab_size=500)
        logits_i2t, logits_t2i, _ = model(images, text)
        np.testing.assert_allclose(logits_i2t.data, logits_t2i.data.T, atol=1e-5)

    def test_compute_loss_scalar(self):
        """compute_loss should return a scalar."""
        model = self._make_clip()
        images = make_image(4, 3, 32, 32)
        text = make_token_ids(4, 16, vocab_size=500)
        loss = model.compute_loss(images, text)
        assert loss.data.shape == () or loss.data.size == 1
        assert np.isfinite(loss.data)

    def test_compute_loss_backward(self):
        model = self._make_clip()
        images = make_image(4, 3, 32, 32)
        text = make_token_ids(4, 16, vocab_size=500)
        loss = model.compute_loss(images, text)
        loss.backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_matching_pairs_higher_similarity(self):
        """Diagonal entries (matching pairs) should generally have higher similarity
        than off-diagonal after some training, but at init we just check the shapes."""
        model = self._make_clip()
        images = make_image(4, 3, 32, 32)
        text = make_token_ids(4, 16, vocab_size=500)
        logits_i2t, _, _ = model(images, text)
        # Just verify finite and correct shape
        assert np.all(np.isfinite(logits_i2t.data))
        assert logits_i2t.data.shape == (4, 4)

    def test_configs_are_valid_dicts(self):
        for cfg in [CLIP_VIT_B32_CONFIG, CLIP_VIT_B16_CONFIG, CLIP_VIT_L14_CONFIG]:
            assert "embed_dim" in cfg
            assert "vision_cfg" in cfg
            assert "text_cfg" in cfg


# =============================================================================
# Section I: DETR Tests
# =============================================================================

class TestDETREncoder:
    """Tests for DETREncoder."""

    def test_forward_shape(self):
        enc = DETREncoder(d_model=64, num_heads=4, num_layers=2, d_ff=128, dropout=0.0)
        src = make_hidden(2, 49, 64)  # e.g., 7x7 spatial grid flattened
        out = enc(src)
        assert out.data.shape == (2, 49, 64)

    def test_output_finite(self):
        enc = DETREncoder(d_model=64, num_heads=4, num_layers=2, d_ff=128, dropout=0.0)
        src = make_hidden(2, 16, 64)
        out = enc(src)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        enc = DETREncoder(d_model=64, num_heads=4, num_layers=2, d_ff=128, dropout=0.0)
        src = make_hidden(2, 16, 64)
        out = enc(src)
        out.sum().backward()
        grads_exist = sum(1 for p in enc.parameters() if p.grad is not None)
        assert grads_exist > 0


class TestDETRDecoder:
    """Tests for DETRDecoder."""

    def test_forward_shape(self):
        """Output: [batch, num_queries, d_model]."""
        dec = DETRDecoder(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                          num_queries=10, dropout=0.0)
        memory = make_hidden(2, 49, 64)
        out = dec(memory)
        assert out.data.shape == (2, 10, 64)

    def test_output_finite(self):
        dec = DETRDecoder(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                          num_queries=10, dropout=0.0)
        memory = make_hidden(2, 16, 64)
        out = dec(memory)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        dec = DETRDecoder(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                          num_queries=10, dropout=0.0)
        memory = make_hidden(2, 16, 64)
        out = dec(memory)
        out.sum().backward()
        grads_exist = sum(1 for p in dec.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_different_num_queries(self):
        for nq in [5, 10, 50]:
            dec = DETRDecoder(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                              num_queries=nq, dropout=0.0)
            memory = make_hidden(1, 16, 64)
            out = dec(memory)
            assert out.data.shape == (1, nq, 64)


class TestDETR:
    """Comprehensive tests for the full DETR model."""

    def test_forward_shapes(self):
        """Should return (class_logits, bbox_pred)."""
        model = DETR(**SMALL_DETR_CFG)
        images = make_image(2, 3, 32, 32)
        class_logits, bbox_pred = model(images)
        assert class_logits.data.shape == (2, 10, 11)  # num_classes + 1
        assert bbox_pred.data.shape == (2, 10, 4)

    def test_output_finite(self):
        model = DETR(**SMALL_DETR_CFG)
        images = make_image(2, 3, 32, 32)
        class_logits, bbox_pred = model(images)
        assert np.all(np.isfinite(class_logits.data))
        assert np.all(np.isfinite(bbox_pred.data))

    def test_bbox_normalized(self):
        """Bounding box predictions should be in [0, 1] (normalized via sigmoid)."""
        model = DETR(**SMALL_DETR_CFG)
        images = make_image(2, 3, 32, 32)
        _, bbox_pred = model(images)
        assert np.all(bbox_pred.data >= 0.0)
        assert np.all(bbox_pred.data <= 1.0)

    def test_backward_produces_gradients(self):
        model = DETR(**SMALL_DETR_CFG)
        images = make_image(2, 3, 32, 32)
        class_logits, bbox_pred = model(images)
        loss = class_logits.sum() + bbox_pred.sum()
        loss.backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_gradients_finite(self):
        model = DETR(**SMALL_DETR_CFG)
        images = make_image(1, 3, 32, 32)
        class_logits, bbox_pred = model(images)
        loss = class_logits.sum() + bbox_pred.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad))

    def test_different_num_classes(self):
        for nc in [10, 20, 91]:
            cfg = {**SMALL_DETR_CFG, "num_classes": nc}
            model = DETR(**cfg)
            images = make_image(1, 3, 32, 32)
            class_logits, bbox_pred = model(images)
            assert class_logits.data.shape == (1, 10, nc + 1)
            assert bbox_pred.data.shape == (1, 10, 4)

    def test_configs_are_valid_dicts(self):
        for cfg in [DETR_RESNET50_CONFIG, DETR_RESNET101_CONFIG]:
            assert "num_classes" in cfg
            assert "d_model" in cfg
            assert "num_queries" in cfg


# =============================================================================
# Section J: Switch Transformer Tests
# =============================================================================

class TestSwitchTransformerLayer:
    """Tests for SwitchTransformerLayer."""

    def test_forward_shape(self):
        """Output: ([batch, seq_len, d_model], scalar balance_loss)."""
        layer = SwitchTransformerLayer(d_model=64, num_heads=4, d_ff=128,
                                       num_experts=4, dropout=0.0)
        x = make_hidden(2, 16, 64)
        out, balance_loss = layer(x)
        assert out.data.shape == (2, 16, 64)

    def test_balance_loss_scalar(self):
        """Balance loss should be a non-negative scalar."""
        layer = SwitchTransformerLayer(d_model=64, num_heads=4, d_ff=128,
                                       num_experts=4, dropout=0.0)
        x = make_hidden(2, 8, 64)
        _, balance_loss = layer(x)
        assert balance_loss.data.shape == () or balance_loss.data.size == 1
        assert np.isfinite(balance_loss.data)
        assert float(balance_loss.data) >= 0.0

    def test_output_finite(self):
        layer = SwitchTransformerLayer(d_model=64, num_heads=4, d_ff=128,
                                       num_experts=4, dropout=0.0)
        x = make_hidden(2, 8, 64)
        out, _ = layer(x)
        assert np.all(np.isfinite(out.data))

    def test_backward_produces_gradients(self):
        layer = SwitchTransformerLayer(d_model=64, num_heads=4, d_ff=128,
                                       num_experts=4, dropout=0.0)
        x = make_hidden(2, 8, 64)
        out, balance_loss = layer(x)
        total = out.sum() + balance_loss
        total.backward()
        grads_exist = sum(1 for p in layer.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_different_num_experts(self):
        for ne in [2, 4, 8]:
            layer = SwitchTransformerLayer(d_model=64, num_heads=4, d_ff=128,
                                           num_experts=ne, dropout=0.0)
            x = make_hidden(1, 8, 64)
            out, _ = layer(x)
            assert out.data.shape == (1, 8, 64)


class TestSwitchTransformer:
    """Comprehensive tests for the full SwitchTransformer model."""

    def test_forward_shapes(self):
        """Output: (logits, total_balance_loss)."""
        model = SwitchTransformer(**SMALL_SWITCH_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        logits, balance_loss = model(ids)
        assert logits.data.shape == (2, 16, 500)
        assert balance_loss.data.shape == () or balance_loss.data.size == 1

    def test_output_finite(self):
        model = SwitchTransformer(**SMALL_SWITCH_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits, balance_loss = model(ids)
        assert np.all(np.isfinite(logits.data))
        assert np.isfinite(balance_loss.data)

    def test_backward_produces_gradients(self):
        model = SwitchTransformer(**SMALL_SWITCH_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits, balance_loss = model(ids)
        total = logits.sum() + balance_loss
        total.backward()
        grads_exist = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_exist > 0

    def test_gradients_finite(self):
        model = SwitchTransformer(**SMALL_SWITCH_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        logits, balance_loss = model(ids)
        total = logits.sum() + balance_loss
        total.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(np.isfinite(p.grad))

    def test_different_sequence_lengths(self):
        model = SwitchTransformer(**SMALL_SWITCH_CFG)
        for seq_len in [4, 8, 16]:
            ids = make_token_ids(1, seq_len, vocab_size=500)
            logits, _ = model(ids)
            assert logits.data.shape == (1, seq_len, 500)

    def test_balance_loss_nonnegative(self):
        model = SwitchTransformer(**SMALL_SWITCH_CFG)
        ids = make_token_ids(2, 8, vocab_size=500)
        _, balance_loss = model(ids)
        assert float(balance_loss.data) >= 0.0

    def test_with_padding_mask(self):
        model = SwitchTransformer(**SMALL_SWITCH_CFG)
        ids = make_token_ids(2, 16, vocab_size=500)
        mask = make_padding_mask(2, 16)
        logits, _ = model(ids, padding_mask=mask)
        assert logits.data.shape == (2, 16, 500)

    def test_more_experts_more_params(self):
        """More experts should mean more parameters."""
        cfg4 = {**SMALL_SWITCH_CFG, "num_experts": 4}
        cfg8 = {**SMALL_SWITCH_CFG, "num_experts": 8}
        model4 = SwitchTransformer(**cfg4)
        model8 = SwitchTransformer(**cfg8)
        params4 = sum(p.data.size for p in model4.parameters())
        params8 = sum(p.data.size for p in model8.parameters())
        assert params8 > params4, "More experts should increase parameter count"

    def test_configs_are_valid_dicts(self):
        for cfg in [SWITCH_BASE_8_CONFIG, SWITCH_BASE_64_CONFIG, SWITCH_LARGE_128_CONFIG]:
            assert "d_model" in cfg
            assert "num_experts" in cfg
            assert "vocab_size" in cfg


# =============================================================================
# Section K: Cross-Architecture Tests
# =============================================================================

class TestCrossArchitecture:
    """Tests that compare or verify properties across architectures."""

    def test_encoder_more_layers_more_params(self):
        """More layers should mean more parameters."""
        small = TransformerEncoder(**{**SMALL_ENCODER_CFG, "num_layers": 2})
        large = TransformerEncoder(**{**SMALL_ENCODER_CFG, "num_layers": 4})
        ps = sum(p.data.size for p in small.parameters())
        pl = sum(p.data.size for p in large.parameters())
        assert pl > ps

    def test_decoder_more_layers_more_params(self):
        small = TransformerDecoder(**{**SMALL_DECODER_CFG, "num_layers": 2})
        large = TransformerDecoder(**{**SMALL_DECODER_CFG, "num_layers": 4})
        ps = sum(p.data.size for p in small.parameters())
        pl = sum(p.data.size for p in large.parameters())
        assert pl > ps

    def test_larger_d_model_more_params(self):
        """Larger d_model should increase parameter count."""
        small = GPT(**{**SMALL_GPT_CFG, "d_model": 32, "num_heads": 4, "d_ff": 64})
        large = GPT(**{**SMALL_GPT_CFG, "d_model": 128, "num_heads": 4, "d_ff": 256})
        ps = sum(p.data.size for p in small.parameters())
        pl = sum(p.data.size for p in large.parameters())
        assert pl > ps

    def test_vit_larger_patch_fewer_tokens(self):
        """Larger patches should produce fewer tokens (and thus fewer computations)."""
        vit_small_patch = VisionTransformer(**{**SMALL_VIT_CFG, "patch_size": 4})
        vit_large_patch = VisionTransformer(**{**SMALL_VIT_CFG, "patch_size": 16})
        # Larger patches = fewer params in positional embeddings
        ps = sum(p.data.size for p in vit_small_patch.parameters())
        pl = sum(p.data.size for p in vit_large_patch.parameters())
        assert ps > pl, "Smaller patches should mean more positional embeddings"


# =============================================================================
# Section L: Inference Benchmarks (slow)
# =============================================================================

BENCHMARK_MODELS = {
    "encoder": (TransformerEncoder, SMALL_ENCODER_CFG, "text"),
    "decoder": (TransformerDecoder, SMALL_DECODER_CFG, "text"),
    "gpt": (GPT, SMALL_GPT_CFG, "text"),
    "vit": (VisionTransformer, SMALL_VIT_CFG, "image"),
}


@pytest.mark.slow
class TestTransformerBenchmarks:
    """Benchmark inference speed for transformer models."""

    @pytest.mark.parametrize("name", list(BENCHMARK_MODELS.keys()))
    def test_inference_speed(self, name):
        factory, cfg, input_type = BENCHMARK_MODELS[name]
        model = factory(**cfg)
        model.eval()

        timings = {}
        for bs in [1, 4]:
            if input_type == "text":
                x = make_token_ids(bs, 16, vocab_size=cfg.get("vocab_size", 500))
            else:
                size = cfg.get("img_size", 32)
                x = make_image(bs, 3, size, size)

            # Warmup
            with no_grad():
                _ = model(x) if input_type == "image" else model(x)
            # Timed
            start = time.time()
            with no_grad():
                _ = model(x) if input_type == "image" else model(x)
            elapsed = time.time() - start
            timings[f"batch_{bs}"] = round(elapsed, 4)

        result = {"model": name, **timings}

        results_file = RESULTS_DIR / "transformer_inference_speed.json"
        if results_file.exists():
            with open(results_file) as f:
                all_results = json.load(f)
        else:
            all_results = []
        all_results = [r for r in all_results if r["model"] != name]
        all_results.append(result)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        assert timings["batch_1"] >= 0
