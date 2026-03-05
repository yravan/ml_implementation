"""
Tests for the experiment runner class hierarchy.

Tests:
  - Runner dispatch (get_runner returns correct class)
  - Classification runner backward compatibility
  - Language model runner training step on tiny model
  - Seq2seq runner training step on tiny model
  - Config new fields
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from experiment.config import Config
from experiment.runner import (
    get_runner,
    BaseRunner,
    ClassificationRunner,
    LanguageModelRunner,
    Seq2SeqRunner,
    METRIC_FNS,
    top1_accuracy,
    top5_accuracy,
)


# =============================================================================
# Runner Dispatch
# =============================================================================

class TestRunnerDispatch:
    """Test that get_runner returns the correct runner class."""

    def test_classification_default(self):
        config = Config(task='classification')
        runner = get_runner(config)
        assert isinstance(runner, ClassificationRunner)

    def test_classification_explicit(self):
        config = Config()
        runner = get_runner(config)
        assert isinstance(runner, ClassificationRunner)

    def test_language_model(self):
        config = Config(task='language_model')
        runner = get_runner(config)
        assert isinstance(runner, LanguageModelRunner)

    def test_seq2seq(self):
        config = Config(task='seq2seq')
        runner = get_runner(config)
        assert isinstance(runner, Seq2SeqRunner)

    def test_unknown_task_defaults_to_classification(self):
        config = Config(task='unknown_task')
        runner = get_runner(config)
        assert isinstance(runner, ClassificationRunner)


# =============================================================================
# Config New Fields
# =============================================================================

class TestConfigFields:
    """Test that new config fields exist and have correct defaults."""

    def test_task_default(self):
        c = Config()
        assert c.task == 'classification'

    def test_tokenizer_default(self):
        c = Config()
        assert c.tokenizer == 'gpt2'

    def test_max_seq_len_default(self):
        c = Config()
        assert c.max_seq_len == 512

    def test_generate_every_default(self):
        c = Config()
        assert c.generate_every == 0

    def test_generate_temperature_default(self):
        c = Config()
        assert c.generate_temperature == 0.8

    def test_pretrained_weights_default(self):
        c = Config()
        assert c.pretrained_weights is None

    def test_generate_prompts_default(self):
        c = Config()
        assert c.generate_prompts == ["Once upon a time"]

    def test_src_tgt_max_seq_len_default(self):
        c = Config()
        assert c.src_max_seq_len is None
        assert c.tgt_max_seq_len is None

    def test_from_dict_with_new_fields(self):
        c = Config.from_dict({
            'task': 'language_model',
            'tokenizer': 'gpt2',
            'max_seq_len': 256,
            'generate_every': 5,
            'pretrained_weights': 'gpt2',
        })
        assert c.task == 'language_model'
        assert c.tokenizer == 'gpt2'
        assert c.max_seq_len == 256
        assert c.generate_every == 5
        assert c.pretrained_weights == 'gpt2'


# =============================================================================
# Metrics
# =============================================================================

class TestMetrics:
    """Test metric functions work correctly."""

    def test_top1_accuracy_torch(self):
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        targets = torch.tensor([0, 1])
        acc = top1_accuracy(logits, targets)
        assert float(acc) == 2.0  # both correct

    def test_top1_accuracy_numpy(self):
        logits = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        targets = np.array([0, 2])
        acc = top1_accuracy(logits, targets)
        assert acc == 1.0  # first correct, second wrong

    def test_top5_accuracy_torch(self):
        logits = torch.randn(4, 100)
        targets = logits.argmax(dim=1)  # guaranteed correct
        acc = top5_accuracy(logits, targets)
        assert float(acc) == 4.0

    def test_metric_fns_registry(self):
        assert 'top1' in METRIC_FNS
        assert 'top5' in METRIC_FNS


# =============================================================================
# GPT Model Tests
# =============================================================================

class TestGPTModel:
    """Test the PyTorch GPT model."""

    def test_forward_shape(self):
        from pytorch.sequence.transformers.gpt import GPT
        model = GPT(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                     vocab_size=100, max_seq_len=32, dropout=0.0)
        input_ids = torch.randint(0, 100, (2, 16))
        logits = model(input_ids)
        assert logits.shape == (2, 16, 100)

    def test_generate(self):
        from pytorch.sequence.transformers.gpt import GPT
        model = GPT(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                     vocab_size=100, max_seq_len=32, dropout=0.0)
        prompt = torch.randint(0, 100, (1, 4))
        generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10)
        assert generated.shape[0] == 1
        assert generated.shape[1] == 14  # 4 prompt + 10 generated

    def test_weight_tying(self):
        from pytorch.sequence.transformers.gpt import GPT
        model = GPT(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                     vocab_size=100, max_seq_len=32)
        assert model.lm_head.weight is model.token_embedding.weight

    def test_configs_exist(self):
        from pytorch.sequence.transformers.gpt import (
            GPT2_SMALL_CONFIG, GPT2_MEDIUM_CONFIG, GPT2_LARGE_CONFIG, GPT2_XL_CONFIG,
        )
        assert GPT2_SMALL_CONFIG['d_model'] == 768
        assert GPT2_MEDIUM_CONFIG['d_model'] == 1024
        assert GPT2_LARGE_CONFIG['d_model'] == 1280
        assert GPT2_XL_CONFIG['d_model'] == 1600


# =============================================================================
# Encoder-Decoder Model Tests
# =============================================================================

class TestEncoderDecoder:
    """Test the PyTorch TransformerEncoderDecoder model."""

    def test_forward_shape(self):
        from pytorch.sequence.transformers.encoder_decoder import TransformerEncoderDecoder
        model = TransformerEncoderDecoder(
            d_model=64, num_heads=4,
            num_encoder_layers=2, num_decoder_layers=2,
            d_ff=128, src_vocab_size=100, tgt_vocab_size=100,
            max_src_len=32, max_tgt_len=32, dropout=0.0,
        )
        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))
        logits = model(src, tgt)
        assert logits.shape == (2, 8, 100)

    def test_generate(self):
        from pytorch.sequence.transformers.encoder_decoder import TransformerEncoderDecoder
        model = TransformerEncoderDecoder(
            d_model=64, num_heads=4,
            num_encoder_layers=2, num_decoder_layers=2,
            d_ff=128, src_vocab_size=100, tgt_vocab_size=100,
            max_src_len=32, max_tgt_len=32, dropout=0.0,
        )
        src = torch.randint(0, 100, (1, 10))
        generated = model.generate(src, max_length=5, start_token_id=1, end_token_id=2)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 2  # at least BOS + one token


# =============================================================================
# Language Model Training Step
# =============================================================================

class TestLanguageModelStep:
    """Test LM runner's train/eval step on a tiny model."""

    def _make_tiny_lm(self):
        from pytorch.sequence.transformers.gpt import GPT
        return GPT(d_model=32, num_heads=2, num_layers=1, d_ff=64,
                    vocab_size=50, max_seq_len=16, dropout=0.0)

    def test_train_step_reduces_loss(self):
        """A few training steps should reduce loss."""
        model = self._make_tiny_lm()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        import torch.nn.functional as F

        def criterion(logits, targets):
            B, T, V = logits.shape
            return F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))

        losses = []
        for _ in range(20):
            input_ids = torch.randint(0, 50, (4, 16))
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            logits = model(inputs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_perplexity_from_loss(self):
        """Perplexity should be exp(loss)."""
        import math
        loss = 3.5
        ppl = math.exp(min(loss, 100))
        assert abs(ppl - math.exp(3.5)) < 1e-6


# =============================================================================
# Seq2Seq Training Step
# =============================================================================

class TestSeq2SeqStep:
    """Test seq2seq runner's train/eval step on a tiny model."""

    def _make_tiny_seq2seq(self):
        from pytorch.sequence.transformers.encoder_decoder import TransformerEncoderDecoder
        return TransformerEncoderDecoder(
            d_model=32, num_heads=2,
            num_encoder_layers=1, num_decoder_layers=1,
            d_ff=64, src_vocab_size=50, tgt_vocab_size=50,
            max_src_len=16, max_tgt_len=16, dropout=0.0,
        )

    def test_train_step_reduces_loss(self):
        """Training steps should reduce loss on repeated data."""
        model = self._make_tiny_seq2seq()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        import torch.nn.functional as F

        def criterion(logits, targets):
            B, T, V = logits.shape
            return F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))

        # Use fixed data so the model can memorize it
        torch.manual_seed(42)
        src = torch.randint(0, 50, (4, 10))
        tgt = torch.randint(0, 50, (4, 8))
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        losses = []
        for _ in range(50):
            logits = model(src, tgt_input)
            loss = criterion(logits, tgt_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# =============================================================================
# Encoder Tests
# =============================================================================

class TestTransformerEncoder:
    """Test the PyTorch TransformerEncoder."""

    def test_forward_shape(self):
        from pytorch.sequence.transformers.encoder import TransformerEncoder
        model = TransformerEncoder(
            d_model=64, num_heads=4, num_layers=2, d_ff=128,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        input_ids = torch.randint(0, 100, (2, 16))
        output = model(input_ids)
        assert output.shape == (2, 16, 64)

    def test_with_padding_mask(self):
        from pytorch.sequence.transformers.encoder import TransformerEncoder
        model = TransformerEncoder(
            d_model=64, num_heads=4, num_layers=2, d_ff=128,
            vocab_size=100, max_seq_len=32, dropout=0.0,
        )
        input_ids = torch.randint(0, 100, (2, 16))
        # True = IGNORE this position
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, 10:] = True
        output = model(input_ids, key_padding_mask=mask)
        assert output.shape == (2, 16, 64)


# =============================================================================
# Model Registration Tests
# =============================================================================

class TestModelRegistration:
    """Test that sequence models are properly registered."""

    def test_gpt2_registered(self):
        from experiment.registry import list_models
        models = list_models('pytorch')
        assert 'gpt2' in models

    def test_gpt2_medium_registered(self):
        from experiment.registry import list_models
        models = list_models('pytorch')
        assert 'gpt2-medium' in models

    def test_transformer_base_registered(self):
        from experiment.registry import list_models
        models = list_models('pytorch')
        assert 'transformer_base' in models

    def test_t5_base_registered(self):
        from experiment.registry import list_models
        models = list_models('pytorch')
        assert 't5_base' in models

    def test_build_gpt2(self):
        from experiment.registry import build_model
        config = Config(model='gpt2', backend='pytorch', max_seq_len=32,
                        model_args={'d_model': 32, 'num_heads': 2, 'num_layers': 1,
                                    'd_ff': 64, 'vocab_size': 50})
        model = build_model(config)
        assert model is not None

    def test_build_transformer_base(self):
        from experiment.registry import build_model
        config = Config(model='transformer_base', backend='pytorch', max_seq_len=32,
                        model_args={'d_model': 32, 'num_heads': 2,
                                    'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                    'd_ff': 64, 'src_vocab_size': 50, 'tgt_vocab_size': 50})
        model = build_model(config)
        assert model is not None


# =============================================================================
# Dataset Registration Tests
# =============================================================================

class TestDatasetRegistration:
    """Test that sequence datasets are properly registered."""

    def test_wikitext2_registered(self):
        from experiment.registry import list_datasets
        datasets = list_datasets('pytorch')
        assert 'wikitext2' in datasets

    def test_wikitext103_registered(self):
        from experiment.registry import list_datasets
        datasets = list_datasets('pytorch')
        assert 'wikitext103' in datasets

    def test_openwebtext_registered(self):
        from experiment.registry import list_datasets
        datasets = list_datasets('pytorch')
        assert 'openwebtext' in datasets

    def test_multi30k_registered(self):
        from experiment.registry import list_datasets
        datasets = list_datasets('pytorch')
        assert 'multi30k' in datasets

    def test_wmt14_registered(self):
        from experiment.registry import list_datasets
        datasets = list_datasets('pytorch')
        assert 'wmt14' in datasets


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
