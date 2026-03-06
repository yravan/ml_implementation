"""Seq2seq model registrations: Transformer base, T5."""

from experiment.registry import register_model


# =============================================================================
# PyTorch
# =============================================================================

@register_model('transformer_base', 'pytorch')
def _pt_transformer_base(config):
    from pytorch.sequence.transformers.encoder_decoder import (
        TransformerEncoderDecoder, TRANSFORMER_BASE_CONFIG,
    )
    cfg = {**TRANSFORMER_BASE_CONFIG, **config.model_args}
    if hasattr(config, 'max_seq_len'):
        cfg['max_src_len'] = getattr(config, 'src_max_seq_len', None) or config.max_seq_len
        cfg['max_tgt_len'] = getattr(config, 'tgt_max_seq_len', None) or config.max_seq_len
    return TransformerEncoderDecoder(**cfg)


@register_model('t5_base', 'pytorch')
def _pt_t5_base(config):
    from pytorch.sequence.transformers.encoder_decoder import (
        TransformerEncoderDecoder, T5_BASE_CONFIG,
    )
    cfg = {**T5_BASE_CONFIG, **config.model_args}
    if hasattr(config, 'max_seq_len'):
        cfg['max_src_len'] = getattr(config, 'src_max_seq_len', None) or config.max_seq_len
        cfg['max_tgt_len'] = getattr(config, 'tgt_max_seq_len', None) or config.max_seq_len
    return TransformerEncoderDecoder(**cfg)


# =============================================================================
# Numpy
# =============================================================================

@register_model('transformer_base', 'numpy')
def _np_transformer_base(config):
    from python.sequence.transformers.encoder_decoder import (
        TransformerEncoderDecoder, TRANSFORMER_BASE_CONFIG,
    )
    cfg = {**TRANSFORMER_BASE_CONFIG, **config.model_args}
    cfg['max_src_len'] = getattr(config, 'src_max_seq_len', None) or config.max_seq_len
    cfg['max_tgt_len'] = getattr(config, 'tgt_max_seq_len', None) or config.max_seq_len
    return TransformerEncoderDecoder(**cfg)


@register_model('t5_base', 'numpy')
def _np_t5_base(config):
    from python.sequence.transformers.encoder_decoder import (
        TransformerEncoderDecoder, T5_BASE_CONFIG,
    )
    cfg = {**T5_BASE_CONFIG, **config.model_args}
    cfg['max_src_len'] = getattr(config, 'src_max_seq_len', None) or config.max_seq_len
    cfg['max_tgt_len'] = getattr(config, 'tgt_max_seq_len', None) or config.max_seq_len
    return TransformerEncoderDecoder(**cfg)
