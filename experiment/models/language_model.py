"""Language model registrations: GPT-2 variants."""

from experiment.registry import register_model


# =============================================================================
# PyTorch
# =============================================================================

@register_model('gpt2', 'pytorch')
def _pt_gpt2(config):
    from pytorch.sequence.transformers.gpt import GPT, GPT2_SMALL_CONFIG
    cfg = {**GPT2_SMALL_CONFIG, **config.model_args}
    cfg['max_seq_len'] = config.max_seq_len
    model = GPT(**cfg)
    if getattr(config, 'pretrained_weights', None):
        from pytorch.sequence.transformers.pretrained import load_gpt2_weights
        model = load_gpt2_weights(model, config.pretrained_weights)
    return model


@register_model('gpt2-medium', 'pytorch')
def _pt_gpt2_medium(config):
    from pytorch.sequence.transformers.gpt import GPT, GPT2_MEDIUM_CONFIG
    cfg = {**GPT2_MEDIUM_CONFIG, **config.model_args}
    cfg['max_seq_len'] = config.max_seq_len
    model = GPT(**cfg)
    if getattr(config, 'pretrained_weights', None):
        from pytorch.sequence.transformers.pretrained import load_gpt2_weights
        model = load_gpt2_weights(model, config.pretrained_weights)
    return model


@register_model('gpt2-large', 'pytorch')
def _pt_gpt2_large(config):
    from pytorch.sequence.transformers.gpt import GPT, GPT2_LARGE_CONFIG
    cfg = {**GPT2_LARGE_CONFIG, **config.model_args}
    cfg['max_seq_len'] = config.max_seq_len
    model = GPT(**cfg)
    if getattr(config, 'pretrained_weights', None):
        from pytorch.sequence.transformers.pretrained import load_gpt2_weights
        model = load_gpt2_weights(model, config.pretrained_weights)
    return model


# =============================================================================
# Numpy
# =============================================================================

@register_model('gpt2', 'numpy')
def _np_gpt2(config):
    from python.sequence.transformers.gpt import GPT, GPT2_SMALL_CONFIG
    cfg = {**GPT2_SMALL_CONFIG, **config.model_args}
    cfg['max_seq_len'] = config.max_seq_len
    return GPT(**cfg)


@register_model('gpt2-medium', 'numpy')
def _np_gpt2_medium(config):
    from python.sequence.transformers.gpt import GPT, GPT2_MEDIUM_CONFIG
    cfg = {**GPT2_MEDIUM_CONFIG, **config.model_args}
    cfg['max_seq_len'] = config.max_seq_len
    return GPT(**cfg)


@register_model('gpt2-large', 'numpy')
def _np_gpt2_large(config):
    from python.sequence.transformers.gpt import GPT, GPT2_LARGE_CONFIG
    cfg = {**GPT2_LARGE_CONFIG, **config.model_args}
    cfg['max_seq_len'] = config.max_seq_len
    return GPT(**cfg)
