"""BERT model registrations for all task variants."""

from experiment.registry import register_model


# =============================================================================
# PyTorch BERT Models
# =============================================================================

@register_model('bert_pretrain', 'pytorch')
def _pt_bert_pretrain(config):
    from pytorch.sequence.transformers.bert import BertForPreTraining, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForPreTraining(**base_cfg)


@register_model('bert_mlm', 'pytorch')
def _pt_bert_mlm(config):
    from pytorch.sequence.transformers.bert import BertForMaskedLM, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForMaskedLM(**base_cfg)


@register_model('bert_nsp', 'pytorch')
def _pt_bert_nsp(config):
    from pytorch.sequence.transformers.bert import BertForNextSentencePrediction, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForNextSentencePrediction(**base_cfg)


@register_model('bert_token_cls', 'pytorch')
def _pt_bert_token_cls(config):
    from pytorch.sequence.transformers.bert import BertForTokenClassification, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    num_classes = config.model_args.get('num_classes', 17)
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    base_cfg.pop('num_classes', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForTokenClassification(num_classes=num_classes, **base_cfg)


@register_model('bert_seq_cls', 'pytorch')
def _pt_bert_seq_cls(config):
    from pytorch.sequence.transformers.bert import BertForSequenceClassification, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    num_classes = config.model_args.get('num_classes', 2)
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    base_cfg.pop('num_classes', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForSequenceClassification(num_classes=num_classes, **base_cfg)


# =============================================================================
# Numpy BERT Models
# =============================================================================

@register_model('bert_pretrain', 'numpy')
def _np_bert_pretrain(config):
    from python.sequence.transformers.bert import BertForPreTraining, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForPreTraining(**base_cfg)


@register_model('bert_mlm', 'numpy')
def _np_bert_mlm(config):
    from python.sequence.transformers.bert import BertForMaskedLM, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForMaskedLM(**base_cfg)


@register_model('bert_nsp', 'numpy')
def _np_bert_nsp(config):
    from python.sequence.transformers.bert import BertForNextSentencePrediction, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForNextSentencePrediction(**base_cfg)


@register_model('bert_token_cls', 'numpy')
def _np_bert_token_cls(config):
    from python.sequence.transformers.bert import BertForTokenClassification, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    num_classes = config.model_args.get('num_classes', 17)
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    base_cfg.pop('num_classes', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForTokenClassification(num_classes=num_classes, **base_cfg)


@register_model('bert_seq_cls', 'numpy')
def _np_bert_seq_cls(config):
    from python.sequence.transformers.bert import BertForSequenceClassification, BERT_BASE_CONFIG, BERT_TINY_CONFIG
    num_classes = config.model_args.get('num_classes', 2)
    base_cfg = dict(BERT_TINY_CONFIG) if config.model_args.get('tiny', False) else dict(BERT_BASE_CONFIG)
    base_cfg.update(config.model_args)
    base_cfg.pop('tiny', None)
    base_cfg.pop('num_classes', None)
    if hasattr(config, 'max_seq_len'):
        base_cfg['max_seq_len'] = config.max_seq_len
    return BertForSequenceClassification(num_classes=num_classes, **base_cfg)
