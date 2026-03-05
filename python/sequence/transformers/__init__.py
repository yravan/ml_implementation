"""
Transformer Architecture Implementations

This package provides implementations of major transformer architectures,
all built on top of nn_core primitives (MultiHeadAttention, CausalMask,
LayerNorm, Linear, etc.).

Architectures:
    - Encoder: Bidirectional transformer encoder (BERT-style)
    - Decoder: Autoregressive decoder (GPT-style)
    - EncoderDecoder: Seq2seq (Vaswani et al. original transformer)
    - BERT: Bidirectional encoder for understanding tasks
    - GPT: Decoder-only autoregressive language model
    - ViT: Vision Transformer for image classification
    - DETR: Detection Transformer for object detection
    - Switch Transformer: Sparse Mixture-of-Experts transformer
    - CLIP: Contrastive Language-Image Pre-training
"""

# Encoder (bidirectional)
from python.sequence.transformers.encoder import (
    EncoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    BERT_CONFIG,
    BERT_LARGE_CONFIG,
    ROBERTA_CONFIG,
)

# Decoder (autoregressive)
from python.sequence.transformers.decoder import (
    DecoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
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
    GPT2_LARGE_CONFIG as GPT2_LARGE,
    GPT2_XL_CONFIG as GPT2_XL,
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

# DETR (Detection Transformer)
from python.sequence.transformers.detr import (
    DETREncoder,
    DETRDecoder,
    DETR,
    DETR_RESNET50_CONFIG,
    DETR_RESNET101_CONFIG,
)

# Switch Transformer (Mixture-of-Experts)
from python.sequence.transformers.switch_transformer import (
    SwitchTransformerLayer,
    SwitchTransformer,
    SWITCH_BASE_8_CONFIG,
    SWITCH_BASE_64_CONFIG,
    SWITCH_LARGE_128_CONFIG,
)

# CLIP (Contrastive Language-Image Pre-training)
from python.sequence.transformers.clip import (
    CLIPTextEncoder,
    CLIPVisionEncoder,
    CLIP,
    CLIP_VIT_B32_CONFIG,
    CLIP_VIT_B16_CONFIG,
    CLIP_VIT_L14_CONFIG,
)
