"""
PyTorch Transformer Models.

Ports of the custom numpy transformer implementations to torch.nn.
"""

from .encoder import EncoderLayer, TransformerEncoder
from .decoder import DecoderLayer, TransformerDecoder
from .encoder_decoder import (
    DecoderLayerWithCrossAttention,
    TransformerEncoderDecoder,
    TRANSFORMER_BASE_CONFIG,
    T5_BASE_CONFIG,
    T5_LARGE_CONFIG,
)
from .gpt import (
    GPT, GPTBlock,
    GPT2_SMALL_CONFIG, GPT2_MEDIUM_CONFIG, GPT2_LARGE_CONFIG, GPT2_XL_CONFIG,
)
from .pretrained import load_gpt2_weights

__all__ = [
    'EncoderLayer', 'TransformerEncoder',
    'DecoderLayer', 'TransformerDecoder',
    'DecoderLayerWithCrossAttention', 'TransformerEncoderDecoder',
    'GPT', 'GPTBlock',
    'GPT2_SMALL_CONFIG', 'GPT2_MEDIUM_CONFIG', 'GPT2_LARGE_CONFIG', 'GPT2_XL_CONFIG',
    'TRANSFORMER_BASE_CONFIG', 'T5_BASE_CONFIG', 'T5_LARGE_CONFIG',
    'load_gpt2_weights',
]
