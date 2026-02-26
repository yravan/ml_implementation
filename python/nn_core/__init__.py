"""
Neural Network Core Module
==========================

Core building blocks for neural networks including:
- Module: Base class for all layers (provides parameter tracking)
- Parameter: Learnable tensor wrapper
- Layers: Linear, Conv2D, etc.
- Activations: ReLU, GELU, Softmax, etc.
- Normalization: BatchNorm, LayerNorm, etc.
- Attention: Multi-head attention, etc.
- Pooling: MaxPool, AvgPool, etc.
- Regularization: Dropout, DropPath, etc.
- Recurrent: RNN, LSTM, GRU, etc.
- Initialization: Xavier, Kaiming, etc.

All layers inherit from Module and use Parameter for weights.

Module Structure:
- *_functional.py: Function classes with forward/backward for autograd
- *.py: Module classes that wrap functional operations for Tensor use
"""

from .module import (
    Module,
    Parameter,
    Sequential,
    ModuleList,
    ModuleDict,
    ParameterList,
    ParameterDict,
)

# Linear layer
from .linear import Linear

# Activations
from .activations import (
    ReLU, LeakyReLU, PReLU, ELU, SELU,
    ReLU6, GELU, QuickGELU, SiLU,
    Sigmoid, LogSigmoid, HardSigmoid,
    Softmax, LogSoftmax, Softmax2D,
    Tanh, Hardtanh, Tanhshrink,
    Softplus, Softsign, Mish, Threshold,
)

# Activation functional
from . import activations_functional

# Convolutions
from .conv import (
    Conv1d, Conv2d, Conv3d,
    ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    DepthwiseConv2d, PointwiseConv2d, SeparableConv2d,
)

# Convolution functional
from . import conv_functional

# Normalization
from .normalization import (
    BatchNorm1d, BatchNorm2d, BatchNorm3d,
    LayerNorm, GroupNorm,
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d,
    LocalResponseNorm, RMSNorm,
)

# Normalization functional
from . import normalization_functional

# Pooling
from .pooling import (
    MaxPool1d, MaxPool2d, MaxPool3d,
    AvgPool1d, AvgPool2d, AvgPool3d,
    AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d,
    LPPool2d, MaxUnpool2d,
    GlobalAvgPool1d, GlobalAvgPool2d,
    GlobalMaxPool1d, GlobalMaxPool2d,
)

# Pooling functional
from . import pooling_functional

# Attention
from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention, CachedMultiHeadAttention,
    CausalMask,
    MultiQueryAttention, GroupedQueryAttention,
)

# Attention functional
from . import attention_functional

# Regularization
from .regularization import (
    Dropout, Dropout1d, Dropout2d, Dropout3d,
    DropPath, DropPathScheduled,
    DropoutScheduled,
)

# Regularization functional
from . import regularization_functional

# Recurrent
from .recurrent import (
    RNNCell, LSTMCell, GRUCell,
    RNN, LSTM, GRU,
)

# Recurrent functional
from . import recurrent_functional

# Initialization (all functions take Tensor as first argument)
from .init import (
    xavier_uniform_, xavier_normal_,
    kaiming_uniform_, kaiming_normal_,
    normal_, uniform_, zeros_, ones_, constant_,
    orthogonal_,
    GainConfig, ActivationConfig,
    calculate_fan_in_fan_out,
)

# Positional encodings
from .positional import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RelativePositionalEmbedding,
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    create_sinusoidal_encoding,
    create_rope_encoding,
    compare_positional_encodings,
)


__all__ = [
    # Base classes
    'Module',
    'Parameter',
    'Sequential',
    'ModuleList',
    'ModuleDict',
    'ParameterList',
    'ParameterDict',

    # Linear
    'Linear',

    # Activations
    'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU',
    'ReLU6', 'GELU', 'QuickGELU', 'SiLU',
    'Sigmoid', 'LogSigmoid', 'HardSigmoid',
    'Softmax', 'LogSoftmax', 'Softmax2D',
    'Tanh', 'Hardtanh', 'Tanhshrink',
    'Softplus', 'Softsign', 'Mish', 'Threshold',

    # Convolutions
    'Conv1d', 'Conv2d', 'Conv3d',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'DepthwiseConv2d', 'PointwiseConv2d', 'SeparableConv2d',

    # Normalization
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
    'LayerNorm', 'GroupNorm',
    'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    'LocalResponseNorm', 'RMSNorm',

    # Pooling
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
    'LPPool2d', 'MaxUnpool2d',
    'GlobalAvgPool1d', 'GlobalAvgPool2d',
    'GlobalMaxPool1d', 'GlobalMaxPool2d',

    # Attention
    'ScaledDotProductAttention',
    'MultiHeadAttention', 'CachedMultiHeadAttention'
    'CausalMask',
    'MultiQueryAttention', 'GroupedQueryAttention',

    # Regularization
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d',
    'DropPath', 'DropPathScheduled',
    'DropoutScheduled',

    # Recurrent
    'RNNCell', 'LSTMCell', 'GRUCell',
    'RNN', 'LSTM', 'GRU',

    # Functional modules
    'activations_functional',
    'conv_functional',
    'normalization_functional',
    'pooling_functional',
    'attention_functional',
    'regularization_functional',
    'recurrent_functional',

    # Initialization (all functions take Tensor as first argument)
    'xavier_uniform_', 'xavier_normal_',
    'kaiming_uniform_', 'kaiming_normal_',
    'normal_', 'uniform_', 'zeros_', 'ones_', 'constant_',
    'orthogonal_',
    'GainConfig', 'ActivationConfig',
    'calculate_fan_in_fan_out',

    # Positional encodings
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEmbedding',
    'RelativePositionalEmbedding',
    'RotaryPositionalEmbedding',
    'ALiBiPositionalBias',
    'create_sinusoidal_encoding',
    'create_rope_encoding',
    'compare_positional_encodings',
]
