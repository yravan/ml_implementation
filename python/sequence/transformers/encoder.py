"""
Transformer Encoder Implementation (Bidirectional)

Module: sequence.transformers.encoder

COMPLEXITY:
    Time:  O(n^2 * d) per layer for self-attention (quadratic in sequence length)
    Space: O(n * d) for storing activations
    Params: ~7.1M per layer (for d_model=768)

REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
      (Devlin et al., 2018)
    - "Layer Normalization" (Ba et al., 2016)

================================================================================
THEORY: Transformer Encoder Architecture
================================================================================

The encoder is the foundation of bidirectional transformer models like BERT.
It processes the ENTIRE input sequence at once without causal constraints,
allowing each token to attend to all other tokens (past, present, and future).

KEY DESIGN PRINCIPLES:
1. Self-Attention Mechanism:
   - Allows parallel processing of sequences
   - Each position can directly interact with all others
   - Replaces RNNs and enables better gradient flow
   - Facilitates transfer learning through pre-training

2. Multi-Head Attention:
   - Splits representation into multiple "subspaces"
   - Each head learns different aspects (syntax, semantics, position)
   - Improves model expressiveness with minimal overhead
   - Concatenate heads and project back to d_model

3. Position-wise Feed-Forward:
   - Two dense layers with activation: d_model -> d_ff -> d_model
   - Typically d_ff = 4 * d_model for good capacity
   - Applied independently to each position (no sequential dependency)

4. Residual Connections & Layer Norm:
   - Pre-LN structure: x + Sublayer(LN(x))
   - Stabilizes training and allows deeper networks
   - Critical for training stability

5. Bidirectional Context:
   - Unlike decoders, no causal masking
   - Each token can see all surrounding context
   - Foundation for pre-training objectives (MLM - Masked Language Modeling)

================================================================================
MATHEMATICAL FORMULATION
================================================================================

SELF-ATTENTION (Single Head):
    Q = X @ W_Q  ,  K = X @ W_K  ,  V = X @ W_V      [batch, seq_len, d_k]

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V    [batch, seq_len, d_v]

MULTI-HEAD ATTENTION:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O    [batch, seq_len, d_model]

    head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)

FEED-FORWARD NETWORK:
    FFN(x) = activation(x @ W_1 + b_1) @ W_2 + b_2        [batch, seq_len, d_model]

    Where W_1: [d_model, d_ff], W_2: [d_ff, d_model]

ENCODER LAYER (Pre-LN):
    x' = x + MultiHeadAttention(LN(x), LN(x), LN(x))
    x'' = x' + FFN(LN(x'))

POSITIONAL ENCODING:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

================================================================================
"""

import numpy as np
from typing import Optional, List

from python.foundations import Tensor
from python.nn_core import Module, Parameter, ModuleList, Sequential
from python.nn_core.linear import Linear
from python.nn_core.normalization import LayerNorm
from python.nn_core.attention import MultiHeadAttention
from python.nn_core.regularization import Dropout
from python.nn_core.activations import GELU, ReLU
from python.nn_core.positional import SinusoidalPositionalEncoding, LearnedPositionalEmbedding


class EncoderLayer(Module):
    """
    Single Transformer Encoder Layer.

    A bidirectional encoder block with multi-head self-attention,
    a position-wise feed-forward network, pre-layer-normalization,
    and residual connections.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward inner dimension. Default: 4 * d_model
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN ('relu' or 'gelu'). Default: 'relu'

    Shape:
        Input:  [batch_size, seq_len, d_model]
        Output: [batch_size, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.pre_attn_norm = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.post_attn_norm = LayerNorm(d_model)
        if activation == "gelu":
            self.ffn = Sequential(Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model))
        else:
            self.ffn = Sequential(Linear(d_model, d_ff), ReLU(), Linear(d_ff, d_model))

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Apply transformer encoder layer.

        Args:
            x: [batch_size, seq_len, d_model]
            padding_mask: [batch_size, seq_len] bool (True=valid, False=pad)

        Returns:
            [batch_size, seq_len, d_model]
        """
        # Convert bool mask to float, reshape [batch, seq] -> [batch, 1, 1, seq]
        mask = padding_mask
        if mask is not None:
            m = mask.data
            if m.dtype == bool or m.dtype == np.bool_:
                m = np.where(m, 0.0, -np.inf).astype(np.float32)
            if m.ndim == 2:
                m = m[:, np.newaxis, np.newaxis, :]
            mask = Tensor(m)
        residual = x
        x = self.pre_attn_norm(x)
        x = self.self_attn(x, x, x, mask)
        x = x + residual

        residual = x
        x = self.post_attn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


class TransformerEncoder(Module):
    """
    Complete Transformer Encoder Stack (Bidirectional).

    Stacks N encoder layers with token embeddings and positional encodings.
    Suitable for classification, NER, masked language modeling, and any task
    where full bidirectional context is available.

    Args:
        d_model (int): Model dimension. Default: 768
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of encoder layers. Default: 12
        d_ff (int): Feed-forward hidden dimension. Default: 4*d_model
        vocab_size (int): Vocabulary size. Default: 30522
        max_seq_len (int): Maximum sequence length. Default: 512
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'
        use_learnable_pos_embed (bool): Use learnable positional embeddings. Default: True

    Shape:
        Input:  [batch_size, seq_len]
        Output: [batch_size, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: Optional[int] = None,
        vocab_size: int = 30522,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_learnable_pos_embed: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.use_learnable_pos_embed = use_learnable_pos_embed
        if use_learnable_pos_embed:
            self.positional_encoding = LearnedPositionalEmbedding(max_seq_len, d_model)
        else:
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.token_embedding = Parameter(
            np.random.randn(vocab_size, d_model).astype(np.float64) * 0.02
        )
        self.drop = Dropout(dropout)
        self.layers = ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        input_ids: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode input tokens through transformer stack.

        Args:
            input_ids: [batch_size, seq_len] - Token indices
            padding_mask: [batch_size, seq_len] float mask (0/-inf)

        Returns:
            [batch_size, seq_len, d_model] - Encoded representations
        """
        B, T = input_ids.data.shape
        ids = input_ids.data.astype(int)
        x = Tensor(
            self.token_embedding.data[ids.flatten()].reshape(B, T, self.d_model),
            requires_grad=True,
        )

        # Positional encoding
        if self.use_learnable_pos_embed:
            pos_ids = np.arange(T)
            pos_embeds = self.positional_encoding(pos_ids)
            x = x + pos_embeds
        else:
            x = self.positional_encoding(x)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x, padding_mask)

        return x


# Configuration dictionaries for common models
BERT_CONFIG = {
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "d_ff": 3072,
    "vocab_size": 30522,
    "max_seq_len": 512,
    "dropout": 0.1,
    "activation": "gelu",
}

BERT_LARGE_CONFIG = {
    "d_model": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "d_ff": 4096,
    "vocab_size": 30522,
    "max_seq_len": 512,
    "dropout": 0.1,
    "activation": "gelu",
}

ROBERTA_CONFIG = {
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "d_ff": 3072,
    "vocab_size": 50265,
    "max_seq_len": 514,
    "dropout": 0.1,
    "activation": "gelu",
}

TransformerEncoderLayer = EncoderLayer
