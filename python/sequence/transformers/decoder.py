"""
Transformer Decoder Implementation (Autoregressive)

Module: sequence.transformers.decoder

COMPLEXITY:
    Time:  O(n^2 * d) per layer for causal self-attention
    Space: O(n * d) for storing activations
    Params: ~7.1M per layer (for d_model=768)

REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017) Section 3.2.3
    - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)

================================================================================
THEORY: Transformer Decoder Architecture
================================================================================

The decoder processes sequences AUTOREGRESSIVELY, where each token can only
attend to previous tokens (and itself), not future tokens. This is essential
for language generation where we predict tokens one at a time.

KEY DESIGN PRINCIPLES:
1. Causal Self-Attention (Autoregressive Attention):
   - Each position can ONLY attend to its own position and earlier positions
   - Implemented via causal mask: mask[i, j] = -inf if j > i
   - Prevents the model from "cheating" by looking at future information

2. Key Differences from Encoder:
   - Causal masking prevents attending to future tokens
   - Can have cross-attention to encoder output (in seq2seq)
   - Often used alone (decoder-only) for language models like GPT
   - Autoregressive inference: generate tokens one at a time

3. Efficient Inference with KV Cache:
   - During training: process full sequences (still respect causality)
   - During inference: only process current token + KV cache from previous tokens
   - KV cache reduces computation from O(n^2) to O(n)

================================================================================
MATHEMATICAL FORMULATION
================================================================================

CAUSAL MASK GENERATION:
    Causal_Mask[i, j] = 1 if j <= i, 0 if j > i

CAUSAL SELF-ATTENTION:
    Q = X @ W_Q  ,  K = X @ W_K  ,  V = X @ W_V
    scores = (Q @ K^T) / sqrt(d_k)
    scores = scores.masked_fill(~causal_mask, -inf)
    Attention(Q, K, V) = softmax(scores) @ V

DECODER LAYER (Pre-LN, decoder-only):
    x' = x + CausalMultiHeadAttention(LN(x))
    x'' = x' + FFN(LN(x'))

================================================================================
"""

import numpy as np
from typing import Optional, Tuple, List

from python.foundations import Tensor
from python.nn_core import Module, Parameter, ModuleList, Sequential
from python.nn_core.linear import Linear
from python.nn_core.normalization import LayerNorm
from python.nn_core.attention import MultiHeadAttention, CausalMask
from python.nn_core.regularization import Dropout
from python.nn_core.activations import GELU, ReLU
from python.nn_core.positional import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEncoding,
)


class DecoderLayer(Module):
    """
    Single Transformer Decoder Layer (Decoder-Only).

    A decoder block that enforces the autoregressive constraint via causal
    masking. Contains causal multi-head self-attention, a position-wise
    feed-forward network, pre-layer-normalization, and residual connections.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward inner dimension. Default: 4*d_model
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'relu'

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
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply transformer decoder layer.

        Args:
            x: [batch_size, seq_len, d_model]
            causal_mask: [seq_len, seq_len] causal attention mask

        Returns:
            [batch_size, seq_len, d_model]
        """
        residual = x
        x = self.pre_attn_norm(x)
        x = self.self_attn(x, x, x, causal_mask)
        x = x + residual

        residual = x
        x = self.post_attn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


class TransformerDecoder(Module):
    """
    Full Autoregressive Decoder with Generation Support.

    Stacks N decoder layers with token and learned positional embeddings,
    and includes an LM head for next-token prediction. Supports autoregressive
    generation with temperature, top-k, and top-p sampling strategies.

    Args:
        d_model (int): Model dimension. Default: 768
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of decoder layers. Default: 12
        d_ff (int): Feed-forward hidden dimension. Default: 4*d_model
        vocab_size (int): Vocabulary size. Default: 50257
        max_seq_len (int): Maximum sequence length. Default: 1024
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'

    Shape:
        Input:  [batch_size, seq_len]
        Output: [batch_size, seq_len, vocab_size] (logits)
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: Optional[int] = None,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_learnable_pos_embed: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_learnable_pos_embed = use_learnable_pos_embed
        # Causal mask: 0 for attend, -inf for don't attend
        # Lower triangular = attend to current and past positions
        self._causal_mask = np.where(
            np.tril(np.ones((max_seq_len, max_seq_len))),
            0.0,
            -np.inf,
        ).astype(np.float32)
        if use_learnable_pos_embed:
            self.positional_encoding = LearnedPositionalEmbedding(max_seq_len, d_model)
        else:
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.token_embedding = Parameter(
            np.random.randn(vocab_size, d_model).astype(np.float64) * 0.02
        )
        self.drop = Dropout(dropout)
        self.layers = ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        """
        Decode input tokens through decoder stack.

        Args:
            input_ids: [batch_size, seq_len] - Token indices

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        B, T = input_ids.data.shape
        # Token embeddings: index into embedding table
        ids = input_ids.data.astype(int)
        x = Tensor(self.token_embedding.data[ids.flatten()].reshape(B, T, self.d_model),
                    requires_grad=True)

        # Add positional encoding
        if self.use_learnable_pos_embed:
            pos_ids = np.arange(T)
            pos_embeds = self.positional_encoding(pos_ids)  # [T, d_model]
            x = x + pos_embeds
        else:
            x = self.positional_encoding(x)  # adds in-place
        x = self.drop(x)

        # Create causal mask for this sequence length
        mask = Tensor(self._causal_mask[:T, :T])

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm + project to vocab
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
    ) -> Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: [batch_size, seq_len] - Initial prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens
            top_p: Nucleus sampling threshold

        Returns:
            generated: [batch_size, seq_len + max_length]
        """
        self.eval()
        generated = input_ids.data.astype(int).copy()  # [B, T] numpy array

        for _ in range(max_length):
            # Forward pass on current sequence
            logits = self.forward(Tensor(generated))
            # Get logits for the last position
            next_logits = logits.data[:, -1, :].copy()  # [B, vocab_size]

            # Temperature scaling
            if temperature > 1e-8:
                next_logits = next_logits / temperature

            # Top-k filtering: zero out everything except top-k
            if top_k is not None:
                for b in range(next_logits.shape[0]):
                    top_k_idx = np.argpartition(next_logits[b], -top_k)[-top_k:]
                    mask = np.full(next_logits.shape[1], -np.inf)
                    mask[top_k_idx] = next_logits[b, top_k_idx]
                    next_logits[b] = mask

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                for b in range(next_logits.shape[0]):
                    sorted_idx = np.argsort(next_logits[b])[::-1]
                    sorted_logits = next_logits[b, sorted_idx]
                    # Convert to probs for cumsum
                    sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    cumulative_probs = np.cumsum(sorted_probs)
                    # Remove tokens with cumulative prob above threshold
                    cutoff = np.searchsorted(cumulative_probs, top_p) + 1
                    remove_idx = sorted_idx[cutoff:]
                    next_logits[b, remove_idx] = -np.inf

            # Convert logits to probabilities via softmax
            max_logits = np.max(next_logits, axis=1, keepdims=True)
            exp_logits = np.exp(next_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Sample next token (greedy if temperature is very low)
            if temperature < 0.05:
                next_tokens = np.argmax(probs, axis=1).reshape(-1, 1)
            else:
                next_tokens = np.array([
                    np.random.choice(probs.shape[1], p=probs[b])
                    for b in range(probs.shape[0])
                ]).reshape(-1, 1)

            # Append to sequence
            generated = np.concatenate([generated, next_tokens], axis=1)

        return Tensor(generated)



# Configuration dictionaries for common models
GPT2_SMALL_CONFIG = {
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "d_ff": 3072,
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "activation": "gelu",
}

GPT2_MEDIUM_CONFIG = {
    "d_model": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "d_ff": 4096,
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "activation": "gelu",
}

GPT2_LARGE_CONFIG = {
    "d_model": 1280,
    "num_heads": 20,
    "num_layers": 36,
    "d_ff": 5120,
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "activation": "gelu",
}

GPT2_XL_CONFIG = {
    "d_model": 1600,
    "num_heads": 25,
    "num_layers": 48,
    "d_ff": 6400,
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "activation": "gelu",
}

TransformerDecoderLayer = DecoderLayer
