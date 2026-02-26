"""
Transformer Decoder Implementation (GPT-style Architecture)

Module: sequence.transformers.decoder

IMPLEMENTATION STATUS:
    - [ ] Causal self-attention (autoregressive masking)
    - [ ] Cross-attention for encoder-decoder models
    - [ ] Position-wise feed-forward network
    - [ ] Residual connections & layer normalization
    - [ ] N stacked decoder layers
    - [ ] Positional encodings
    - [ ] Causal masking generation

COMPLEXITY:
    Time:  O(n^2 * d) per layer for causal self-attention
           For generation: O(T^2) where T is generation length
    Space: O(n * d) for storing activations
    Params: ~7.1M per layer (for d_model=768)

PREREQUISITES:
    - Understanding of attention mechanisms
    - Knowledge of causal masking and autoregressive generation
    - Familiarity with masked language modeling
    - PyTorch basics

REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017) Section 3.2.3
    - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
    - "Unified Transformer Tracker for Object Tracking" - Causal masking techniques

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
   - Critical for generation tasks and autoregressive language modeling

2. Three Types of Attention in Seq2Seq Decoders:
   a) Causal Self-Attention (on decoder tokens)
   b) Cross-Attention (decoder attends to encoder output)
   c) Encoder Self-Attention (in encoder, separate from decoder)

3. Key Differences from Encoder:
   - Causal masking prevents attending to future tokens
   - Can have cross-attention to encoder output
   - Often used alone (decoder-only) for language models like GPT
   - Autoregressive inference: generate tokens one at a time

4. Positional Encodings:
   - Same as encoder (sinusoidal or learnable)
   - Critical for decoder to track position in generation

5. Efficient Inference:
   - During training: process full sequences (still respect causality)
   - During inference: only process current token + KV cache from previous tokens
   - KV cache reduces computation from O(n^2) to O(n)

================================================================================
MATHEMATICAL FORMULATION
================================================================================

CAUSAL MASK GENERATION:
    For sequence length n, create lower triangular mask:

    Causal_Mask[i, j] = 1 if j <= i
                       = 0 if j > i

    Example (4x4):
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]

    Applied to attention scores BEFORE softmax:
    attention_scores = attention_scores.masked_fill(~causal_mask, -1e9)
    softmax(-1e9) ≈ 0

CAUSAL SELF-ATTENTION:
    Same as encoder self-attention but with causal mask:

    Q = X @ W_Q  ,  K = X @ W_K  ,  V = X @ W_V

    scores = (Q @ K^T) / sqrt(d_k)
    scores = scores.masked_fill(~causal_mask, -1e9)  # <-- KEY DIFFERENCE

    Attention(Q, K, V) = softmax(scores) @ V

CROSS-ATTENTION (in Seq2Seq models):
    Decoder attends to encoder output:

    Q_decoder = decoder_x @ W_Q      [batch, dec_seq, d_k]
    K_encoder = encoder_out @ W_K    [batch, enc_seq, d_k]
    V_encoder = encoder_out @ W_V    [batch, enc_seq, d_v]

    CrossAttn = softmax((Q @ K^T) / sqrt(d_k)) @ V   (no causal mask here)

DECODER LAYER (with self-attention and cross-attention):
    x' = x + MultiHeadSelfAttention_Causal(LN(x), LN(x), LN(x), causal_mask)
    x'' = x' + MultiHeadCrossAttention(LN(x'), encoder_out, encoder_out)
    x''' = x'' + FFN(LN(x''))

DECODER-ONLY MODEL (like GPT):
    Just uses causal self-attention, no cross-attention:

    x' = x + MultiHeadSelfAttention_Causal(LN(x), LN(x), LN(x), causal_mask)
    x'' = x' + FFN(LN(x'))

================================================================================
ARCHITECTURE OVERVIEW: N-Layer Decoder Stack (Decoder-Only)
================================================================================

Input Shape: [batch_size, seq_len, d_model]
    |
    v
[Token Embeddings + Positional Embeddings]
    |
    v
Decoder Layer 0:
    |-- Causal Multi-Head Self-Attention (with causal mask)
    |-- Add & Norm
    |-- Position-wise Feed-Forward
    |-- Add & Norm
    v
Decoder Layer 1:
    |-- (Same as Layer 0)
    v
    ...
Decoder Layer N-1:
    |-- (Same as Layer 0)
    v
Output: [batch_size, seq_len, d_model]
    |
    v
Language Modeling Head: Linear layer to vocab
    |
    v
Logits: [batch_size, seq_len, vocab_size]

GENERATION PROCESS:
    1. Pass prompt through decoder (compute all positions)
    2. Extract last token logits: logits[-1, :]
    3. Sample or argmax to get next token
    4. Append to sequence and repeat (or use KV cache for efficiency)

================================================================================
FORWARD PASS SHAPE DOCUMENTATION
================================================================================

Input (Training):
    x:                  [batch_size, seq_len, d_model]
    encoder_output:     [batch_size, enc_seq, d_model] (if seq2seq)
    causal_mask:        [seq_len, seq_len] bool (lower triangular)

Inside Causal Self-Attention:
    Q, K, V:            [batch_size, seq_len, d_model]
    Q_heads:            [batch_size, num_heads, seq_len, d_k]
    K_heads:            [batch_size, num_heads, seq_len, d_k]
    V_heads:            [batch_size, num_heads, seq_len, d_v]
    attention_scores:   [batch_size, num_heads, seq_len, seq_len]
    causal_mask:        [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
    masked_scores:      [batch_size, num_heads, seq_len, seq_len] (-inf at future)
    attention_weights:  [batch_size, num_heads, seq_len, seq_len]
    output:             [batch_size, seq_len, d_model]

Input (Generation / Inference):
    x:                  [batch_size, 1, d_model] (single token at each step)
    k_cache, v_cache:   [batch_size, num_heads, prev_len, d_k/d_v]

    Process:
    1. Compute Q from current token
    2. Compute K, V from current token
    3. Append K, V to cache
    4. Attention uses cached K, V (seq_len at t becomes t+1)
    5. Output: [batch_size, 1, d_model]
    6. Project to vocab size: [batch_size, 1, vocab_size]
    7. Sample/argmax to get next token
    8. Repeat from step 1 with new token

================================================================================
KEY DIFFERENCES: DECODER vs ENCODER
================================================================================

ENCODER (BERT-style):
    - Bidirectional attention (each token sees all others)
    - No causal mask
    - No cross-attention (uses self-attention only)
    - Pre-training: masked language modeling
    - Inference: process full sequence at once
    - Applications: classification, NER, understanding

DECODER (GPT-style):
    - Unidirectional/causal attention (token sees only past and itself)
    - Causal mask applied in self-attention
    - May have cross-attention (if seq2seq)
    - Pre-training: next token prediction (autoregressive)
    - Inference: generation token by token
    - Applications: language generation, continuation

================================================================================
"""

import math
import numpy as np
from typing import Optional, Tuple, List

from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.normalization.layernorm import LayerNorm
from python.nn_core.attention.multihead import MultiHeadAttention
from python.nn_core.regularization.dropout import Dropout


def create_causal_mask(seq_len: int, device: str) -> np.ndarray:
    """
    Create a causal (lower triangular) attention mask.

    Prevents attention to future tokens in autoregressive generation.

    Args:
        seq_len (int): Length of sequence
        device (device): Device to place mask on

    Returns:
        np.ndarray: [seq_len, seq_len] bool tensor
            True = attend to this position
            False = mask out this position

    Mathematical Example (4x4):
        [[T, F, F, F],    # pos 0 attends to pos 0 only
         [T, T, F, F],    # pos 1 attends to pos 0,1
         [T, T, T, F],    # pos 2 attends to pos 0,1,2
         [T, T, T, T]]    # pos 3 attends to pos 0,1,2,3

    Note:
        Lower triangular matrix where mask[i, j] = True if j <= i
    """
    raise NotImplementedError(
        "create_causal_mask not yet implemented.\n"
        "TODO:\n"
        "  1. Create a boolean tensor of shape [seq_len, seq_len] on device\n"
        "  2. Fill with lower triangular matrix: np.tril(np.ones(...))\n"
        "  3. Convert to bool\n"
        "  4. Return mask\n"
        "\nHint: np.tril fills lower triangular with 1s, rest with 0s"
    )


class CausalSelfAttention(Module):
    """
    Causal (Autoregressive) Multi-Head Self-Attention.

    Used in decoder-only models like GPT. Each position can attend to
    positions at or before it, but not future positions.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability. Default: 0.1
        use_kv_cache (bool): Enable KV caching for efficient generation

    Shape:
        Input:  (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    Example:
        >>> attn = CausalSelfAttention(d_model=768, num_heads=12)
        >>> x = np.random.randn(2, 10, 768)
        >>> output = attn(x)
        >>> output.shape
        Array shape([2, 10, 768])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_kv_cache: bool = False,
    ):
        """
        Initialize causal self-attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_kv_cache: Enable KV cache for generation
        """
        super().__init__()
        raise NotImplementedError(
            "CausalSelfAttention.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Store d_model, num_heads, use_kv_cache\n"
            "  2. Calculate d_k = d_model // num_heads\n"
            "  3. Create linear layers for Q, K, V projections\n"
            "  4. Create output projection (W_O)\n"
            "  5. Create dropout layer\n"
            "  6. If use_kv_cache: initialize k_cache and v_cache as None\n"
            "  7. Create and register causal mask (as buffer, not parameter)"
        )

    def forward(
        self,
        x: np.ndarray,
        use_cache: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Apply causal self-attention.

        Args:
            x: [batch_size, seq_len, d_model] - Input
            use_cache: If True, return cached K, V for next generation step

        Returns:
            output: [batch_size, seq_len, d_model]
            cache: (k_cache, v_cache) if use_cache=True, else None
                   k_cache, v_cache: [batch_size, num_heads, seq_len, d_k/d_v]

        Process:
            1. Project to Q, K, V
            2. Split into num_heads
            3. Compute attention scores with causal mask
            4. Apply softmax
            5. Multiply by values
            6. Concatenate heads and project

        Causal Masking:
            - Before softmax, set attention_scores[i, j] = -inf if j > i
            - softmax(-inf) = 0, so future positions have zero attention
        """
        raise NotImplementedError(
            "CausalSelfAttention.forward not yet implemented.\n"
            "TODO:\n"
            "  1. Get batch_size, seq_len from x\n"
            "  2. Project to Q, K, V: [batch, seq_len, d_model]\n"
            "  3. Reshape for multi-head: [batch, seq_len, num_heads, d_k]\n"
            "  4. Transpose to: [batch, num_heads, seq_len, d_k]\n"
            "  5. Compute attention_scores = Q @ K^T / sqrt(d_k)\n"
            "  6. Create causal mask [seq_len, seq_len] and expand to batch\n"
            "  7. Apply mask: scores[~mask] = -1e9\n"
            "  8. Apply softmax along key dimension\n"
            "  9. Multiply by V\n"
            "  10. Concatenate heads and project to d_model\n"
            "  11. If use_cache: return (output, (K, V)), else (output, None)\n"
            "\nHint: For generation, K and V contain full history for next token"
        )

    def reset_cache(self):
        """Reset KV cache for starting new generation sequence."""
        raise NotImplementedError(
            "CausalSelfAttention.reset_cache not yet implemented.\n"
            "TODO: Set k_cache = None and v_cache = None"
        )


class DecoderLayer(Module):
    """
    Single Transformer Decoder Layer (Decoder-Only).

    Consists of:
    1. Causal Multi-Head Self-Attention
    2. Residual connection + Layer Normalization
    3. Position-wise Feed-Forward Network
    4. Residual connection + Layer Normalization

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward inner dimension. Default: 4*d_model
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'relu'
        use_kv_cache (bool): Enable KV caching for generation

    Shape:
        Input:  [batch_size, seq_len, d_model]
        Output: [batch_size, seq_len, d_model]

    Note:
        This is the decoder-only variant (like GPT).
        For seq2seq decoders, add a cross-attention layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_kv_cache: bool = False,
    ):
        """
        Initialize a decoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function
            use_kv_cache: Enable KV cache
        """
        super().__init__()
        raise NotImplementedError(
            "DecoderLayer.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create CausalSelfAttention module\n"
            "  2. Create FeedForwardNetwork module\n"
            "  3. Create two LayerNorm instances\n"
            "  4. Create dropout layer\n"
            "  5. Store d_model"
        )

    def forward(
        self,
        x: np.ndarray,
        use_cache: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Apply transformer decoder layer.

        Args:
            x: [batch_size, seq_len, d_model] - Input
            use_cache: If True, cache K,V for generation

        Returns:
            output: [batch_size, seq_len, d_model]
            cache: (k, v) tensors for generation, or None

        Process:
            1. Apply causal self-attention
            2. Add residual connection
            3. Apply feed-forward
            4. Add residual connection
            5. Return output and cache
        """
        raise NotImplementedError(
            "DecoderLayer.forward not yet implemented.\n"
            "TODO:\n"
            "  1. attn_out, cache = self_attn(x, use_cache=use_cache)\n"
            "  2. x = x + dropout(attn_out)\n"
            "  3. x = ln_1(x)\n"
            "  4. ffn_out = ffn(x)\n"
            "  5. x = x + dropout(ffn_out)\n"
            "  6. x = ln_2(x)\n"
            "  7. return x, cache"
        )


class TransformerDecoder(Module):
    """
    Transformer Decoder Stack (Decoder-Only, GPT-style).

    N stacked decoder layers with causal self-attention.
    Can be used for:
    - Autoregressive language generation (GPT-1, GPT-2, GPT-3)
    - Text completion
    - Pre-training with next-token prediction

    Args:
        d_model (int): Model dimension. Default: 768 (GPT-2 small)
        num_heads (int): Number of attention heads. Default: 12 (GPT-2 small)
        num_layers (int): Number of decoder layers. Default: 12 (GPT-2 small)
        d_ff (int): Feed-forward hidden dimension. Default: 4*d_model
        vocab_size (int): Vocabulary size. Default: 50257 (GPT-2)
        max_seq_len (int): Maximum sequence length. Default: 1024
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN ('relu' or 'gelu'). Default: 'gelu'
        use_kv_cache (bool): Enable KV cache for generation. Default: True

    Shape:
        Input:  [batch_size, seq_len]
        Output: [batch_size, seq_len, d_model]
        Logits: [batch_size, seq_len, vocab_size]

    Example:
        >>> decoder = TransformerDecoder(d_model=768, num_heads=12, num_layers=12)
        >>> input_ids = np.random.randint(0, 50257, (2, 128))
        >>> output = decoder(input_ids)
        >>> output.shape
        Array shape([2, 128, 768])

    GENERATION EXAMPLE:
        >>> decoder = TransformerDecoder()
        >>> decoder.eval()
        >>> prompt = torch.tensor([[1, 2, 3]])  # [batch=1, seq=3]
        >>>
        >>> # Generate 50 tokens
        >>> for _ in range(50):
        ...     logits = decoder(prompt)  # [1, seq_len, vocab]
        ...     next_logits = logits[:, -1, :]  # [1, vocab]
        ...     next_token = next_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
        ...     prompt = torch.cat([prompt, next_token], dim=1)
        >>>
        >>> generated = prompt[0].tolist()  # Convert to list of token IDs
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
        use_kv_cache: bool = True,
    ):
        """
        Initialize transformer decoder.

        Args:
            d_model: Model/embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function
            use_kv_cache: Enable KV cache for generation
        """
        super().__init__()
        raise NotImplementedError(
            "TransformerDecoder.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Store hyperparameters\n"
            "  2. Create Embedding(vocab_size, d_model)\n"
            "  3. Create positional embeddings (learnable)\n"
            "  4. Create ModuleList with num_layers DecoderLayer instances\n"
            "  5. Create final LayerNorm\n"
            "  6. Create language modeling head: Linear(d_model, vocab_size)\n"
            "  7. Create embedding dropout\n"
            "  8. Store max_seq_len, vocab_size"
        )

    def forward(
        self,
        input_ids: np.ndarray,
        use_cache: bool = False,
    ) -> np.ndarray:
        """
        Decode input tokens through decoder stack.

        Args:
            input_ids: [batch_size, seq_len] - Token indices
            use_cache: If True, cache K,V for generation

        Returns:
            logits: [batch_size, seq_len, vocab_size] - Logits for next token

        Process:
            1. Embed tokens and add positional encodings
            2. Apply embedding dropout
            3. Pass through N decoder layers (with causal masking)
            4. Apply final layer normalization
            5. Project to vocabulary size

        Note:
            For generation, only the last token's logits are used:
            next_token_logits = logits[:, -1, :]
        """
        raise NotImplementedError(
            "TransformerDecoder.forward not yet implemented.\n"
            "TODO:\n"
            "  1. x = token_embeddings(input_ids)\n"
            "  2. seq_len = x.size(1)\n"
            "  3. positions = np.arange(seq_len, device=x.device)\n"
            "  4. x = x + pos_embeddings(positions)\n"
            "  5. x = dropout(x)\n"
            "  6. For each decoder_layer:\n"
            "     x, _ = decoder_layer(x, use_cache=use_cache)\n"
            "  7. x = final_ln(x)\n"
            "  8. logits = lm_head(x)\n"
            "  9. return logits"
        )

    def generate(
        self,
        input_ids: np.ndarray,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: [batch_size, seq_len] - Initial prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely next tokens
            top_p: If set, only sample from tokens with cumulative prob <= top_p

        Returns:
            generated: [batch_size, seq_len + max_length] - Full sequence

        Algorithm:
            1. Pass prompt through decoder
            2. Extract last token logits
            3. Apply temperature scaling
            4. Apply top-k filtering (optional)
            5. Apply nucleus sampling (top-p) (optional)
            6. Sample next token
            7. Append to sequence
            8. Repeat until max_length tokens generated

        Sampling Strategies:
            - Greedy: argmax(logits) - deterministic but can get stuck
            - Temperature sampling: softmax(logits / T)
              T=1: normal distribution
              T>1: more uniform (more random)
              T<1: sharper (more confident)
            - Top-k: only consider k most likely tokens
            - Top-p (nucleus): only consider tokens with cumulative prob <= p
        """
        raise NotImplementedError(
            "TransformerDecoder.generate not yet implemented.\n"
            "TODO: Implement autoregressive generation:\n"
            "  1. Model should be in eval mode\n"
            "  2. For each generation step:\n"
            "     a. Pass current sequence through model\n"
            "     b. Extract logits for last token\n"
            "     c. Apply temperature scaling\n"
            "     d. Apply top-k filtering if specified\n"
            "     e. Apply top-p sampling if specified\n"
            "     f. Sample next token (or argmax for greedy)\n"
            "     g. Append to sequence\n"
            "  3. Stop when max_length reached\n"
            "  4. Return generated sequence\n"
            "\nHelper functions to implement:\n"
            "  - _apply_temperature_scaling\n"
            "  - _apply_top_k_filtering\n"
            "  - _apply_top_p_sampling\n"
            "  - _sample_from_logits"
        )

    def get_attention_weights(self) -> List[np.ndarray]:
        """
        Extract attention weights from all layers for visualization.

        Returns:
            List of attention weight tensors from each decoder layer.
            Each tensor shape: [batch_size, num_heads, seq_len, seq_len]
        """
        raise NotImplementedError(
            "TransformerDecoder.get_attention_weights not yet implemented.\n"
            "TODO: Store and return attention weights from all decoder layers"
        )


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


# Alias for common naming
TransformerDecoderLayer = DecoderLayer

