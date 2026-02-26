"""
GPT (Generative Pre-trained Transformer) Implementation

Module: sequence.transformers.gpt

IMPLEMENTATION STATUS:
    - [ ] Causal self-attention mechanism
    - [ ] Multi-head attention with KV cache
    - [ ] Feed-forward networks with GELU activation
    - [ ] Positional embeddings (learnable or sinusoidal)
    - [ ] Autoregressive token generation
    - [ ] Sampling strategies (temperature, top-k, top-p, beam search)
    - [ ] Attention weight extraction
    - [ ] Model checkpointing and loading

COMPLEXITY:
    Time:  O(n^2 * d) for forward pass (n=seq_len, d=d_model)
           O(T^2 * d) for generation (T=total_length)
           O(T * d) with KV cache (linear in generation length)
    Space: O(n * d) for activations
           O(T * d) for KV cache
    Params: ~125M (GPT-2 small), ~355M (GPT-2 medium), ~774M (GPT-2 large)

PREREQUISITES:
    - Deep understanding of transformer architecture
    - Knowledge of causal masking and autoregressive generation
    - Familiarity with token sampling strategies
    - Understanding of pre-training objectives (language modeling)
    - PyTorch intermediate to advanced skills

REFERENCES:
    - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
      https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
    - "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3
      https://arxiv.org/abs/2005.14165
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - OpenAI GPT-2 blog post: https://openai.com/blog/language-models-are-unsupervised-multitask-learners/

================================================================================
THEORY: GPT Architecture and Design Philosophy
================================================================================

GPT (Generative Pre-trained Transformer) represents a paradigm shift in NLP:
from supervised learning on specific tasks to unsupervised pre-training followed
by task-agnostic transfer learning.

KEY INNOVATIONS AND DESIGN CHOICES:

1. DECODER-ONLY ARCHITECTURE:
   - Uses only transformer decoder (no encoder)
   - Processes input unidirectionally (left-to-right)
   - Causal masking prevents attending to future tokens
   - Much simpler than encoder-decoder (like original transformer)
   - More parameter-efficient than BERT-style (bidirectional) models

2. AUTOREGRESSIVE LANGUAGE MODELING:
   - Pre-training objective: predict next token given previous tokens
   - P(x_1, x_2, ..., x_n) = ∏ P(x_i | x_1, ..., x_{i-1})
   - Simple, elegant, and naturally leads to text generation
   - Scales well to very large models (billions of parameters)
   - Different from BERT's masked language modeling (MLM)

3. BIDIRECTIONAL SCALING:
   - Unlike BERT (encoder-only), GPT can GENERATE text
   - GPT can also be used for classification (add task-specific head)
   - More general-purpose due to generation capability

4. FEW-SHOT AND ZERO-SHOT LEARNING:
   - Large models learn "in-context learning"
   - Can solve tasks with just a few examples or instruction
   - No fine-tuning needed for many tasks (prompt engineering)
   - GPT-3 showed remarkable few-shot abilities

5. SCALING LAWS:
   - Performance scales predictably with:
     a) Model size (number of parameters)
     b) Data size (number of tokens in training)
     c) Compute budget (FLOPs)
   - Formula: Loss ≈ E / (N + N_c)^α + B
     where N = parameters, E = epochs, α ≈ 0.07
   - Implies continuous improvement with larger models

6. TRAINING OPTIMIZATIONS:
   - Mixed precision training (float16 + float32)
   - Gradient accumulation (simulate larger batch sizes)
   - Gradient checkpointing (trade compute for memory)
   - Flash attention (efficient attention implementations)

================================================================================
ARCHITECTURAL DETAILS: Decoder-Only GPT Stack
================================================================================

INPUT: Text sequence as token indices
    |
    v
TOKEN EMBEDDINGS: Token ID -> d_model dimensional vector
    |
    v
POSITIONAL EMBEDDINGS: Add position information (learnable in GPT-2/3)
    |
    v
EMBEDDINGS DROPOUT: Regularization
    |
    v
TRANSFORMER BLOCKS (N layers):
    |
    +----> Block 0
    |       ├── LayerNorm (pre-normalization)
    |       ├── Causal Multi-Head Self-Attention
    |       ├── Residual Connection (skip)
    |       ├── LayerNorm
    |       ├── Position-wise Feed-Forward (MLP)
    |       └── Residual Connection
    |
    +----> Block 1
    |       ├── (same structure)
    |
    +----> Block N-1
    |
    v
FINAL LAYER NORMALIZATION
    |
    v
LANGUAGE MODELING HEAD: Linear(d_model, vocab_size)
    |
    v
LOGITS: [batch_size, seq_len, vocab_size]
    |
    v
SOFTMAX (for classification): P(next_token | history)

================================================================================
MATHEMATICAL FORMULATION: Detailed
================================================================================

TOKEN EMBEDDINGS:
    E: [vocab_size, d_model] - learnable matrix
    x_tokens = E[input_ids]  # [batch, seq_len, d_model]

POSITIONAL EMBEDDINGS:
    P: [max_seq_len, d_model] - learnable matrix
    positions = np.arange(seq_len)
    x_pos = P[positions]  # [seq_len, d_model]

INPUT EMBEDDINGS:
    x = x_tokens + x_pos + dropout(noise)  # [batch, seq_len, d_model]

TRANSFORMER BLOCK (Pre-LN, as in GPT-2/3):
    # Self-Attention with Causal Masking
    x_norm = LayerNorm(x)                    # [batch, seq_len, d_model]
    attn_out = CausalMultiHeadAttn(x_norm)  # [batch, seq_len, d_model]
    x = x + attn_out                         # Residual skip

    # Feed-Forward with Activation
    x_norm = LayerNorm(x)                    # [batch, seq_len, d_model]
    ffn_out = Linear2(GELU(Linear1(x_norm))) # [batch, seq_len, d_model]
    x = x + ffn_out                          # Residual skip

MULTI-HEAD CAUSAL ATTENTION (per-block):
    Q = x_norm @ W_Q  # [batch, seq_len, d_model]
    K = x_norm @ W_K  # [batch, seq_len, d_model]
    V = x_norm @ W_V  # [batch, seq_len, d_model]

    # Split into heads
    Q = reshape(Q, [batch, seq_len, num_heads, d_k])
    K = reshape(K, [batch, seq_len, num_heads, d_k])
    V = reshape(V, [batch, seq_len, num_heads, d_k])

    # Transpose: [batch, num_heads, seq_len, d_k]
    Q = transpose(Q)
    K = transpose(K)
    V = transpose(V)

    # Attention scores
    scores = Q @ K^T / √d_k  # [batch, num_heads, seq_len, seq_len]

    # Apply causal mask (CRITICAL for autoregression)
    causal_mask = tril(ones(seq_len, seq_len))  # Lower triangular
    scores = scores.masked_fill(~causal_mask, -∞)

    # Softmax and dropout
    weights = softmax(scores)  # [batch, num_heads, seq_len, seq_len]
    weights = dropout(weights)

    # Apply to values
    output = weights @ V  # [batch, num_heads, seq_len, d_v]

    # Concatenate heads
    output = reshape(output, [batch, seq_len, num_heads * d_v])
    output = output @ W_O  # [batch, seq_len, d_model]

    return output

FEED-FORWARD NETWORK (MLP):
    hidden = GELU(linear1(x_norm))  # [batch, seq_len, d_ff]
    output = linear2(hidden)         # [batch, seq_len, d_model]

    Where:
    - linear1: Linear(d_model, d_ff), typically d_ff = 4 * d_model
    - GELU: Gaussian Error Linear Unit activation
    - linear2: Linear(d_ff, d_model)
    - Applied position-wise (same weights for all positions)

LANGUAGE MODELING HEAD:
    logits = final_linear(x)  # [batch, seq_len, vocab_size]

LOSS FUNCTION (Training):
    # Shift targets to predict next token
    predictions = logits[:, :-1, :]    # [batch, seq_len-1, vocab_size]
    targets = input_ids[:, 1:]         # [batch, seq_len-1]

    loss = CrossEntropyLoss(predictions, targets)

================================================================================
GENERATION PROCESS: Autoregressive Decoding
================================================================================

GREEDY DECODING:
    1. Initialize: prompt = input_ids (e.g., [BOS])
    2. For t in range(max_length):
       a. logits = model(prompt)  # [batch, t+1, vocab]
       b. next_logits = logits[:, -1, :]  # [batch, vocab]
       c. next_token = argmax(next_logits)  # [batch]
       d. prompt = append(prompt, next_token)
       e. If next_token == EOS: break
    3. Return prompt

SAMPLING WITH TEMPERATURE:
    1. logits = model(prompt)
    2. logits = logits / temperature
       - T < 1: distribution becomes sharper (less random)
       - T = 1: no change
       - T > 1: distribution becomes smoother (more random)
    3. probabilities = softmax(logits)
    4. next_token = sample(probabilities)

TOP-K FILTERING:
    1. Get top-k most likely tokens
    2. Set all other token probabilities to 0
    3. Renormalize probabilities
    4. Sample from filtered distribution

TOP-P (NUCLEUS) SAMPLING:
    1. Sort tokens by probability (descending)
    2. Compute cumulative probabilities
    3. Select tokens until cumulative prob >= p
    4. Set all other probabilities to 0
    5. Renormalize and sample

KV CACHE OPTIMIZATION:
    During inference, we recompute attention at every step O(t^2)
    This can be optimized using KV caching:

    1. First token: compute Q, K, V normally
    2. Subsequent tokens: only compute Q for new token
       - Reuse K, V from all previous tokens (cached)
       - Attention: [batch, num_heads, 1, t] @ [batch, num_heads, t, d_v]
       - Result: [batch, num_heads, 1, d_v]
    3. Cache K, V for next iteration

    Benefits:
    - Reduces attention computation from O(t^2) to O(t)
    - Memory trade-off: store K, V cache [batch, num_heads, t, d_k/d_v]
    - Essential for real-time generation

BEAM SEARCH DECODING:
    Keep top-k hypotheses at each step:
    1. Initialize: beam_size hypotheses with start token
    2. For each step:
       a. Expand: each hypothesis can generate vocab_size continuations
       b. Score: compute log probability: log(P(next | history))
       c. Select: keep top beam_size by cumulative score
       d. Track: sequences that reached end-of-sequence
    3. Return: best completed sequence

================================================================================
FORWARD PASS SHAPE DOCUMENTATION
================================================================================

TRAINING FORWARD PASS:
    Input:
        input_ids: [batch_size, seq_len]  # Token indices

    Inside model:
        x = token_embed(input_ids)  # [batch, seq_len, d_model]
        x = x + pos_embed(positions)  # [batch, seq_len, d_model]
        x = dropout(x)

        For each transformer block:
            # Self-attention path
            x_norm = ln(x)  # [batch, seq_len, d_model]
            Q = x_norm @ W_Q  # [batch, seq_len, d_model]
            K = x_norm @ W_K  # [batch, seq_len, d_model]
            V = x_norm @ W_V  # [batch, seq_len, d_model]

            # Split heads: [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
            # Transpose: [batch, num_heads, seq_len, d_k]
            attn_scores = Q_heads @ K_heads^T  # [batch, num_heads, seq_len, seq_len]
            attn_scores = apply_causal_mask(attn_scores)
            attn_weights = softmax(attn_scores)  # [batch, num_heads, seq_len, seq_len]
            attn_out = attn_weights @ V_heads  # [batch, num_heads, seq_len, d_v]

            # Concatenate heads: [batch, seq_len, d_model]
            x = x + attn_out

            # Feed-forward path
            x_norm = ln(x)
            hidden = GELU(linear1(x_norm))  # [batch, seq_len, d_ff]
            x = x + linear2(hidden)  # [batch, seq_len, d_model]

        x = final_ln(x)  # [batch, seq_len, d_model]

    Output:
        logits = lm_head(x)  # [batch, seq_len, vocab_size]

GENERATION (SINGLE TOKEN):
    Input:
        prompt: [batch_size, prompt_len]  # So far generated tokens

    Processing:
        x = embed + pos_embed + dropout  # [batch, prompt_len, d_model]

        For each transformer block:
            (same as above, but only last token used for generation)

        logits = lm_head(x)  # [batch, prompt_len, vocab_size]
        next_logits = logits[:, -1, :]  # [batch, vocab_size]

    Output:
        next_logits: [batch_size, vocab_size]  # Probabilities for next token

GENERATION WITH KV CACHE:
    First token (compute full):
        Q, K, V = project(x)  # All positions
        attn_out = attention(Q, K, V)
        cache = (K, V)  # Store for reuse

    Subsequent tokens (single token input):
        Input: [batch, 1, d_model]

        Q = project_q(x)  # [batch, 1, d_model]

        Reuse cached K, V from all previous tokens
        attn_scores = Q @ K_cached^T  # [batch, num_heads, 1, prev_len+1]
        weights = softmax(attn_scores)  # [batch, num_heads, 1, prev_len+1]
        attn_out = weights @ V_cached  # [batch, num_heads, 1, d_v]

        Update cache: cache = (cat(K_cached, K_new), cat(V_cached, V_new))

    Benefit: O(t) instead of O(t^2) attention computation

================================================================================
COMMON GPT MODELS AND CONFIGURATIONS
================================================================================

GPT (OpenAI, 2018) - Original:
    - d_model: 768
    - num_heads: 12
    - num_layers: 12
    - d_ff: 3072 (4 * d_model)
    - vocab_size: 40,000
    - total_params: ~110M

GPT-2 (OpenAI, 2019):
    - GPT-2 Small:
        d_model: 768, num_heads: 12, num_layers: 12, params: 124M
    - GPT-2 Medium:
        d_model: 1024, num_heads: 16, num_layers: 24, params: 355M
    - GPT-2 Large:
        d_model: 1280, num_heads: 20, num_layers: 36, params: 774M
    - GPT-2 XL:
        d_model: 1600, num_heads: 25, num_layers: 48, params: 1.5B
    - vocab_size: 50,257

GPT-3 (OpenAI, 2020):
    - GPT-3 Small (Ada):
        d_model: 1024, num_layers: 24, params: 350M
    - GPT-3 Medium (Babbage):
        d_model: 2048, num_layers: 24, params: 1.3B
    - GPT-3 Large (Curie):
        d_model: 4096, num_layers: 32, params: 6.7B
    - GPT-3 XL (Davinci):
        d_model: 12288, num_layers: 96, params: 175B
    - vocab_size: 50,257
    - Few-shot learning capability

KEY INSIGHTS:
    - Scaling laws hold: performance improves predictably with size
    - Large models can do tasks with few examples (few-shot learning)
    - Larger models are better at following instructions
    - Model size matters more than architectural tweaks
    - Effective context length: typically 1024-2048 tokens (GPT-2/3)

================================================================================
"""

import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Union

from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.normalization.layernorm import LayerNorm
from python.nn_core.attention.multihead import MultiHeadAttention
from python.nn_core.regularization.dropout import Dropout


class CausalSelfAttention(Module):
    """
    Causal (Autoregressive) Multi-Head Self-Attention for GPT.

    Core building block of GPT models. Implements scaled dot-product attention
    with causal masking to prevent attending to future tokens.

    Mathematical Formula:
        Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V

        With causal mask:
        attention_scores[i, j] = -inf if j > i (future positions masked)

    Args:
        d_model (int): Model dimension (e.g., 768 for GPT-2 small)
        num_heads (int): Number of attention heads (e.g., 12 for GPT-2 small)
        dropout (float): Attention dropout probability. Default: 0.1
        use_cache (bool): Enable KV caching for inference. Default: True

    Shape:
        Input:  [batch_size, seq_len, d_model]
        Output: [batch_size, seq_len, d_model]

    Attributes:
        d_k (int): Dimension per attention head (d_model // num_heads)
        scale (float): Scaling factor for attention scores (1 / sqrt(d_k))

    Example:
        >>> attn = CausalSelfAttention(d_model=768, num_heads=12)
        >>> x = np.random.randn(2, 100, 768)  # [batch=2, seq=100, d_model=768]
        >>> output = attn(x)
        >>> output.shape
        Array shape([2, 100, 768])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        dropout: float = 0.1,
        use_cache: bool = True,
    ):
        """
        Initialize causal self-attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_cache: Enable KV caching

        Raises:
            AssertionError: if d_model is not divisible by num_heads
        """
        super().__init__()
        raise NotImplementedError(
            "CausalSelfAttention.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Assert d_model % num_heads == 0, else raise error\n"
            "  2. Store d_model, num_heads\n"
            "  3. Calculate d_k = d_model // num_heads\n"
            "  4. Calculate scale = 1.0 / sqrt(d_k) for attention scaling\n"
            "  5. Create Linear(d_model, d_model) for Q projection\n"
            "  6. Create Linear(d_model, d_model) for K projection\n"
            "  7. Create Linear(d_model, d_model) for V projection\n"
            "  8. Create Linear(d_model, d_model) for output projection (W_O)\n"
            "  9. Create Dropout(dropout) for attention weights\n"
            "  10. Register buffer for causal mask (lower triangular)\n"
            "  11. Store use_cache flag\n"
            "  12. Initialize k_cache and v_cache to None if use_cache=True"
        )

    def forward(
        self,
        x: np.ndarray,
        use_cache: bool = False,
        return_attention_weights: bool = False,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    ]:
        """
        Apply causal self-attention.

        Args:
            x: [batch_size, seq_len, d_model] - Input tensor
            use_cache: If True, return and cache K, V for next generation step
            return_attention_weights: If True, also return attention weights

        Returns:
            If use_cache=False and return_attention_weights=False:
                output: [batch_size, seq_len, d_model]

            If use_cache=True and return_attention_weights=False:
                output: [batch_size, seq_len, d_model]
                cache: (k, v) where k, v are [batch, num_heads, seq_len, d_k/d_v]

            If return_attention_weights=True:
                output: [batch_size, seq_len, d_model]
                weights: [batch_size, num_heads, seq_len, seq_len]

        Process:
            1. Project x to Query, Key, Value
            2. Reshape for multi-head attention
            3. Compute attention scores: Q @ K^T / sqrt(d_k)
            4. Apply causal mask (set future positions to -inf)
            5. Apply softmax to get attention weights
            6. Apply dropout
            7. Multiply by values
            8. Concatenate heads
            9. Apply output projection
            10. Optionally cache K, V for generation
        """
        raise NotImplementedError(
            "CausalSelfAttention.forward not yet implemented.\n"
            "TODO:\n"
            "  1. batch_size, seq_len, _ = x.shape\n"
            "  2. Q = W_q(x)  # [batch, seq_len, d_model]\n"
            "  3. K = W_k(x)  # [batch, seq_len, d_model]\n"
            "  4. V = W_v(x)  # [batch, seq_len, d_model]\n"
            "  5. # Reshape for multi-head: split d_model into num_heads\n"
            "     Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)\n"
            "     K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)\n"
            "     V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)\n"
            "     # Shapes: [batch, num_heads, seq_len, d_k]\n"
            "  6. # Compute attention scores\n"
            "     scores = Q @ K.transpose(-2, -1) * scale  # [batch, heads, seq, seq]\n"
            "  7. # Apply causal mask\n"
            "     scores = scores.masked_fill(causal_mask[:seq_len, :seq_len] == 0, -1e9)\n"
            "  8. # Softmax and dropout\n"
            "     weights = softmax(scores, dim=-1)  # [batch, heads, seq, seq]\n"
            "     weights = dropout(weights)\n"
            "  9. # Apply to values\n"
            "     out = weights @ V  # [batch, heads, seq, d_k]\n"
            "  10. # Concatenate heads and project\n"
            "      out = out.transpose(1, 2)  # [batch, seq, heads, d_k]\n"
            "      out = out.contiguous().view(batch, seq, d_model)\n"
            "      out = W_o(out)  # [batch, seq, d_model]\n"
            "  11. # Optionally return cache and/or attention weights\n"
            "  12. return output (and optional cache/weights)"
        )

    def reset_cache(self):
        """Reset KV cache for starting new generation sequence."""
        raise NotImplementedError(
            "CausalSelfAttention.reset_cache not yet implemented.\n"
            "TODO: Set self.k_cache = None and self.v_cache = None"
        )


class GPTBlock(Module):
    """
    Transformer Block for GPT (Decoder-only).

    Consists of:
    1. Layer Normalization (pre-normalization)
    2. Causal Multi-Head Self-Attention
    3. Residual Connection
    4. Layer Normalization
    5. Position-wise Feed-Forward Network (MLP)
    6. Residual Connection

    Uses Pre-LN architecture (layer norm before sublayers), which is more
    stable than post-LN used in original transformer.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension. Default: 4 * d_model
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation function ('relu', 'gelu', 'gelu_new').
                         Default: 'gelu'
        use_cache (bool): Enable KV caching

    Shape:
        Input:  [batch_size, seq_len, d_model]
        Output: [batch_size, seq_len, d_model]

    Note:
        Pre-LN formula:
        x' = x + MultiHeadAttention(LayerNorm(x))
        x'' = x' + FFN(LayerNorm(x'))

        Post-LN formula (original transformer):
        x' = LayerNorm(x + MultiHeadAttention(x))
        x'' = LayerNorm(x' + FFN(x'))

        Pre-LN is more stable for very deep networks.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_cache: bool = True,
    ):
        """
        Initialize GPT block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function
            use_cache: Enable KV caching
        """
        super().__init__()
        raise NotImplementedError(
            "GPTBlock.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create LayerNorm(d_model) for attention normalization\n"
            "  2. Create CausalSelfAttention(d_model, num_heads, dropout, use_cache)\n"
            "  3. Create LayerNorm(d_model) for FFN normalization\n"
            "  4. Create MLPBlock (feed-forward):\n"
            "     - Linear layer: d_model -> d_ff (default: 4 * d_model)\n"
            "     - GELU/ReLU activation\n"
            "     - Dropout\n"
            "     - Linear layer: d_ff -> d_model\n"
            "  5. Create Dropout(dropout) layer\n"
            "  6. Store d_model"
        )

    def forward(
        self,
        x: np.ndarray,
        use_cache: bool = False,
        return_attention_weights: bool = False,
    ) -> Union[np.ndarray, Tuple]:
        """
        Apply GPT transformer block.

        Args:
            x: [batch_size, seq_len, d_model] - Input tensor
            use_cache: If True, cache K, V for inference
            return_attention_weights: If True, also return attention weights

        Returns:
            output: [batch_size, seq_len, d_model]
            Optional: (cache, attention_weights) if requested

        Pre-LN Process:
            1. x_norm = LayerNorm(x)
            2. attn_out = CausalSelfAttention(x_norm, use_cache)
            3. x = x + attn_out (residual)
            4. x_norm = LayerNorm(x)
            5. ffn_out = FFN(x_norm)
            6. x = x + ffn_out (residual)
            7. return x
        """
        raise NotImplementedError(
            "GPTBlock.forward not yet implemented.\n"
            "TODO:\n"
            "  1. # Self-attention with residual\n"
            "     attn_input = ln_attn(x)\n"
            "     attn_output = self_attention(attn_input, use_cache)\n"
            "     if use_cache:\n"
            "        attn_output, cache = attn_output\n"
            "     x = x + dropout(attn_output)\n"
            "  2. # Feed-forward with residual\n"
            "     ffn_input = ln_ffn(x)\n"
            "     ffn_output = mlp(ffn_input)  # [batch, seq, d_model]\n"
            "     x = x + dropout(ffn_output)\n"
            "  3. if return_attention_weights:\n"
            "        return x, weights\n"
            "     elif use_cache:\n"
            "        return x, cache\n"
            "     else:\n"
            "        return x"
        )


class GPT(Module):
    """
    GPT (Generative Pre-trained Transformer) Language Model.

    Decoder-only transformer for autoregressive text generation.
    Foundation model for OpenAI's GPT-2, GPT-3, and subsequent models.

    Suitable for:
    - Language generation (continuation, completion, creative writing)
    - Few-shot learning (in-context learning with prompts)
    - Text classification (add task-specific head)
    - Question answering
    - Summarization
    - Code generation
    - Any downstream task via prompting or fine-tuning

    Args:
        d_model (int): Model dimension. Default: 768 (GPT-2 small)
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of transformer blocks. Default: 12
        d_ff (int): Feed-forward hidden dimension. Default: 4 * d_model
        vocab_size (int): Vocabulary size. Default: 50257 (GPT-2/3)
        max_seq_len (int): Maximum sequence length. Default: 1024 (GPT-2)
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN ('relu', 'gelu', 'gelu_new').
                         Default: 'gelu'
        use_cache (bool): Enable KV caching for inference. Default: True
        tie_embeddings (bool): Tie token embedding and output layer weights.
                              Default: True (weight tying)
        pad_token_id (int): Index of padding token. Default: 0

    Shape:
        Input:  [batch_size, seq_len]
        Output: [batch_size, seq_len, vocab_size] (logits)

    Example:
        >>> # GPT-2 small configuration
        >>> model = GPT(d_model=768, num_heads=12, num_layers=12)
        >>> input_ids = np.random.randint(0, 50257, (2, 512))
        >>> logits = model(input_ids)
        >>> logits.shape
        Array shape([2, 512, 50257])

    Pre-training Objective:
        Language Modeling: Predict next token given previous tokens
        L = -Σ log P(x_i | x_1, ..., x_{i-1})

    Key Properties:
    - Autoregressive: Each token only sees previous tokens
    - Causal masking: Enforces left-to-right dependency
    - Unidirectional context: Unlike BERT (bidirectional)
    - Generative: Can generate coherent multi-token sequences
    - Transferable: Pre-trained models transfer well to downstream tasks

    References:
        - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
        - "Language Models are Few-Shot Learners" (Brown et al., 2020)
        - OpenAI GPT blog: https://openai.com/blog/gpt-2-1-5b-release/
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
        use_cache: bool = True,
        tie_embeddings: bool = True,
        pad_token_id: int = 0,
    ):
        """
        Initialize GPT language model.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function
            use_cache: Enable KV caching
            tie_embeddings: Share weights between token embedding and output
            pad_token_id: Index of padding token
        """
        super().__init__()
        raise NotImplementedError(
            "GPT.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Store all hyperparameters (d_model, num_heads, num_layers, etc.)\n"
            "  2. Create token embeddings:\n"
            "     Embedding(vocab_size, d_model)\n"
            "  3. Create position embeddings (learnable):\n"
            "     Embedding(max_seq_len, d_model)\n"
            "  4. Create embedding dropout: Dropout(dropout)\n"
            "  5. Create transformer blocks:\n"
            "     ModuleList([GPTBlock(...) for _ in range(num_layers)])\n"
            "  6. Create final layer normalization: LayerNorm(d_model)\n"
            "  7. Create language modeling head:\n"
            "     Linear(d_model, vocab_size)\n"
            "  8. If tie_embeddings: set lm_head.weight = token_embeddings.weight\n"
            "  9. Store max_seq_len, pad_token_id, use_cache\n"
            "  10. Initialize weights (Xavier uniform for linear, normal for embedding)"
        )

    def forward(
        self,
        input_ids: np.ndarray,
        use_cache: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple]]]:
        """
        Forward pass for training or inference.

        Args:
            input_ids: [batch_size, seq_len] - Token indices in vocabulary
            use_cache: If True, return and use KV cache for inference

        Returns:
            logits: [batch_size, seq_len, vocab_size] - Predicted logits for next token

            If use_cache=True:
            (logits, cache): where cache is list of (k, v) from each layer

        Process:
            1. Embed input tokens
            2. Add positional embeddings
            3. Apply embedding dropout
            4. Pass through N transformer blocks
            5. Apply final layer normalization
            6. Project to vocabulary size
            7. Return logits (and optionally cache for generation)

        Shape Changes:
            input_ids:  [batch, seq_len]
            -> embeddings: [batch, seq_len, d_model]
            -> + positions: [batch, seq_len, d_model]
            -> after blocks: [batch, seq_len, d_model]
            -> final_ln: [batch, seq_len, d_model]
            -> logits: [batch, seq_len, vocab_size]
        """
        raise NotImplementedError(
            "GPT.forward not yet implemented.\n"
            "TODO:\n"
            "  1. seq_len = input_ids.size(1)\n"
            "  2. assert seq_len <= max_seq_len, 'Sequence too long'\n"
            "  3. # Token embeddings\n"
            "     x = token_embeddings(input_ids)  # [batch, seq_len, d_model]\n"
            "  4. # Position embeddings (learnable)\n"
            "     positions = np.arange(seq_len, device=input_ids.device)\n"
            "     x = x + pos_embeddings(positions)  # [batch, seq_len, d_model]\n"
            "  5. x = embedding_dropout(x)\n"
            "  6. # Pass through transformer blocks\n"
            "     caches = []\n"
            "     for block in transformer_blocks:\n"
            "        if use_cache:\n"
            "            x, cache = block(x, use_cache=True)\n"
            "            caches.append(cache)\n"
            "        else:\n"
            "            x = block(x, use_cache=False)\n"
            "  7. x = final_ln(x)  # [batch, seq_len, d_model]\n"
            "  8. logits = lm_head(x)  # [batch, seq_len, vocab_size]\n"
            "  9. if use_cache:\n"
            "        return logits, caches\n"
            "     else:\n"
            "        return logits"
        )

    def generate(
        self,
        input_ids: np.ndarray,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        num_beams: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: [batch_size, prompt_len] - Initial prompt/context
            max_length: Maximum number of NEW tokens to generate
            temperature: Sampling temperature (< 1: sharper, > 1: smoother)
            top_k: Keep only top-k most likely next tokens. None = no filtering
            top_p: Keep tokens with cumulative probability >= top_p (nucleus sampling)
            num_beams: Number of beams for beam search. 1 = greedy or sampling
            pad_token_id: Index of padding token
            eos_token_id: Index of end-of-sequence token
            repetition_penalty: Penalty for repeating previous tokens (>1 = penalize)

        Returns:
            generated_ids: [batch_size, prompt_len + max_length] - Full sequence

        Sampling Strategies:
        1. GREEDY DECODING (temperature=1, top_k=None, top_p=1.0):
           - Always pick most likely next token
           - Deterministic, can be repetitive

        2. TEMPERATURE SAMPLING:
           - Adjust logits by temperature before softmax
           - logits_scaled = logits / temperature
           - Higher temp: more uniform (random), Lower: sharper (confident)

        3. TOP-K FILTERING:
           - Only consider top-k most probable tokens
           - Rest get probability 0
           - Removes low-probability "tails"

        4. TOP-P / NUCLEUS SAMPLING:
           - Select smallest set of tokens with cumulative prob >= p
           - More adaptive than top-k (k varies with distribution)
           - Preferred strategy (generates more natural text)

        5. BEAM SEARCH:
           - Keep top num_beams hypotheses at each step
           - More computational cost
           - Often better quality than sampling
           - Returns best hypothesis (or top-k)

        Generation Algorithm (greedy):
            1. while len(sequence) < max_length:
               a. logits = model(sequence)
               b. next_logits = logits[:, -1, :]
               c. next_token = argmax(next_logits)
               d. if next_token == eos_token: break
               e. sequence = append(sequence, next_token)
            2. return sequence

        Generation Algorithm (sampling):
            1. while len(sequence) < max_length:
               a. logits = model(sequence)
               b. next_logits = logits[:, -1, :]
               c. Apply temperature: next_logits /= temperature
               d. Apply top_k and top_p filtering
               e. probs = softmax(next_logits)
               f. next_token = sample(probs)
               g. if next_token == eos_token: break
               h. sequence = append(sequence, next_token)
            2. return sequence
        """
        raise NotImplementedError(
            "GPT.generate not yet implemented.\n"
            "TODO: Implement autoregressive generation:\n"
            "\n"
            "PSEUDOCODE:\n"
            "  1. batch_size = input_ids.size(0)\n"
            "  2. current_seq = input_ids.clone()\n"
            "  3. model.eval()  # Set to evaluation mode\n"
            "  4. with torch.no_grad():  # No gradients needed\n"
            "  5.     for step in range(max_length):\n"
            "  6.         # Forward pass\n"
            "  7.         if use_cache and step > 0:\n"
            "  8.             logits = model(current_seq[:, -1:], cache=cache)\n"
            "  9.         else:\n"
            "  10.            logits = model(current_seq)\n"
            "  11.        \n"
            "  12.        # Get next token logits (last position)\n"
            "  13.        next_logits = logits[:, -1, :]  # [batch, vocab]\n"
            "  14.        \n"
            "  15.        # Apply temperature\n"
            "  16.        if temperature != 1.0:\n"
            "  17.            next_logits = next_logits / temperature\n"
            "  18.        \n"
            "  19.        # Apply top-k filtering\n"
            "  20.        if top_k is not None:\n"
            "  21.            # Implement top_k_filtering function\n"
            "  22.            next_logits = top_k_filtering(next_logits, top_k)\n"
            "  23.        \n"
            "  24.        # Apply top-p (nucleus) sampling\n"
            "  25.        if top_p < 1.0:\n"
            "  26.            # Implement top_p_filtering function\n"
            "  27.            next_logits = top_p_filtering(next_logits, top_p)\n"
            "  28.        \n"
            "  29.        # Get probabilities\n"
            "  30.        probs = F.softmax(next_logits, dim=-1)  # [batch, vocab]\n"
            "  31.        \n"
            "  32.        # Sample or greedy\n"
            "  33.        if temperature > 0 and (top_k or top_p < 1):\n"
            "  34.            # Sample from distribution\n"
            "  35.            next_token = torch.multinomial(probs, num_samples=1)\n"
            "  36.        else:\n"
            "  37.            # Greedy: argmax\n"
            "  38.            next_token = probs.argmax(dim=-1, keepdim=True)\n"
            "  39.        \n"
            "  40.        # Check for end token\n"
            "  41.        if eos_token_id is not None:\n"
            "  42.            if (next_token == eos_token_id).all():\n"
            "  43.                break\n"
            "  44.        \n"
            "  45.        # Append to sequence\n"
            "  46.        current_seq = torch.cat([current_seq, next_token], dim=1)\n"
            "  47.    \n"
            "  48. return current_seq\n"
            "\n"
            "HELPER FUNCTIONS TO IMPLEMENT:\n"
            "  - _top_k_filtering(logits, top_k)\n"
            "  - _top_p_filtering(logits, top_p)\n"
            "  - _apply_repetition_penalty(logits, prev_tokens)\n"
            "  - _apply_temperature(logits, temperature)"
        )

    def compute_loss(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute language modeling loss (next token prediction).

        Args:
            input_ids: [batch_size, seq_len] - Token indices
            attention_mask: [batch_size, seq_len] - True for tokens, False for padding

        Returns:
            loss: scalar tensor (mean loss across batch and sequence)

        Process:
            1. Get model logits
            2. Shift logits and targets for next-token prediction
            3. Compute cross-entropy loss
            4. Mask padding tokens
            5. Return average loss

        Loss Formula:
            L = -1/N * Σ log P(x_i | x_1, ..., x_{i-1})

            Where N = number of non-padding tokens
            P(x_i | history) is obtained from softmax(logits)
        """
        raise NotImplementedError(
            "GPT.compute_loss not yet implemented.\n"
            "TODO:\n"
            "  1. logits = self.forward(input_ids)  # [batch, seq_len, vocab]\n"
            "  2. # Shift: predict x_i from x_{i-1}\n"
            "     shift_logits = logits[:, :-1, :].contiguous()\n"
            "     shift_labels = input_ids[:, 1:].contiguous()\n"
            "  3. # Reshape for cross-entropy\n"
            "     shift_logits = shift_logits.view(-1, vocab_size)\n"
            "     shift_labels = shift_labels.view(-1)\n"
            "  4. # Compute loss\n"
            "     loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')\n"
            "  5. # Apply attention mask if provided\n"
            "     if attention_mask is not None:\n"
            "        mask = attention_mask[:, 1:].contiguous().view(-1)\n"
            "        loss = loss * mask\n"
            "        loss = loss.sum() / mask.sum()\n"
            "     else:\n"
            "        loss = loss.mean()\n"
            "  6. return loss"
        )


# ============================================================================
# CONFIGURATION DICTIONARIES FOR COMMON GPT MODELS
# ============================================================================

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

GPT3_SMALL_CONFIG = {
    "d_model": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "d_ff": 4096,
    "vocab_size": 50257,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "activation": "gelu",
}

GPT3_MEDIUM_CONFIG = {
    "d_model": 2048,
    "num_heads": 24,
    "num_layers": 24,
    "d_ff": 8192,
    "vocab_size": 50257,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "activation": "gelu",
}
