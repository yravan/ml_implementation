"""
GPT (Generative Pre-trained Transformer) Implementation

Module: sequence.transformers.gpt

COMPLEXITY:
    Time:  O(n^2 * d) for forward pass (n=seq_len, d=d_model)
           O(T * d) with KV cache (linear in generation length)
    Space: O(n * d) for activations, O(T * d) for KV cache
    Params: ~125M (GPT-2 small), ~355M (GPT-2 medium), ~774M (GPT-2 large)

REFERENCES:
    - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
    - "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3
    - "Attention Is All You Need" (Vaswani et al., 2017)

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

2. AUTOREGRESSIVE LANGUAGE MODELING:
   - Pre-training objective: predict next token given previous tokens
   - P(x_1, x_2, ..., x_n) = prod P(x_i | x_1, ..., x_{i-1})
   - Scales well to very large models

3. FEW-SHOT AND ZERO-SHOT LEARNING:
   - Large models learn "in-context learning"
   - Can solve tasks with just a few examples or instruction

4. SCALING LAWS:
   - Performance scales predictably with model size, data, and compute

================================================================================
MATHEMATICAL FORMULATION
================================================================================

TOKEN + POSITIONAL EMBEDDINGS:
    x = TokenEmbed(input_ids) + PosEmbed(positions)

TRANSFORMER BLOCK (Pre-LN):
    x' = x + CausalMultiHeadAttention(LayerNorm(x))
    x'' = x' + FFN(LayerNorm(x'))

    Where FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2

LANGUAGE MODELING HEAD:
    logits = Linear(final_ln(x))  -> [batch, seq_len, vocab_size]

GENERATION:
    - Greedy: next_token = argmax(logits[:, -1, :])
    - Temperature: logits_scaled = logits / T
    - Top-k: keep only k most probable tokens
    - Top-p (nucleus): keep smallest set with cumulative prob >= p

================================================================================
"""

import numpy as np
from typing import Optional, Union, Tuple, List

from python.foundations import Tensor
from python.nn_core import Module, Parameter, ModuleList
from python.nn_core.linear import Linear
from python.nn_core.normalization import LayerNorm
from python.nn_core.attention import MultiHeadAttention, CausalMask
from python.nn_core.regularization import Dropout
from python.nn_core.activations import GELU
from python.nn_core.positional import LearnedPositionalEmbedding
from python.sequence.transformers import DecoderLayer


class GPT(Module):
    """
    GPT (Generative Pre-trained Transformer) Language Model.

    A decoder-only autoregressive language model with next-token prediction
    objective. Uses learned positional embeddings, N decoder layers with
    causal self-attention and GELU feed-forward networks, and an LM head
    projecting to vocabulary logits.

    Args:
        d_model (int): Model dimension. Default: 768
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of transformer blocks. Default: 12
        d_ff (int): Feed-forward hidden dimension. Default: 4 * d_model
        vocab_size (int): Vocabulary size. Default: 50257
        max_seq_len (int): Maximum sequence length. Default: 1024
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'
        tie_embeddings (bool): Share token embedding and LM head weights. Default: True

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
        tie_embeddings: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Causal mask: 0 for attend, -inf for don't attend
        # Lower triangular = attend to current and past positions
        self._causal_mask = np.where(
            np.tril(np.ones((max_seq_len, max_seq_len))),
            0.0,
            -np.inf,
        ).astype(np.float32)
        self.positional_encoding = LearnedPositionalEmbedding(max_seq_len, d_model)
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
        self.tie_embeddings = tie_embeddings
        if tie_embeddings:
            # Remove lm_head.weight from parameters — we'll use token_embedding.T in forward
            del self.lm_head._parameters['weight']

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        """
        Forward pass for training or inference.

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
        pos_ids = np.arange(T)
        pos_embeds = self.positional_encoding(pos_ids)  # [T, d_model]
        x = x + pos_embeds
        x = self.drop(x)

        # Create causal mask for this sequence length
        mask = Tensor(self._causal_mask[:T, :T])

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm + project to vocab
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = x @ self.token_embedding.T + self.lm_head.bias
        else:
            logits = self.lm_head(x)
        return logits


    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: [batch_size, prompt_len] - Initial prompt
            max_length: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: End-of-sequence token ID
            repetition_penalty: Penalty for repeating tokens (>1 = penalize)

        Returns:
            generated_ids: [batch_size, prompt_len + max_length]
        """
        self.eval()
        generated = input_ids.data.astype(int).copy()  # [B, T] numpy array

        for _ in range(max_length):
            # Forward pass on current sequence
            logits = self.forward(Tensor(generated))
            # Get logits for the last position
            next_logits = logits.data[:, -1, :].copy()  # [B, vocab_size]

            # Repetition penalty: penalize tokens that already appeared
            # For each token in the generated sequence, divide its logit by the
            # penalty if positive, or multiply by the penalty if negative.
            # This makes already-seen tokens less likely to be sampled again.
            # Reference: Keskar et al., "CTRL: A Conditional Transformer Language
            # Model for Controllable Generation" (2019)
            if repetition_penalty != 1.0:
                for b in range(next_logits.shape[0]):
                    seen_tokens = set(generated[b].tolist())
                    for token_id in seen_tokens:
                        if next_logits[b, token_id] > 0:
                            next_logits[b, token_id] /= repetition_penalty
                        else:
                            next_logits[b, token_id] *= repetition_penalty

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
            if np.sum(generated[-1] == eos_token_id) > 0:
                break

        return Tensor(generated)

    def compute_loss(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute language modeling loss (next token prediction).

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            loss: scalar
        """
        raise NotImplementedError(
            "Computes the autoregressive language modeling loss by shifting "
            "the logits and labels so that position i predicts token i+1. "
            "Applies cross-entropy loss, optionally masking padding tokens, "
            "and returns the mean loss."
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
