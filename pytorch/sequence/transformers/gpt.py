"""
PyTorch GPT (Generative Pre-trained Transformer) Implementation.

Port of python/sequence/transformers/gpt.py to torch.nn.

Key differences from custom numpy version:
  - nn.Embedding instead of Parameter + manual lookup
  - F.scaled_dot_product_attention with is_causal=True (Flash Attention)
  - torch.multinomial + torch.topk for generation
  - Weight tying via shared nn.Embedding.weight
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GPTBlock(nn.Module):
    """
    Single GPT Transformer Block (Pre-LN).

    x' = x + Attn(LN(x))
    x'' = x' + FFN(LN(x'))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        residual = x
        x = self.ln1(x)
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [B, T, num_heads, head_dim]
        q = q.transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.out_proj(x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + residual
        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) Language Model.

    Decoder-only autoregressive model with:
    - Learned token + positional embeddings
    - N transformer blocks with causal self-attention
    - Weight-tied LM head
    - Autoregressive generation with temperature, top-k, top-p

    Args:
        d_model: Model dimension. Default: 768
        num_heads: Number of attention heads. Default: 12
        num_layers: Number of transformer blocks. Default: 12
        d_ff: Feed-forward hidden dimension. Default: 3072
        vocab_size: Vocabulary size. Default: 50257
        max_seq_len: Maximum sequence length. Default: 1024
        dropout: Dropout probability. Default: 0.1

    Shape:
        Input:  [batch_size, seq_len] (token IDs)
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
        **kwargs,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm + LM head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: LM head shares weights with token embedding
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Scale residual projections by 1/sqrt(2*num_layers) per GPT-2 paper
        for block in self.blocks:
            nn.init.normal_(block.out_proj.weight, std=0.02 / math.sqrt(2 * self.num_layers))
            nn.init.normal_(block.ffn[-2].weight, std=0.02 / math.sqrt(2 * self.num_layers))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training or inference.

        Args:
            input_ids: [batch_size, seq_len] — Token indices

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive token generation.

        Args:
            input_ids: [batch_size, prompt_len] — Initial prompt tokens
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens
            top_p: Nucleus sampling threshold
            eos_token_id: End-of-sequence token ID (stops generation)

        Returns:
            generated: [batch_size, prompt_len + generated_len]
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = generated[:, -self.max_seq_len:]

            # Forward pass
            logits = self.forward(idx_cond)
            next_logits = logits[:, -1, :]  # [B, vocab_size]

            # Temperature scaling
            next_logits = next_logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                # Scatter back to original ordering
                next_logits = torch.zeros_like(next_logits).scatter(
                    1, sorted_indices, sorted_logits
                )

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            if temperature < 0.05:
                next_token = probs.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


# ============================================================================
# Configuration Dictionaries
# ============================================================================

GPT2_SMALL_CONFIG = {
    "d_model": 768, "num_heads": 12, "num_layers": 12, "d_ff": 3072,
    "vocab_size": 50257, "max_seq_len": 1024, "dropout": 0.1,
}

GPT2_MEDIUM_CONFIG = {
    "d_model": 1024, "num_heads": 16, "num_layers": 24, "d_ff": 4096,
    "vocab_size": 50257, "max_seq_len": 1024, "dropout": 0.1,
}

GPT2_LARGE_CONFIG = {
    "d_model": 1280, "num_heads": 20, "num_layers": 36, "d_ff": 5120,
    "vocab_size": 50257, "max_seq_len": 1024, "dropout": 0.1,
}

GPT2_XL_CONFIG = {
    "d_model": 1600, "num_heads": 25, "num_layers": 48, "d_ff": 6400,
    "vocab_size": 50257, "max_seq_len": 1024, "dropout": 0.1,
}
