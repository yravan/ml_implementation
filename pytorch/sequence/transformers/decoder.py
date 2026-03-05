"""
PyTorch Transformer Decoder (Autoregressive).

Port of python/sequence/transformers/decoder.py to torch.nn.
"""

import torch
import torch.nn as nn
from typing import Optional


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer (decoder-only, causal).

    Pre-LN causal self-attention + FFN with residual connections.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension. Default: 4 * d_model
        dropout: Dropout probability. Default: 0.1
        activation: Activation in FFN. Default: 'gelu'
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.pre_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), act, nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.pre_attn_norm(x)
        if attn_mask is None:
            T = x.size(1)
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=True)
        x = x + residual

        residual = x
        x = self.post_attn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


class TransformerDecoder(nn.Module):
    """
    Full Autoregressive Decoder with Generation Support.

    Token + positional embeddings, N decoder layers, LM head.

    Args:
        d_model: Model dimension. Default: 768
        num_heads: Number of attention heads. Default: 12
        num_layers: Number of decoder layers. Default: 12
        d_ff: Feed-forward hidden dimension. Default: 4*d_model
        vocab_size: Vocabulary size. Default: 50257
        max_seq_len: Maximum sequence length. Default: 1024
        dropout: Dropout probability. Default: 0.1
        activation: Activation in FFN. Default: 'gelu'
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
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)  # is_causal=True by default when attn_mask is None
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
        top_p: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = generated[:, -self.max_seq_len:]
            logits = self.forward(idx_cond)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


# Config dictionaries
GPT2_SMALL_CONFIG = {
    "d_model": 768, "num_heads": 12, "num_layers": 12, "d_ff": 3072,
    "vocab_size": 50257, "max_seq_len": 1024, "dropout": 0.1, "activation": "gelu",
}

GPT2_MEDIUM_CONFIG = {
    "d_model": 1024, "num_heads": 16, "num_layers": 24, "d_ff": 4096,
    "vocab_size": 50257, "max_seq_len": 1024, "dropout": 0.1, "activation": "gelu",
}

GPT2_LARGE_CONFIG = {
    "d_model": 1280, "num_heads": 20, "num_layers": 36, "d_ff": 5120,
    "vocab_size": 50257, "max_seq_len": 1024, "dropout": 0.1, "activation": "gelu",
}

GPT2_XL_CONFIG = {
    "d_model": 1600, "num_heads": 25, "num_layers": 48, "d_ff": 6400,
    "vocab_size": 50257, "max_seq_len": 1024, "dropout": 0.1, "activation": "gelu",
}
