"""
PyTorch Transformer Encoder (Bidirectional).

Port of python/sequence/transformers/encoder.py to torch.nn.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Pre-LN bidirectional self-attention + FFN with residual connections.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension. Default: 4 * d_model
        dropout: Dropout probability. Default: 0.1
        activation: Activation in FFN ('relu' or 'gelu'). Default: 'gelu'
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            key_padding_mask: [batch, seq_len] bool — True means IGNORE
        """
        residual = x
        x = self.pre_attn_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + residual

        residual = x
        x = self.post_attn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder Stack (Bidirectional).

    Token embeddings + positional embeddings + N encoder layers.

    Args:
        d_model: Model dimension. Default: 768
        num_heads: Number of attention heads. Default: 12
        num_layers: Number of encoder layers. Default: 12
        d_ff: Feed-forward hidden dimension. Default: 4*d_model
        vocab_size: Vocabulary size. Default: 30522
        max_seq_len: Maximum sequence length. Default: 512
        dropout: Dropout probability. Default: 0.1
        activation: Activation in FFN. Default: 'gelu'

    Shape:
        Input:  [batch, seq_len] (token IDs)
        Output: [batch, seq_len, d_model]
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
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = self.final_norm(x)
        return x


# Config dictionaries
BERT_CONFIG = {
    "d_model": 768, "num_heads": 12, "num_layers": 12, "d_ff": 3072,
    "vocab_size": 30522, "max_seq_len": 512, "dropout": 0.1, "activation": "gelu",
}

BERT_LARGE_CONFIG = {
    "d_model": 1024, "num_heads": 16, "num_layers": 24, "d_ff": 4096,
    "vocab_size": 30522, "max_seq_len": 512, "dropout": 0.1, "activation": "gelu",
}
