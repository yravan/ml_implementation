"""
PyTorch Transformer Encoder-Decoder (Seq2Seq).

Port of python/sequence/transformers/encoder_decoder.py to torch.nn.
"""

import torch
import torch.nn as nn
from typing import Optional

from .encoder import TransformerEncoder


class DecoderLayerWithCrossAttention(nn.Module):
    """
    Transformer Decoder Layer with Cross-Attention.

    Causal self-attention -> cross-attention to encoder -> FFN.
    All with pre-layer-norm and residual connections.
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
        self.pre_self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.pre_cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.pre_ff_norm = nn.LayerNorm(d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), act, nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Causal self-attention
        residual = x
        x = self.pre_self_attn_norm(x)
        if tgt_mask is None:
            T = x.size(1)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, is_causal=True)
        x = x + residual

        # Cross-attention to encoder output
        residual = x
        x = self.pre_cross_attn_norm(x)
        x, _ = self.cross_attn(x, encoder_output, encoder_output,
                                key_padding_mask=memory_key_padding_mask)
        x = x + residual

        # FFN
        residual = x
        x = self.pre_ff_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


class TransformerEncoderDecoder(nn.Module):
    """
    Complete Transformer Encoder-Decoder Model.

    Vaswani et al. (2017) architecture for seq2seq tasks.

    Args:
        d_model: Model dimension. Default: 512
        num_heads: Number of attention heads. Default: 8
        num_encoder_layers: Number of encoder layers. Default: 6
        num_decoder_layers: Number of decoder layers. Default: 6
        d_ff: Feed-forward hidden dimension. Default: 2048
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        max_src_len: Maximum source sequence length. Default: 512
        max_tgt_len: Maximum target sequence length. Default: 512
        dropout: Dropout probability. Default: 0.1
        activation: Activation in FFN. Default: 'relu'
        share_embeddings: Share encoder/decoder embeddings
        pad_token_id: Padding token index. Default: 0
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        max_src_len: int = 512,
        max_tgt_len: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        share_embeddings: bool = False,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Encoder
        self.encoder = TransformerEncoder(
            d_model, num_heads, num_encoder_layers, d_ff,
            src_vocab_size, max_src_len, dropout, activation,
        )

        # Decoder embeddings
        self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_position_embedding = nn.Embedding(max_tgt_len, d_model)
        self.tgt_drop = nn.Dropout(dropout)

        if share_embeddings:
            self.tgt_token_embedding.weight = self.encoder.token_embedding.weight

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayerWithCrossAttention(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tgt_token_embedding.weight, std=0.02)
        nn.init.normal_(self.tgt_position_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode-decode for training with teacher forcing.

        Args:
            src_ids: [batch, src_len]
            tgt_ids: [batch, tgt_len]
            src_key_padding_mask: [batch, src_len] bool (True=IGNORE)

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Auto-generate padding mask from pad_token_id
        if src_key_padding_mask is None:
            src_key_padding_mask = (src_ids == self.pad_token_id)

        # Encode
        encoder_output = self.encoder(src_ids, key_padding_mask=src_key_padding_mask)

        # Decode
        B, T = tgt_ids.shape
        positions = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.tgt_token_embedding(tgt_ids) + self.tgt_position_embedding(positions)
        x = self.tgt_drop(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, memory_key_padding_mask=src_key_padding_mask)

        x = self.decoder_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        max_length: int = 100,
        start_token_id: int = 1,
        end_token_id: int = 2,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy autoregressive generation."""
        self.eval()
        B = src_ids.shape[0]
        if src_key_padding_mask is None:
            src_key_padding_mask = (src_ids == self.pad_token_id)

        encoder_output = self.encoder(src_ids, key_padding_mask=src_key_padding_mask)

        generated = torch.full((B, 1), start_token_id, dtype=torch.long, device=src_ids.device)

        for _ in range(max_length):
            T = generated.shape[1]
            positions = torch.arange(T, device=generated.device).unsqueeze(0)
            x = self.tgt_token_embedding(generated) + self.tgt_position_embedding(positions)
            x = self.tgt_drop(x)
            for layer in self.decoder_layers:
                x = layer(x, encoder_output, memory_key_padding_mask=src_key_padding_mask)
            x = self.decoder_norm(x)
            logits = self.lm_head(x)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_token_id).all():
                break

        return generated


# Configuration dictionaries
TRANSFORMER_BASE_CONFIG = {
    "d_model": 512, "num_heads": 8,
    "num_encoder_layers": 6, "num_decoder_layers": 6,
    "d_ff": 2048, "src_vocab_size": 37000, "tgt_vocab_size": 37000,
    "max_src_len": 512, "max_tgt_len": 512,
    "dropout": 0.1, "activation": "relu",
}

T5_BASE_CONFIG = {
    "d_model": 768, "num_heads": 12,
    "num_encoder_layers": 12, "num_decoder_layers": 12,
    "d_ff": 3072, "src_vocab_size": 32128, "tgt_vocab_size": 32128,
    "max_src_len": 512, "max_tgt_len": 512,
    "dropout": 0.1, "activation": "gelu", "share_embeddings": True,
}

T5_LARGE_CONFIG = {
    "d_model": 1024, "num_heads": 16,
    "num_encoder_layers": 24, "num_decoder_layers": 24,
    "d_ff": 4096, "src_vocab_size": 32128, "tgt_vocab_size": 32128,
    "max_src_len": 512, "max_tgt_len": 512,
    "dropout": 0.1, "activation": "gelu", "share_embeddings": True,
}
