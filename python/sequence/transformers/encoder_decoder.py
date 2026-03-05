"""
Transformer Encoder-Decoder Implementation (Seq2Seq Architecture)

Module: sequence.transformers.encoder_decoder

COMPLEXITY:
    Time:  O(n^2 * d) for encoder, O(m^2 * d + nm * d) for decoder
    Space: O(n * d + m * d)
    Params: ~14.2M per layer pair (encoder + decoder)

REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017) - Original transformer
    - "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)
    - "Neural Machine Translation by Jointly Learning to Align and Translate"
      (Bahdanau et al., 2015)
    - "Exploring the Limits of Transfer Learning" (Raffel et al., 2019) - T5 model

================================================================================
THEORY: Transformer Encoder-Decoder (Seq2Seq) Architecture
================================================================================

The encoder-decoder architecture is fundamental for tasks that transform one
sequence into another:
- Machine Translation (English -> French)
- Text Summarization (article -> summary)
- Question Answering (question + context -> answer)

KEY DESIGN PRINCIPLES:

1. ENCODER (Bidirectional):
   - Processes entire input sequence at once
   - No causal masking
   - Output: contextualized representation of input

2. DECODER (Autoregressive with Cross-Attention):
   - Each position attends to:
     a) Previous output positions (causal masking)
     b) All encoder output positions (cross-attention)

3. CROSS-ATTENTION (Key Innovation):
   - Decoder queries attend to encoder key-values
   - Not restricted by causality
   - Enables alignment between input and output

4. Three Types of Attention in Seq2Seq:
   a) ENCODER SELF-ATTENTION: bidirectional, understands input
   b) DECODER SELF-ATTENTION: causal, maintains autoregressive constraint
   c) DECODER CROSS-ATTENTION: links output generation to input

================================================================================
MATHEMATICAL FORMULATION
================================================================================

DECODER WITH CROSS-ATTENTION LAYER:
    x' = x + CausalSelfAttention(LN(x))
    x'' = x' + CrossAttention(query=LN(x'), key=encoder_out, value=encoder_out)
    x''' = x'' + FFN(LN(x''))

COMPLETE FORWARD PASS:
    encoder_out = encoder(input_ids)
    decoder_out = decoder(target_ids, encoder_output=encoder_out)
    logits = lm_head(decoder_out)

================================================================================
"""

import numpy as np
from typing import Optional, List

from python.foundations import Tensor
from python.nn_core import Module, Parameter, ModuleList, Sequential
from python.nn_core.linear import Linear
from python.nn_core.normalization import LayerNorm
from python.nn_core.attention import MultiHeadAttention, CausalMask
from python.nn_core.regularization import Dropout
from python.nn_core.activations import GELU, ReLU
from python.nn_core.positional import SinusoidalPositionalEncoding, LearnedPositionalEmbedding
from python.sequence.transformers import TransformerEncoder


class DecoderLayerWithCrossAttention(Module):
    """
    Transformer Decoder Layer with Cross-Attention.

    A decoder block that performs causal self-attention on the target sequence,
    cross-attention to the encoder output, and a position-wise feed-forward
    transformation. All three sublayers use pre-layer-normalization and
    residual connections.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward inner dimension. Default: 4*d_model
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'

    Shape:
        decoder_input:   [batch, tgt_len, d_model]
        encoder_output:  [batch, src_len, d_model]
        output:          [batch, tgt_len, d_model]
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
        self.pre_self_attn_norm = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.pre_cross_attn_norm = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.pre_ff_norm = LayerNorm(d_model)
        self.ffn = Sequential(Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model))

    def forward(
        self,
        decoder_input: Tensor,
        encoder_output: Tensor,
        causal_mask: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply decoder layer with cross-attention.

        Args:
            decoder_input: [batch, tgt_len, d_model]
            encoder_output: [batch, src_len, d_model]
            causal_mask: [tgt_len, tgt_len] causal mask
            encoder_padding_mask: [batch, src_len]

        Returns:
            [batch, tgt_len, d_model]
        """
        # Reshape encoder padding mask [batch, src_len] -> [batch, 1, 1, src_len]
        enc_mask = encoder_padding_mask
        if enc_mask is not None and enc_mask.data.ndim == 2:
            enc_mask = Tensor(enc_mask.data[:, np.newaxis, np.newaxis, :])

        # Causal self-attention + residual
        residual = decoder_input
        x = self.pre_self_attn_norm(decoder_input)
        x = self.self_attn(x, x, x, causal_mask)
        x = x + residual

        # Cross-attention to encoder output + residual
        residual = x
        x = self.pre_cross_attn_norm(x)
        x = self.cross_attn(x, encoder_output, encoder_output, enc_mask)
        x = x + residual

        # FFN + residual
        residual = x
        x = self.pre_ff_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


class TransformerDecoderWithCrossAttention(Module):
    """Full decoder stack with cross-attention for seq2seq models."""

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
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Causal mask: 0 = attend, -inf = don't attend
        self._causal_mask = np.where(
            np.tril(np.ones((max_seq_len, max_seq_len))),
            0.0,
            -np.inf,
        ).astype(np.float32)
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
            DecoderLayerWithCrossAttention(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.lm_head = Sequential(LayerNorm(d_model), Linear(d_model, vocab_size))

    def forward(
        self,
        input_ids: Tensor,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode input tokens through decoder stack.

        Args:
            input_ids: [batch_size, seq_len] - Token indices
            encoder_output: [batch_size, src_len, d_model]
            encoder_padding_mask: [batch_size, src_len] float mask (0/-inf)
            tgt_padding_mask: [batch_size, tgt_len] float mask (0/-inf)

        Returns:
            logits: [batch_size, seq_len, vocab_size]
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

        # Causal mask, optionally combined with target padding
        causal_mask = Tensor(self._causal_mask[:T, :T])

        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, encoder_padding_mask)

        return self.lm_head(x)


class TransformerEncoderDecoder(Module):
    """
    Complete Transformer Encoder-Decoder Model.

    The original Vaswani et al. (2017) architecture for sequence-to-sequence tasks.
    Encodes the source sequence bidirectionally, then decodes the target sequence
    autoregressively with cross-attention to the encoder output.

    Args:
        d_model (int): Model dimension. Default: 512
        num_heads (int): Number of attention heads. Default: 8
        num_encoder_layers (int): Number of encoder layers. Default: 6
        num_decoder_layers (int): Number of decoder layers. Default: 6
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        max_src_len (int): Maximum source sequence length. Default: 512
        max_tgt_len (int): Maximum target sequence length. Default: 512
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'relu'
        share_embeddings (bool): Share encoder/decoder embeddings
        pad_token_id (int): Padding token index. Default: 0

    Shape:
        Encoder input:  [batch, src_len]
        Decoder input:  [batch, tgt_len]
        Output logits:  [batch, tgt_len, tgt_vocab_size]
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
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder = TransformerEncoder(
            d_model,
            num_heads,
            num_encoder_layers,
            d_ff,
            src_vocab_size,
            max_src_len,
            dropout,
            activation,
            use_learnable_pos_embed=False,
        )
        self.decoder = TransformerDecoderWithCrossAttention(
            d_model,
            num_heads,
            num_decoder_layers,
            d_ff,
            tgt_vocab_size,
            max_tgt_len,
            dropout,
            activation,
            use_learnable_pos_embed=False,
        )
        if share_embeddings:
            self.decoder.token_embedding = self.encoder.token_embedding
        self.encoder_norm = LayerNorm(d_model)
        # Cache for generation
        self._cached_encoding = None
        self._cached_src_ids = None

    def _make_padding_mask(self, ids: Tensor) -> Tensor:
        """Create float mask: 0 for real tokens, -inf for pad tokens."""
        mask_data = np.where(ids.data.astype(int) == self.pad_token_id, -np.inf, 0.0)
        return Tensor(mask_data.astype(np.float32))

    @staticmethod
    def _to_float_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
        """Convert a bool mask (True=valid) to float mask (0=valid, -inf=masked)."""
        if mask is None:
            return None
        if mask.data.dtype == bool or mask.data.dtype == np.bool_:
            return Tensor(np.where(mask.data, 0.0, -np.inf).astype(np.float32))
        return mask

    def forward(
        self,
        src_ids: Tensor,
        tgt_ids: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode-decode for training with teacher forcing.

        Args:
            src_ids: [batch, src_len] - Source token indices
            tgt_ids: [batch, tgt_len] - Target token indices
            src_padding_mask: [batch, src_len] float mask (0/-inf)
            tgt_padding_mask: [batch, tgt_len] float mask (0/-inf)

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Build padding masks from pad_token_id, convert bool->float if needed
        auto_src_mask = self._make_padding_mask(src_ids)
        auto_tgt_mask = self._make_padding_mask(tgt_ids)

        src_padding_mask = self._to_float_mask(src_padding_mask)
        tgt_padding_mask = self._to_float_mask(tgt_padding_mask)

        if src_padding_mask is not None:
            enc_mask = Tensor(src_padding_mask.data + auto_src_mask.data)
        else:
            enc_mask = auto_src_mask

        if tgt_padding_mask is not None:
            dec_tgt_mask = Tensor(tgt_padding_mask.data + auto_tgt_mask.data)
        else:
            dec_tgt_mask = auto_tgt_mask

        encoding = self.encoder(src_ids, enc_mask)
        encoding = self.encoder_norm(encoding)

        logits = self.decoder(tgt_ids, encoding, enc_mask, dec_tgt_mask)
        return logits

    def encode(
        self,
        src_ids: Tensor,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode source sequence (cacheable for generation).

        Args:
            src_ids: [batch, src_len]
            src_padding_mask: [batch, src_len]

        Returns:
            encoder_output: [batch, src_len, d_model]
        """
        # Re-encode if cache is empty or source changed
        need_encode = (
            self._cached_encoding is None
            or self._cached_src_ids is None
            or not np.array_equal(src_ids.data, self._cached_src_ids.data)
        )
        if need_encode:
            self._cached_encoding = self.encoder(src_ids, src_padding_mask)
            self._cached_encoding = self.encoder_norm(self._cached_encoding)
            self._cached_src_ids = src_ids
        return self._cached_encoding

    def generate(
        self,
        src_ids: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        start_token_id: int = 1,
        end_token_id: int = 2,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate target sequence autoregressively.

        Args:
            src_ids: [batch, src_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            start_token_id: BOS token ID
            end_token_id: EOS token ID
            src_padding_mask: [batch, src_len]

        Returns:
            generated_ids: [batch, generated_len]
        """
        self.eval()
        B = src_ids.data.shape[0]

        # Build encoder padding mask
        enc_mask = self._make_padding_mask(src_ids)
        if src_padding_mask is not None:
            enc_mask = Tensor(enc_mask.data + src_padding_mask.data)

        # Encode source once
        encoder_output = self.encode(src_ids, enc_mask)

        # Start with BOS token
        generated = np.full((B, 1), start_token_id, dtype=np.int64)

        for _ in range(max_length):
            # Forward decoder on current generated sequence
            logits = self.decoder(Tensor(generated), encoder_output, enc_mask)
            # Get logits for last position
            next_logits = logits.data[:, -1, :].copy()  # [B, vocab_size]

            # Temperature scaling
            if temperature > 1e-8:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                for b in range(B):
                    top_k_idx = np.argpartition(next_logits[b], -top_k)[-top_k:]
                    mask = np.full(next_logits.shape[1], -np.inf)
                    mask[top_k_idx] = next_logits[b, top_k_idx]
                    next_logits[b] = mask

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                for b in range(B):
                    sorted_idx = np.argsort(next_logits[b])[::-1]
                    sorted_logits = next_logits[b, sorted_idx]
                    sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    cumulative_probs = np.cumsum(sorted_probs)
                    cutoff = np.searchsorted(cumulative_probs, top_p) + 1
                    remove_idx = sorted_idx[cutoff:]
                    next_logits[b, remove_idx] = -np.inf

            # Softmax
            max_logits = np.max(next_logits, axis=1, keepdims=True)
            exp_logits = np.exp(next_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Sample or greedy
            if temperature < 0.05:
                next_tokens = np.argmax(probs, axis=1).reshape(-1, 1)
            else:
                next_tokens = np.array([
                    np.random.choice(probs.shape[1], p=probs[b])
                    for b in range(B)
                ]).reshape(-1, 1)

            generated = np.concatenate([generated, next_tokens], axis=1)

            # Stop if all sequences have generated EOS
            if (next_tokens == end_token_id).all():
                break

        # Clear encoding cache
        self._cached_encoding = None
        self._cached_src_ids = None

        return Tensor(generated)




# Configuration for original Transformer (Vaswani et al., 2017)
TRANSFORMER_BASE_CONFIG = {
    "d_model": 512,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "d_ff": 2048,
    "src_vocab_size": 37000,
    "tgt_vocab_size": 37000,
    "max_src_len": 512,
    "max_tgt_len": 512,
    "dropout": 0.1,
    "activation": "relu",
}

# Configuration for T5 (Raffel et al., 2019)
T5_BASE_CONFIG = {
    "d_model": 768,
    "num_heads": 12,
    "num_encoder_layers": 12,
    "num_decoder_layers": 12,
    "d_ff": 3072,
    "src_vocab_size": 32128,
    "tgt_vocab_size": 32128,
    "max_src_len": 512,
    "max_tgt_len": 512,
    "dropout": 0.1,
    "activation": "gelu",
    "share_embeddings": True,
}

T5_LARGE_CONFIG = {
    "d_model": 1024,
    "num_heads": 16,
    "num_encoder_layers": 24,
    "num_decoder_layers": 24,
    "d_ff": 4096,
    "src_vocab_size": 32128,
    "tgt_vocab_size": 32128,
    "max_src_len": 512,
    "max_tgt_len": 512,
    "dropout": 0.1,
    "activation": "gelu",
    "share_embeddings": True,
}
