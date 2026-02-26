"""
Transformer Encoder-Decoder Implementation (Seq2Seq Architecture)

Module: sequence.transformers.encoder_decoder

IMPLEMENTATION STATUS:
    - [ ] Encoder stack (bidirectional attention)
    - [ ] Decoder stack (causal attention with cross-attention)
    - [ ] Cross-attention between decoder and encoder
    - [ ] Positional encodings
    - [ ] Training and inference modes
    - [ ] Beam search decoding

COMPLEXITY:
    Time:  O(n^2 * d) for encoder, O(m^2 * d) for decoder (n=input, m=output)
           Total: O(n^2 + nm) per forward pass
    Space: O(n * d + m * d)
    Params: ~14.2M per layer pair (encoder + decoder)

PREREQUISITES:
    - Understanding of transformer encoder
    - Understanding of transformer decoder with causal masking
    - Knowledge of cross-attention mechanisms
    - Familiarity with seq2seq tasks (machine translation, summarization)

REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017) - Original transformer
    - "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)
    - "Neural Machine Translation by Jointly Learning to Align and Translate"
      (Bahdanau et al., 2015) - Attention mechanism introduction
    - "Explore the Limits of Transfer Learning" (Raffel et al., 2019) - T5 model

================================================================================
THEORY: Transformer Encoder-Decoder (Seq2Seq) Architecture
================================================================================

The encoder-decoder architecture is fundamental for tasks that transform one
sequence into another:
- Machine Translation (English -> French)
- Text Summarization (article -> summary)
- Question Answering (question + context -> answer)
- Abstractive QA, paraphrase generation, etc.

KEY DESIGN PRINCIPLES:

1. ENCODER (Bidirectional):
   - Processes entire input sequence at once
   - Each position attends to all input positions
   - No causal masking
   - Output: contextualized representation of input
   - Purpose: Extract meaning from input
   - Example: BERT-style processing

2. DECODER (Autoregressive with Cross-Attention):
   - Processes output sequence one token at a time
   - Each position attends to:
     a) Previous output positions (causal masking)
     b) All encoder output positions (cross-attention)
   - Purpose: Generate output by attending to input
   - Example: GPT-style decoding but with encoder cross-attention

3. CROSS-ATTENTION (Key Innovation):
   - Decoder queries attend to encoder key-values
   - Allows decoder to focus on input while generating output
   - Not restricted by causality (decoder can see all encoder outputs)
   - Enables the model to align input and output
   - Used in attention visualization ("what input caused this output?")

4. Three Types of Attention in Seq2Seq:
   a) ENCODER SELF-ATTENTION:
      Query = Encoder
      Key = Encoder
      Value = Encoder
      Mask: None (bidirectional)
      Purpose: Understand input deeply

   b) DECODER SELF-ATTENTION (CAUSAL):
      Query = Decoder
      Key = Decoder
      Value = Decoder
      Mask: Causal (only attend to positions <= current)
      Purpose: Maintain autoregressive constraint

   c) DECODER CROSS-ATTENTION:
      Query = Decoder
      Key = Encoder
      Value = Encoder
      Mask: None (attend to all encoder positions)
      Purpose: Link output generation to input

5. TRAINING vs INFERENCE:
   Training:
   - Process entire output sequence at once
   - Decoder can see all target tokens (teacher forcing)
   - Parallel computation possible

   Inference (Generation):
   - Generate output tokens one at a time
   - Decoder only sees previously generated tokens
   - Slower but more realistic

================================================================================
MATHEMATICAL FORMULATION
================================================================================

ENCODER FORWARD PASS:
    encoder_output = Encoder(input_ids)    [batch, src_len, d_model]

    Process:
    1. Token embeddings + positional encodings
    2. N encoder layers with bidirectional self-attention (no causal mask)
    3. Final layer normalization
    4. Output: contextualized input representation

DECODER WITH CROSS-ATTENTION LAYER:

    Layer = Causal_Self_Attn + Cross_Attn + FFN

    Causal Self-Attention (on decoder input):
    x' = x + MultiHeadAttention(LN(x), LN(x), LN(x), causal_mask)

    Cross-Attention (decoder queries, encoder keys/values):
    x'' = x' + MultiHeadAttention(
        query=LN(x'),
        key=encoder_output,
        value=encoder_output,
        mask=None  # Attend to all encoder positions
    )

    Feed-Forward:
    x''' = x'' + FFN(LN(x''))

COMPLETE FORWARD PASS:

    # Encode input
    encoder_out = encoder(input_ids)           [batch, src_len, d_model]

    # Decode with teacher forcing (training)
    decoder_out = decoder(
        target_ids,                             [batch, tgt_len, d_model]
        encoder_output=encoder_out,
    )

    # Or decode autoregressively (inference)
    generated_ids = []
    for t in range(max_length):
        logits = decoder(
            generated_ids,                      [batch, t, d_model]
            encoder_output=encoder_out,
        )
        next_token = sample(logits[:, -1, :])  [batch]
        generated_ids.append(next_token)

BEAM SEARCH DECODING:

    Decoding strategy: keep top-k hypotheses at each step

    1. Initialize: k hypotheses with start token
    2. For each generation step:
       a. Expand each hypothesis: each can generate vocab_size continuations
       b. Compute log probability: log(P(next_token | history))
       c. Keep top k hypotheses by cumulative probability
       d. Stop when k hypotheses reach end-of-sequence
    3. Return best hypothesis (or best k hypotheses)

LOSS CALCULATION (Training):

    For sequence-to-sequence:
    - Encoder: no loss (no targets)
    - Decoder: standard cross-entropy on next token prediction

    Loss = CrossEntropyLoss(decoder_logits, target_ids)
    Where:
    - decoder_logits: [batch, tgt_len, vocab_size]
    - target_ids: [batch, tgt_len]
    - Shift targets by 1 position (predict next token)

================================================================================
ARCHITECTURE OVERVIEW: Encoder-Decoder Stack
================================================================================

INPUT SEQUENCE:     [batch, src_len] (e.g., English sentence)
    |
    v
ENCODER:
    |-- Token Embeddings + Positional Embeddings
    |-- Encoder Layer 0 (Bidirectional Self-Attention + FFN)
    |-- Encoder Layer 1
    |-- ...
    |-- Encoder Layer N-1
    |-- Final LayerNorm
    v
ENCODER OUTPUT:     [batch, src_len, d_model] (contextualized input)
    |
    +---> CROSS-ATTENTION in Decoder (keys and values)
    |
TARGET SEQUENCE:    [batch, tgt_len] (e.g., French sentence)
    |
    v
DECODER:
    |-- Token Embeddings + Positional Embeddings
    |-- Decoder Layer 0:
    |   |-- Causal Self-Attention + Residual
    |   |-- Cross-Attention (to encoder) + Residual
    |   |-- FFN + Residual
    |-- Decoder Layer 1
    |-- ...
    |-- Decoder Layer M-1
    |-- Final LayerNorm
    v
DECODER OUTPUT:     [batch, tgt_len, d_model]
    |
    v
LANGUAGE MODELING HEAD: Linear(d_model, vocab_size)
    |
    v
LOGITS:             [batch, tgt_len, vocab_size]

TRAINING:
    Target sequence available -> use teacher forcing
    Compute loss between logits and shifted target tokens

INFERENCE (Generation):
    Iteratively generate next token:
    1. Start with start token
    2. Pass through encoder-decoder
    3. Sample/argmax last token logits
    4. Append to sequence
    5. Repeat until end-of-sequence or max length

================================================================================
FORWARD PASS SHAPE DOCUMENTATION
================================================================================

ENCODER FORWARD PASS:
    Input:
        input_ids:      [batch, src_len]
        padding_mask:   [batch, src_len] (True=token, False=pad)

    Output:
        encoder_output: [batch, src_len, d_model]

DECODER FORWARD PASS (Training with Teacher Forcing):
    Input:
        target_ids:     [batch, tgt_len]
        encoder_output: [batch, src_len, d_model]
        target_padding_mask: [batch, tgt_len]

    Inside Each Decoder Layer:
        Self-Attention (causal):
            Q: [batch, tgt_len, d_model]
            K: [batch, tgt_len, d_model]
            V: [batch, tgt_len, d_model]
            Causal mask: [tgt_len, tgt_len]
            Output: [batch, tgt_len, d_model]

        Cross-Attention:
            Q: [batch, tgt_len, d_model]           (from decoder)
            K: [batch, src_len, d_model]           (from encoder)
            V: [batch, src_len, d_model]           (from encoder)
            No mask (attend to all encoder positions)
            Output: [batch, tgt_len, d_model]

    Output:
        decoder_output: [batch, tgt_len, d_model]
        logits:         [batch, tgt_len, vocab_size]

DECODER FORWARD PASS (Inference, Single Token):
    Input:
        target_ids:     [batch, 1]  (only current token)
        encoder_output: [batch, src_len, d_model]

    Output:
        logits:         [batch, 1, vocab_size]
        Use only logits[:, -1, :] to sample next token

================================================================================
COMMON SEQ2SEQ MODELS
================================================================================

TRANSFORMER (Vaswani et al., 2017):
    - Encoder: 6 layers, 512 d_model, 8 heads, 2048 d_ff
    - Decoder: 6 layers, 512 d_model, 8 heads, 2048 d_ff
    - Params: ~65M
    - Task: Machine Translation (WMT 2014 English-to-German)

T5 (Raffel et al., 2019):
    - Encoder: 12 layers, 768 d_model, 12 heads, 3072 d_ff
    - Decoder: 12 layers, 768 d_model, 12 heads, 3072 d_ff
    - Params: 220M (base)
    - Pre-training: Text-to-text format (C4 dataset)
    - Tasks: Translation, summarization, QA, classification

BERT2BERT:
    - Encoder: Pre-trained BERT
    - Decoder: Pre-trained BERT (with cross-attention added)
    - Leverages existing pre-trained models

MBART (Multilingual BART):
    - Encoder: Multilingual pre-training
    - Decoder: Multilingual pre-training
    - Task: Machine translation across 50+ languages

================================================================================
"""

import math
import numpy as np
from typing import Optional, Tuple, List, Dict

from python.nn_core import Module, Parameter, Sequential, ModuleList
from python.nn_core.layers.linear import Linear
from python.nn_core.normalization.layernorm import LayerNorm
from python.nn_core.attention.multihead import MultiHeadAttention
from python.nn_core.regularization.dropout import Dropout


class DecoderLayerWithCrossAttention(Module):
    """
    Transformer Decoder Layer with Cross-Attention.

    Combines:
    1. Causal Self-Attention (on decoder tokens)
    2. Cross-Attention (decoder attends to encoder output)
    3. Position-wise Feed-Forward Network

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
        """
        Initialize decoder layer with cross-attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        raise NotImplementedError(
            "DecoderLayerWithCrossAttention.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create CausalSelfAttention module (for self-attention)\n"
            "  2. Create MultiHeadAttention module (for cross-attention)\n"
            "  3. Create FeedForwardNetwork module\n"
            "  4. Create three LayerNorm instances (self-attn, cross-attn, ffn)\n"
            "  5. Create dropout layer\n"
            "  6. Store d_model"
        )

    def forward(
        self,
        decoder_input: np.ndarray,
        encoder_output: np.ndarray,
        decoder_padding_mask: Optional[np.ndarray] = None,
        encoder_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply decoder layer with cross-attention.

        Args:
            decoder_input: [batch, tgt_len, d_model] - Target sequence
            encoder_output: [batch, src_len, d_model] - Encoder output
            decoder_padding_mask: [batch, tgt_len] (True=token, False=pad)
            encoder_padding_mask: [batch, src_len] (True=token, False=pad)

        Returns:
            [batch, tgt_len, d_model]

        Process:
            1. Causal self-attention on decoder input
            2. Add residual connection
            3. Cross-attention (decoder queries, encoder keys/values)
            4. Add residual connection
            5. Feed-forward
            6. Add residual connection
        """
        raise NotImplementedError(
            "DecoderLayerWithCrossAttention.forward not yet implemented.\n"
            "TODO:\n"
            "  1. # Self-Attention\n"
            "     self_attn_input = ln_1(decoder_input)\n"
            "     self_attn_out = self_attention(self_attn_input, padding_mask=decoder_padding_mask)\n"
            "     x = decoder_input + dropout(self_attn_out)\n"
            "  2. # Cross-Attention (decoder queries, encoder keys/values)\n"
            "     cross_attn_input = ln_2(x)\n"
            "     cross_attn_out = cross_attention(\n"
            "         query=cross_attn_input,           # decoder\n"
            "         key=encoder_output,               # encoder\n"
            "         value=encoder_output,             # encoder\n"
            "         padding_mask=encoder_padding_mask\n"
            "     )\n"
            "     x = x + dropout(cross_attn_out)\n"
            "  3. # Feed-Forward\n"
            "     ffn_input = ln_3(x)\n"
            "     ffn_out = ffn(ffn_input)\n"
            "     x = x + dropout(ffn_out)\n"
            "  4. return x"
        )


class TransformerEncoderDecoder(Module):
    """
    Complete Transformer Encoder-Decoder Model.

    Suitable for sequence-to-sequence tasks:
    - Machine Translation
    - Text Summarization
    - Question Answering
    - Paraphrase Generation
    - Abstractive Summarization

    Args:
        d_model (int): Model dimension. Default: 512 (original transformer)
        num_heads (int): Number of attention heads. Default: 8
        num_encoder_layers (int): Number of encoder layers. Default: 6
        num_decoder_layers (int): Number of decoder layers. Default: 6
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        max_src_len (int): Maximum source sequence length. Default: 512
        max_tgt_len (int): Maximum target sequence length. Default: 512
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN ('relu' or 'gelu'). Default: 'relu'
        share_embeddings (bool): Share embeddings between encoder/decoder
                                 (only if src_vocab_size == tgt_vocab_size)
        pad_token_id (int): Index of padding token. Default: 0

    Shape:
        Encoder input:  [batch, src_len]
        Decoder input:  [batch, tgt_len]
        Output logits:  [batch, tgt_len, vocab_size]

    Example:
        >>> model = TransformerEncoderDecoder(
        ...     src_vocab_size=10000,
        ...     tgt_vocab_size=10000,
        ... )
        >>> encoder_input = np.random.randint(0, 10000, (2, 32))
        >>> decoder_input = np.random.randint(0, 10000, (2, 28))
        >>> logits = model(encoder_input, decoder_input)
        >>> logits.shape
        Array shape([2, 28, 10000])
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
        """
        Initialize encoder-decoder model.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward hidden dimension
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
            dropout: Dropout probability
            activation: Activation function
            share_embeddings: Share embeddings (requires equal vocab sizes)
            pad_token_id: Index of padding token
        """
        super().__init__()
        raise NotImplementedError(
            "TransformerEncoderDecoder.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Store hyperparameters\n"
            "  2. Create encoder (TransformerEncoder from encoder.py)\n"
            "  3. Create decoder embeddings:\n"
            "     - Token embeddings: Embedding(tgt_vocab_size, d_model)\n"
            "     - Positional embeddings: Embedding(max_tgt_len, d_model)\n"
            "  4. Create ModuleList with num_decoder_layers\n"
            "     DecoderLayerWithCrossAttention instances\n"
            "  5. Create final layer norm\n"
            "  6. Create language modeling head: Linear(d_model, tgt_vocab_size)\n"
            "  7. If share_embeddings: tie weights between encoder/decoder embeddings"
        )

    def forward(
        self,
        src_ids: np.ndarray,
        tgt_ids: np.ndarray,
        src_padding_mask: Optional[np.ndarray] = None,
        tgt_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode-decode for training with teacher forcing.

        Args:
            src_ids: [batch, src_len] - Source language token indices
            tgt_ids: [batch, tgt_len] - Target language token indices
            src_padding_mask: [batch, src_len] (True=token, False=pad)
            tgt_padding_mask: [batch, tgt_len] (True=token, False=pad)

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size] - Next token logits

        Note:
            Uses teacher forcing: provides ground truth targets as input
            during training. Much faster than autoregressive generation.
        """
        raise NotImplementedError(
            "TransformerEncoderDecoder.forward not yet implemented.\n"
            "TODO:\n"
            "  1. # Encode source\n"
            "     encoder_output = encoder(src_ids, padding_mask=src_padding_mask)\n"
            "  2. # Embed target\n"
            "     tgt_len = tgt_ids.size(1)\n"
            "     tgt_embed = token_embeddings(tgt_ids)\n"
            "     positions = np.arange(tgt_len, device=tgt_ids.device)\n"
            "     tgt_embed = tgt_embed + pos_embeddings(positions)\n"
            "  3. # Decode with cross-attention to encoder\n"
            "     x = tgt_embed\n"
            "     for decoder_layer in decoder_layers:\n"
            "         x = decoder_layer(\n"
            "             x,\n"
            "             encoder_output,\n"
            "             decoder_padding_mask=tgt_padding_mask,\n"
            "             encoder_padding_mask=src_padding_mask\n"
            "         )\n"
            "  4. x = final_ln(x)\n"
            "  5. logits = lm_head(x)\n"
            "  6. return logits"
        )

    def encode(
        self,
        src_ids: np.ndarray,
        src_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode source sequence (used during inference).

        Args:
            src_ids: [batch, src_len] - Source language token indices
            src_padding_mask: [batch, src_len] (True=token, False=pad)

        Returns:
            encoder_output: [batch, src_len, d_model]

        Note:
            Separate encode step allows caching encoder output
            when generating multiple target sequences.
        """
        raise NotImplementedError(
            "TransformerEncoderDecoder.encode not yet implemented.\n"
            "TODO: Call self.encoder(src_ids, padding_mask=src_padding_mask)"
        )

    def decode(
        self,
        tgt_ids: np.ndarray,
        encoder_output: np.ndarray,
        tgt_padding_mask: Optional[np.ndarray] = None,
        src_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Decode target sequence with encoder output (used during inference).

        Args:
            tgt_ids: [batch, tgt_len] - Target language token indices (generated so far)
            encoder_output: [batch, src_len, d_model] - Cached encoder output
            tgt_padding_mask: [batch, tgt_len] (True=token, False=pad)
            src_padding_mask: [batch, src_len] (True=token, False=pad)

        Returns:
            logits: [batch, tgt_len, vocab_size] - Next token logits
        """
        raise NotImplementedError(
            "TransformerEncoderDecoder.decode not yet implemented.\n"
            "TODO: Decode portion of forward pass (similar to forward but without encoding)"
        )

    def generate(
        self,
        src_ids: np.ndarray,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        start_token_id: int = 1,
        end_token_id: int = 2,
        src_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate target sequence autoregressively.

        Args:
            src_ids: [batch, src_len] - Source language tokens
            max_length: Maximum length of target to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            start_token_id: Token ID for sequence start (e.g., <BOS>)
            end_token_id: Token ID for sequence end (e.g., <EOS>)
            src_padding_mask: [batch, src_len]

        Returns:
            generated_ids: [batch, generated_len]

        Algorithm:
            1. Encode source once
            2. Initialize target with start token
            3. For each generation step:
               a. Decode current target sequence with encoder output
               b. Extract logits for last token
               c. Apply temperature and sampling
               d. Append next token
               e. Stop if end_token reached or max_length
        """
        raise NotImplementedError(
            "TransformerEncoderDecoder.generate not yet implemented.\n"
            "TODO:\n"
            "  1. encoder_output = self.encode(src_ids, src_padding_mask)\n"
            "  2. batch_size = src_ids.size(0)\n"
            "  3. generated_ids = torch.full(\n"
            "         (batch_size, 1),\n"
            "         start_token_id,\n"
            "         dtype=src_ids.dtype,\n"
            "         device=src_ids.device\n"
            "     )\n"
            "  4. for _ in range(max_length):\n"
            "     a. logits = self.decode(generated_ids, encoder_output)\n"
            "     b. next_logits = logits[:, -1, :]\n"
            "     c. Apply temperature: next_logits /= temperature\n"
            "     d. Apply top_k and top_p filtering\n"
            "     e. next_token = sample(softmax(next_logits))\n"
            "     f. generated_ids = cat([generated_ids, next_token], dim=1)\n"
            "     g. if all sequences reached end_token: break\n"
            "  5. return generated_ids"
        )

    def beam_search(
        self,
        src_ids: np.ndarray,
        beam_width: int = 5,
        max_length: int = 100,
        length_penalty: float = 1.0,
        start_token_id: int = 1,
        end_token_id: int = 2,
        src_padding_mask: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Decode using beam search.

        Args:
            src_ids: [batch, src_len] - Source language tokens
            beam_width: Number of hypotheses to keep at each step
            max_length: Maximum length of target to generate
            length_penalty: Penalty for long sequences (typically 0.6-1.0)
            start_token_id: Token ID for sequence start
            end_token_id: Token ID for sequence end
            src_padding_mask: [batch, src_len]

        Returns:
            List of best generated sequences (one per batch element)

        Algorithm:
            1. Encode source
            2. Initialize beam_width hypotheses with start token
            3. For each generation step:
               a. Expand each hypothesis (vocab_size possibilities)
               b. Compute log probability of each continuation
               c. Keep top beam_width by cumulative score
               d. Track finished sequences (reached end_token)
            4. Return best complete sequences

        Notes:
            - Beam search explores multiple paths simultaneously
            - More accurate than greedy decoding
            - Slower than sampling (requires multiple forward passes)
            - Length penalty prevents bias toward short sequences
        """
        raise NotImplementedError(
            "TransformerEncoderDecoder.beam_search not yet implemented.\n"
            "TODO: Implement beam search decoding algorithm"
        )


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
