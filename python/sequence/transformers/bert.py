"""
BERT (Bidirectional Encoder Representations from Transformers) Implementation

Module: sequence.transformers.bert

IMPLEMENTATION STATUS:
    - [ ] Bidirectional transformer encoder
    - [ ] Token, segment, and position embeddings
    - [ ] Masked language modeling (MLM) objective
    - [ ] Next sentence prediction (NSP) objective
    - [ ] Attention pooling for classification
    - [ ] Fine-tuning heads for downstream tasks

COMPLEXITY:
    Time:  O(n^2 * d) for self-attention (quadratic in sequence length)
    Space: O(n * d) for storing activations
    Params: 110M (BERT-base), 340M (BERT-large)

PREREQUISITES:
    - Understanding of bidirectional attention
    - Knowledge of masked language modeling
    - Familiarity with pre-training objectives
    - PyTorch intermediate skills

REFERENCES:
    - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
      (Devlin et al., 2018) https://arxiv.org/abs/1810.04805
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - Google AI BERT blog: https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-model.html

================================================================================
THEORY: BERT (Bidirectional Encoder Representations from Transformers)
================================================================================

BERT represents a fundamental shift in NLP pre-training:
from unidirectional (left-to-right or right-to-left) to BIDIRECTIONAL context.

KEY INNOVATIONS:

1. BIDIRECTIONAL CONTEXT:
   - Each token attends to ALL surrounding tokens (past and future)
   - Enables deeper language understanding
   - Different from GPT (unidirectional, left-to-right only)
   - More suited for understanding tasks (classification, NER)
   - Cannot naturally generate text (can't attend to future)

2. MASKED LANGUAGE MODELING (MLM):
   - Pre-training objective: predict randomly masked tokens
   - 15% of tokens randomly selected
   - Of selected tokens:
     - 80%: replace with [MASK] token
     - 10%: replace with random token
     - 10%: keep original token
   - Model must predict original token from context
   - Enables true bidirectional training

   Example:
   Original: "The cat sat on the mat"
   Masked:   "The [MASK] sat on the [MASK]"
   Task:     Predict "cat" and "mat"

3. NEXT SENTENCE PREDICTION (NSP):
   - Pre-training objective: predict if sentence B follows sentence A
   - 50% of sentence pairs are consecutive (label: IsNext)
   - 50% are random sentences (label: NotNext)
   - Uses [CLS] token representation for prediction
   - Helps model understand document structure

4. SPECIAL TOKENS:
   [CLS]: Classification token at start of sequence
          Representation used for classification tasks
   [SEP]: Separator between sentences
          Used in sentence pair tasks
   [MASK]: Masking token
           Replaces 15% of tokens during MLM pre-training
   [PAD]: Padding token
          Used for padding shorter sequences to fixed length
   [UNK]: Unknown token
          Used for out-of-vocabulary words

5. TOKEN TYPE / SEGMENT EMBEDDINGS:
   - Indicate which sentence a token belongs to (A or B)
   - Useful for sentence pair tasks:
     - Sentence similarity
     - Question answering
     - Paraphrase detection
   - Embedding_A for first sentence, Embedding_B for second

6. ATTENTION POOLING:
   - Use [CLS] token representation as sentence embedding
   - Or use mean pooling of all token representations
   - Different from GPT which generates tokens

7. TRANSFER LEARNING:
   - Pre-train on huge unlabeled corpus (Wikipedia, BookCorpus)
   - Fine-tune on downstream tasks with small labeled data
   - Fine-tuning adds task-specific head:
     - Classification: Linear(d_model, num_classes)
     - NER: Linear(d_model, num_entity_types)  per token
     - QA: Linear(d_model, 2) for span start/end

================================================================================
ARCHITECTURE: BERT Stack
================================================================================

INPUT TEXT: "The cat sat on the mat"
    |
    v
TOKENIZATION: [CLS] "the" "cat" "sat" "on" "the" "mat" [SEP]
    |
    v
TOKEN INDICES: [101, 1996, 2176, 2500, 2006, 1996, 2213, 102]
    |
    v
TOKEN EMBEDDINGS: 8 tokens -> [8, 768] (BERT-base)
    |
    v
+ SEGMENT EMBEDDINGS: [0, 0, 0, 0, 0, 0, 0, 0] (all from sentence A)
    |
    v
+ POSITION EMBEDDINGS: [0, 1, 2, 3, 4, 5, 6, 7] -> [8, 768]
    |
    v
EMBEDDING LAYER NORM + DROPOUT: [8, 768]
    |
    v
TRANSFORMER ENCODER LAYERS (12 or 24):
    |
    +----> Layer 0:
    |       ├── LayerNorm
    |       ├── Multi-Head Self-Attention (bidirectional, no causal mask)
    |       ├── Residual + Dropout
    |       ├── LayerNorm
    |       ├── Feed-Forward
    |       └── Residual + Dropout
    |
    +----> Layer 1-11 (same structure)
    |
    v
OUTPUT LAYER NORM: [8, 768]
    |
    v
CONTEXTUAL TOKEN REPRESENTATIONS: [8, 768]
    |
    +----> Token 0 [CLS]: [768] - Used for classification
    |
    +----> Tokens 1-6: [6, 768] - Individual token representations
                                   Used for NER, POS tagging
    |
    +----> Token 7 [SEP]: [768] - Separator representation

================================================================================
MATHEMATICAL FORMULATION
================================================================================

EMBEDDINGS:
    x_tokens = TokenEmbedding(token_ids)      # [batch, seq_len, d_model]
    x_pos = PositionEmbedding(positions)      # [batch, seq_len, d_model]
    x_seg = SegmentEmbedding(segment_ids)     # [batch, seq_len, d_model]

    x = LayerNorm(x_tokens + x_pos + x_seg + dropout)

BIDIRECTIONAL SELF-ATTENTION (no causal mask):
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V

    # Important: NO causal mask!
    # Each token can attend to all other tokens

    Attention(Q,K,V) = softmax((Q @ K^T) / √d_k + mask) @ V

    Where mask only masks PAD tokens, not future tokens

FINE-TUNING HEADS:

    1. CLASSIFICATION (e.g., sentiment):
       representation = output_hidden_states[:, 0, :]  # [CLS] token
       logits = Linear(d_model, num_classes)(representation)

    2. NER (Named Entity Recognition):
       token_representations = output_hidden_states  # [batch, seq_len, d_model]
       logits = Linear(d_model, num_entity_classes)(token_representations)

    3. QA (Question Answering):
       # Predict span [start, end]
       start_logits = Linear(d_model, 1)(output_hidden_states)
       end_logits = Linear(d_model, 1)(output_hidden_states)

    4. SEMANTIC SIMILARITY:
       # Sentence pair similarity
       rep_a = output_hidden_states[0]  # [CLS] for sentence A
       rep_b = output_hidden_states_b[0]  # [CLS] for sentence B
       similarity = cosine_similarity(rep_a, rep_b)

================================================================================
FORWARD PASS SHAPE DOCUMENTATION
================================================================================

INPUT (Pre-training):
    input_ids:      [batch_size, seq_len]
    token_type_ids: [batch_size, seq_len]
    attention_mask: [batch_size, seq_len] (1=token, 0=pad)

EMBEDDINGS:
    token_emb:      [batch, seq_len, d_model]
    pos_emb:        [batch, seq_len, d_model]
    seg_emb:        [batch, seq_len, d_model]
    x:              [batch, seq_len, d_model]

AFTER EACH ENCODER LAYER:
    hidden_states:  [batch, seq_len, d_model]
    attention_weights: [batch, num_heads, seq_len, seq_len]

OUTPUT:
    hidden_states:  [batch, seq_len, d_model]
    pooled_output:  [batch, d_model] (from [CLS] token)

MLM HEAD:
    mlm_logits:     [batch, seq_len, vocab_size]

NSP HEAD:
    nsp_logits:     [batch, 2] (IsNext vs NotNext)

================================================================================
DIFFERENCES: BERT vs GPT
================================================================================

ASPECT              BERT                          GPT
=================== ============================  ==========================
Attention           Bidirectional                 Causal (unidirectional)
Architecture        Encoder only                  Decoder only
Pre-training        Masked LM + NSP               Next token prediction
Generation          No (can't attend to future)   Yes (autoregressive)
Sequence tasks      Good (classification, NER)    Good (generation, QA)
Training data       Wikipedia + BookCorpus        Raw web text
Context window      512 tokens (BERT-base)        1024 tokens (GPT-2)
Output use          Token representations         Token probabilities
Fine-tuning         Add task head to [CLS]        Add decoder head
Real-time speed     Slower (bidirectional)        Faster (unidirectional)
Few-shot learning   Limited                       Strong (GPT-3)

================================================================================
COMMON BERT VARIANTS
================================================================================

BERT-base:
    d_model: 768, num_heads: 12, num_layers: 12
    Parameters: 110M
    Pre-training: Wikipedia + BookCorpus

BERT-large:
    d_model: 1024, num_heads: 16, num_layers: 24
    Parameters: 340M
    Pre-training: Same corpus

RoBERTa (Robustly Optimized BERT):
    - Same architecture as BERT
    - Improved pre-training: larger batch, more data, longer training
    - Better performance on downstream tasks

DistilBERT:
    - Distilled from BERT-base (40% smaller, 60% faster)
    - 6 layers instead of 12
    - 66M parameters
    - Good trade-off for production use

ELECTRA:
    - Replaced masked language modeling with discriminative task
    - Generator (like BERT) + Discriminator (real vs replaced)
    - Better sample efficiency

ALBERT:
    - Parameter reduction through factorization
    - Shared weights across layers
    - 12M parameters (much smaller)

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


class BertEmbeddings(Module):
    """
    BERT Embedding Layer combining token, position, and segment embeddings.

    Combines three embedding types:
    1. Token embeddings: vocab -> d_model
    2. Position embeddings: sequence position -> d_model
    3. Segment/Token type embeddings: sentence A/B -> d_model

    Then applies layer normalization and dropout.

    Args:
        vocab_size (int): Vocabulary size. Default: 30522 (BERT)
        d_model (int): Model dimension. Default: 768 (BERT-base)
        max_seq_len (int): Maximum sequence length. Default: 512 (BERT)
        num_segments (int): Number of segment types (usually 2: A and B)
        dropout (float): Dropout probability. Default: 0.1
        eps (float): LayerNorm epsilon. Default: 1e-12

    Shape:
        Input:  token_ids: [batch, seq_len]
               token_type_ids: [batch, seq_len]
        Output: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        max_seq_len: int = 512,
        num_segments: int = 2,
        dropout: float = 0.1,
        eps: float = 1e-12,
    ):
        """
        Initialize BERT embeddings.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            num_segments: Number of segment types
            dropout: Dropout probability
            eps: LayerNorm epsilon
        """
        super().__init__()
        raise NotImplementedError(
            "BertEmbeddings.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create Embedding(vocab_size, d_model) for token embeddings\n"
            "  2. Create Embedding(max_seq_len, d_model) for position embeddings\n"
            "  3. Create Embedding(num_segments, d_model) for segment/token type embeddings\n"
            "  4. Create LayerNorm(d_model, eps=eps)\n"
            "  5. Create Dropout(dropout)"
        )

    def forward(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Combine all embeddings.

        Args:
            input_ids: [batch, seq_len] - Token indices
            token_type_ids: [batch, seq_len] - Segment IDs (0 or 1)
                           If None, assume all from first segment
            position_ids: [batch, seq_len] - Position indices
                         If None, use sequential positions (0, 1, ..., seq_len-1)

        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        raise NotImplementedError(
            "BertEmbeddings.forward not yet implemented.\n"
            "TODO:\n"
            "  1. seq_len = input_ids.size(1)\n"
            "  2. # Token embeddings\n"
            "     x = token_embed(input_ids)  # [batch, seq_len, d_model]\n"
            "  3. # Position embeddings\n"
            "     if position_ids is None:\n"
            "        position_ids = np.arange(seq_len, device=input_ids.device)\n"
            "        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)\n"
            "     pos_emb = pos_embed(position_ids)  # [batch, seq_len, d_model]\n"
            "  4. # Segment embeddings\n"
            "     if token_type_ids is None:\n"
            "        token_type_ids = np.zeros_like(input_ids)\n"
            "     seg_emb = seg_embed(token_type_ids)  # [batch, seq_len, d_model]\n"
            "  5. # Combine embeddings\n"
            "     x = x + pos_emb + seg_emb\n"
            "  6. x = layer_norm(x)\n"
            "  7. x = dropout(x)\n"
            "  8. return x"
        )


class BertSelfAttention(Module):
    """
    BERT Self-Attention (Bidirectional Multi-Head Attention).

    No causal masking - each token attends to ALL tokens.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability. Default: 0.1

    Shape:
        Input:  [batch_size, seq_len, d_model]
        Output: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, num_heads: int = 12, dropout: float = 0.1):
        """
        Initialize BERT self-attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        raise NotImplementedError(
            "BertSelfAttention.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Assert d_model % num_heads == 0\n"
            "  2. Store d_model, num_heads\n"
            "  3. Calculate d_k = d_model // num_heads\n"
            "  4. Create Q, K, V projections: Linear(d_model, d_model)\n"
            "  5. Create output projection: Linear(d_model, d_model)\n"
            "  6. Create dropout layer\n"
            "  7. Store scale = 1 / sqrt(d_k)"
        )

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply bidirectional self-attention.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] bool or [batch, 1, seq_len, seq_len]
                           True/1 = attend, False/0 = mask out
            return_attention_weights: If True, return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            weights: [batch, num_heads, seq_len, seq_len] (if requested)

        Note:
            Unlike GPT (causal), BERT uses full attention.
            attention_mask only masks padding tokens, not future tokens.
        """
        raise NotImplementedError(
            "BertSelfAttention.forward not yet implemented.\n"
            "TODO:\n"
            "  1. Project to Q, K, V\n"
            "  2. Reshape for multi-head attention\n"
            "  3. Compute attention scores (Q @ K^T / sqrt(d_k))\n"
            "  4. Apply attention_mask if provided (only mask padding)\n"
            "  5. Apply softmax\n"
            "  6. Apply dropout\n"
            "  7. Multiply by V\n"
            "  8. Concatenate heads and project\n"
            "  9. Return output (and optional weights)"
        )


class BertLayer(Module):
    """
    Single BERT Transformer Layer.

    Consists of:
    1. Multi-Head Bidirectional Self-Attention
    2. Residual connection + Layer Normalization
    3. Position-wise Feed-Forward Network
    4. Residual connection + Layer Normalization

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension. Default: 4 * d_model
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'
        eps (float): LayerNorm epsilon. Default: 1e-12

    Shape:
        Input:  [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        eps: float = 1e-12,
    ):
        """
        Initialize BERT layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function
            eps: LayerNorm epsilon
        """
        super().__init__()
        raise NotImplementedError(
            "BertLayer.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create BertSelfAttention\n"
            "  2. Create LayerNorm for attention output\n"
            "  3. Create feed-forward network\n"
            "  4. Create LayerNorm for FFN output\n"
            "  5. Create dropout layer"
        )

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply BERT layer.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len]

        Returns:
            [batch, seq_len, d_model]
        """
        raise NotImplementedError(
            "BertLayer.forward not yet implemented.\n"
            "TODO: Apply self-attention -> residual -> ffn -> residual"
        )


class BertEncoder(Module):
    """
    BERT Transformer Encoder (N-layer stack).

    Args:
        d_model (int): Model dimension. Default: 768 (BERT-base)
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of transformer layers. Default: 12
        d_ff (int): Feed-forward dimension. Default: 4 * d_model
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'
        eps (float): LayerNorm epsilon. Default: 1e-12

    Shape:
        Input:  [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        eps: float = 1e-12,
    ):
        """
        Initialize BERT encoder.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
            eps: LayerNorm epsilon
        """
        super().__init__()
        raise NotImplementedError(
            "BertEncoder.__init__ not yet implemented.\n"
            "TODO: Create ModuleList with num_layers BertLayer instances"
        )

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply encoder layers.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len]

        Returns:
            [batch, seq_len, d_model] - Final hidden states
        """
        raise NotImplementedError(
            "BertEncoder.forward not yet implemented.\n"
            "TODO: Apply each layer sequentially"
        )


class BertPooler(Module):
    """
    Pooling layer to extract [CLS] token representation.

    Used for classification tasks.

    Args:
        d_model (int): Model dimension

    Input:  [batch, seq_len, d_model]
    Output: [batch, d_model] (from first token [CLS])
    """

    def __init__(self, d_model: int = 768):
        """
        Initialize pooler.

        Args:
            d_model: Model dimension
        """
        super().__init__()
        raise NotImplementedError(
            "BertPooler.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create Linear(d_model, d_model) to project [CLS]\n"
            "  2. Create Tanh activation"
        )

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Extract and pool [CLS] token.

        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            pooled: [batch, d_model]
        """
        raise NotImplementedError(
            "BertPooler.forward not yet implemented.\n"
            "TODO:\n"
            "  1. Extract [CLS] token: cls = hidden_states[:, 0, :]\n"
            "  2. Project: cls = linear(cls)\n"
            "  3. Apply tanh: cls = tanh(cls)\n"
            "  4. return cls"
        )


class BertModel(Module):
    """
    Complete BERT Model.

    Encoder-only transformer for understanding tasks.

    Args:
        vocab_size (int): Vocabulary size. Default: 30522 (BERT)
        d_model (int): Model dimension. Default: 768 (BERT-base)
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of transformer layers. Default: 12
        d_ff (int): Feed-forward dimension. Default: 4 * d_model
        max_seq_len (int): Maximum sequence length. Default: 512
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'

    Shape:
        Input:  input_ids: [batch, seq_len]
       Output: hidden_states: [batch, seq_len, d_model]
               pooled_output: [batch, d_model] (from [CLS])
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize BERT model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        raise NotImplementedError(
            "BertModel.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create BertEmbeddings\n"
            "  2. Create BertEncoder\n"
            "  3. Create BertPooler"
        )

    def forward(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        BERT forward pass.

        Args:
            input_ids: [batch, seq_len] - Token indices
            token_type_ids: [batch, seq_len] - Segment IDs
            attention_mask: [batch, seq_len] - Padding mask (1=token, 0=pad)

        Returns:
            hidden_states: [batch, seq_len, d_model]
            pooled_output: [batch, d_model] (from [CLS] token)
        """
        raise NotImplementedError(
            "BertModel.forward not yet implemented.\n"
            "TODO:\n"
            "  1. embeddings = embedding_layer(input_ids, token_type_ids)\n"
            "  2. encoder_output = encoder(embeddings, attention_mask)\n"
            "  3. pooled = pooler(encoder_output)\n"
            "  4. return encoder_output, pooled"
        )


class BertForSequenceClassification(Module):
    """
    BERT model for text classification.

    Adds a classification head on top of [CLS] token representation.

    Args:
        num_classes (int): Number of output classes
        d_model (int): Model dimension. Default: 768
        dropout (float): Dropout in classification head. Default: 0.1
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [batch, seq_len]
        Output: [batch, num_classes] (logits)

    Example:
        >>> model = BertForSequenceClassification(num_classes=2)
        >>> input_ids = np.random.randint(0, 30522, (2, 128))
        >>> logits = model(input_ids)
        >>> logits.shape
        Array shape([2, 2])
    """

    def __init__(self, num_classes: int, d_model: int = 768, dropout: float = 0.1, **bert_kwargs):
        """
        Initialize classification model.

        Args:
            num_classes: Number of output classes
            d_model: Model dimension
            dropout: Dropout probability
            **bert_kwargs: Arguments for BertModel
        """
        super().__init__()
        raise NotImplementedError(
            "BertForSequenceClassification.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create BertModel(**bert_kwargs)\n"
            "  2. Create classification head:\n"
            "     - Dropout(dropout)\n"
            "     - Linear(d_model, num_classes)"
        )

    def forward(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Classify input sequence.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, num_classes]
        """
        raise NotImplementedError(
            "BertForSequenceClassification.forward not yet implemented.\n"
            "TODO:\n"
            "  1. _, pooled = bert(input_ids, token_type_ids, attention_mask)\n"
            "  2. logits = classification_head(pooled)\n"
            "  3. return logits"
        )


class BertForTokenClassification(Module):
    """
    BERT model for token-level classification (NER, POS tagging).

    Adds classification head on each token representation.

    Args:
        num_classes (int): Number of output classes
        d_model (int): Model dimension. Default: 768
        dropout (float): Dropout probability. Default: 0.1
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [batch, seq_len]
        Output: [batch, seq_len, num_classes] (logits)
    """

    def __init__(self, num_classes: int, d_model: int = 768, dropout: float = 0.1, **bert_kwargs):
        """
        Initialize token classification model.

        Args:
            num_classes: Number of output classes
            d_model: Model dimension
            dropout: Dropout probability
            **bert_kwargs: Arguments for BertModel
        """
        super().__init__()
        raise NotImplementedError(
            "BertForTokenClassification.__init__ not yet implemented.\n"
            "TODO:\n"
            "  1. Create BertModel(**bert_kwargs)\n"
            "  2. Create classification head:\n"
            "     - Dropout(dropout)\n"
            "     - Linear(d_model, num_classes)"
        )

    def forward(
        self,
        input_ids: np.ndarray,
        token_type_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Classify each token.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, num_classes]
        """
        raise NotImplementedError(
            "BertForTokenClassification.forward not yet implemented.\n"
            "TODO:\n"
            "  1. hidden, _ = bert(input_ids, token_type_ids, attention_mask)\n"
            "  2. logits = classification_head(hidden)\n"
            "  3. return logits"
        )


# Configuration dictionaries
BERT_BASE_CONFIG = {
    "vocab_size": 30522,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "d_ff": 3072,
    "max_seq_len": 512,
    "dropout": 0.1,
    "activation": "gelu",
}

BERT_LARGE_CONFIG = {
    "vocab_size": 30522,
    "d_model": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "d_ff": 4096,
    "max_seq_len": 512,
    "dropout": 0.1,
    "activation": "gelu",
}

ROBERTA_BASE_CONFIG = {
    "vocab_size": 50265,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "d_ff": 3072,
    "max_seq_len": 514,
    "dropout": 0.1,
    "activation": "gelu",
}
