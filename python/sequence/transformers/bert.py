"""
BERT (Bidirectional Encoder Representations from Transformers) Implementation

Module: sequence.transformers.bert

COMPLEXITY:
    Time:  O(n^2 * d) for self-attention (quadratic in sequence length)
    Space: O(n * d) for storing activations
    Params: 110M (BERT-base), 340M (BERT-large)

REFERENCES:
    - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
      (Devlin et al., 2018) https://arxiv.org/abs/1810.04805
    - "Attention Is All You Need" (Vaswani et al., 2017)

================================================================================
THEORY: BERT (Bidirectional Encoder Representations from Transformers)
================================================================================

BERT represents a fundamental shift in NLP pre-training:
from unidirectional (left-to-right or right-to-left) to BIDIRECTIONAL context.

KEY INNOVATIONS:

1. BIDIRECTIONAL CONTEXT:
   - Each token attends to ALL surrounding tokens (past and future)
   - Enables deeper language understanding
   - More suited for understanding tasks (classification, NER)

2. MASKED LANGUAGE MODELING (MLM):
   - Pre-training objective: predict randomly masked tokens
   - 15% of tokens randomly selected:
     - 80%: replace with [MASK]
     - 10%: replace with random token
     - 10%: keep original
   - Enables true bidirectional training

3. NEXT SENTENCE PREDICTION (NSP):
   - Predict if sentence B follows sentence A
   - Uses [CLS] token representation

4. SPECIAL TOKENS:
   [CLS]: Classification token (first position)
   [SEP]: Separator between sentences
   [MASK]: Masking token for MLM
   [PAD]: Padding token

5. TOKEN TYPE / SEGMENT EMBEDDINGS:
   - Indicate which sentence a token belongs to (A or B)

6. TRANSFER LEARNING:
   - Pre-train on huge unlabeled corpus
   - Fine-tune on downstream tasks with task-specific heads

================================================================================
MATHEMATICAL FORMULATION
================================================================================

EMBEDDINGS:
    x = LayerNorm(TokenEmbed(ids) + PosEmbed(positions) + SegEmbed(segments))

BIDIRECTIONAL SELF-ATTENTION (no causal mask):
    Attention(Q,K,V) = softmax((Q @ K^T) / sqrt(d_k) + padding_mask) @ V

FINE-TUNING HEADS:
    Classification: Linear(d_model, num_classes) on [CLS] token
    Token classification: Linear(d_model, num_classes) on all tokens
    MLM: Linear(d_model, vocab_size) on masked positions

================================================================================
COMMON BERT VARIANTS
================================================================================

BERT-base:    d_model=768,  num_heads=12, num_layers=12, params=110M
BERT-large:   d_model=1024, num_heads=16, num_layers=24, params=340M
RoBERTa:      Same arch, improved pre-training
DistilBERT:   6 layers, 66M params (distilled)
ALBERT:       Shared weights, factorized embeddings, 12M params

================================================================================
"""

import numpy as np
from typing import Optional, Tuple

from python.foundations import Tensor
from python.nn_core import (
    Module,
    Parameter,
    ModuleList,
    Sequential,
    SinusoidalPositionalEncoding,
)
from python.nn_core.linear import Linear
from python.nn_core.normalization import LayerNorm
from python.nn_core.attention import MultiHeadAttention
from python.nn_core.regularization import Dropout
from python.nn_core.activations import GELU, Tanh
from python.nn_core.positional import LearnedPositionalEmbedding
from python.nn_core.init import normal_
from python.sequence.transformers import TransformerEncoder, EncoderLayer


class BertEmbeddings(Module):
    """
    BERT Embedding Layer combining token, position, and segment embeddings.

    Sums three embedding types (token, position, segment), then applies
    layer normalization and dropout.

    Args:
        vocab_size (int): Vocabulary size. Default: 30522
        d_model (int): Model dimension. Default: 768
        max_seq_len (int): Maximum sequence length. Default: 512
        num_segments (int): Number of segment types. Default: 2
        dropout (float): Dropout probability. Default: 0.1
        eps (float): LayerNorm epsilon. Default: 1e-12

    Shape:
        Input:  token_ids: [batch, seq_len], token_type_ids: [batch, seq_len]
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
        super().__init__()
        self.token_embeddings = Parameter(np.zeros((vocab_size, d_model)))
        normal_(self.token_embeddings, std=0.02)
        self.position_embeddings = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.segment_embeddings = LearnedPositionalEmbedding(num_segments, d_model)

        self.drop = Dropout(dropout)
        self.out_norm = LayerNorm(d_model, eps=eps)


    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Combine all embeddings.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len] (0 or 1). Default: all zeros
            position_ids: [batch, seq_len]. Default: sequential 0..seq_len-1

        Returns:
            [batch, seq_len, d_model]
        """
        B, L = input_ids.shape
        tokens = self.token_embeddings[input_ids.data.flatten().astype(int)].reshape(B, L, -1)
        positions = self.position_embeddings(position_ids.data.flatten().astype(int)).reshape(B, L, -1)
        if token_type_ids is None:
            token_type_ids = Tensor(np.zeros((B, L)))
        segments = self.segment_embeddings(token_type_ids.data.flatten().astype(int)).reshape(B, L, -1)

        input = tokens + positions + segments
        input = self.out_norm(input)
        input = self.drop(input)

        return input



class BertModel(Module):
    """
    Complete BERT Model.

    A bidirectional encoder pre-trained with masked language modeling.
    Combines BERT embeddings, a stack of bidirectional encoder layers
    (reusing MultiHeadAttention from nn_core), and a pooler that extracts
    the [CLS] token representation.

    Args:
        vocab_size (int): Vocabulary size. Default: 30522
        d_model (int): Model dimension. Default: 768
        num_heads (int): Number of attention heads. Default: 12
        num_layers (int): Number of transformer layers. Default: 12
        d_ff (int): Feed-forward dimension. Default: 4 * d_model
        max_seq_len (int): Maximum sequence length. Default: 512
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation in FFN. Default: 'gelu'

    Shape:
        Input:  input_ids: [batch, seq_len]
        Output: hidden_states: [batch, seq_len, d_model]
                pooled_output: [batch, d_model]
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
        super().__init__()
        self.embedding = BertEmbeddings(vocab_size, d_model, max_seq_len, 2, dropout)
        self.encoder = ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.pooler = Sequential(Linear(d_model, d_model), Tanh())
        self.positions = None
        self.max_seq_len = max_seq_len
        self._cached_batch_size = None

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        BERT forward pass.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (1=token, 0=pad)

        Returns:
            hidden_states: [batch, seq_len, d_model]
            pooled_output: [batch, d_model]
        """
        B, L = input_ids.shape
        if self.positions is None or self._cached_batch_size != B:
            self.positions = Tensor(np.tile(np.arange(L), (B, 1)))
            self._cached_batch_size = B
        positions = self.positions
        if positions.shape[1] > L:
            positions = Tensor(positions.data[:, :L])
        input = self.embedding(input_ids, token_type_ids, positions)
        # Convert 1/0 attention mask to boolean (True=attend) for encoder layers
        if attention_mask is not None:
            m = attention_mask.data
            attention_mask = Tensor(m.astype(bool))
        for layer in self.encoder:
            input = layer(input, attention_mask)

        hidden_states = input
        pooled = self.pooler(input[:, 0, :])
        return hidden_states, pooled


class BertForSequenceClassification(Module):
    """
    BERT for text classification.

    Adds a classification head on top of the [CLS] token representation
    from BertModel.

    Args:
        num_classes (int): Number of output classes
        d_model (int): Model dimension. Default: 768
        dropout (float): Dropout in classification head. Default: 0.1
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [batch, seq_len]
        Output: [batch, num_classes]
    """

    def __init__(self, num_classes: int, d_model: int = 768, dropout: float = 0.1, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, **bert_kwargs)
        self.classifier = Sequential(
            Linear(in_features=d_model, out_features=num_classes),
            Dropout(dropout),
            Linear(in_features=num_classes, out_features=num_classes),
        )
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Classify input sequence.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, num_classes]
        """
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(pooled)
        return logits


class BertForTokenClassification(Module):
    """
    BERT for token-level classification (NER, POS tagging).

    Adds a per-token classification head on all token representations.

    Args:
        num_classes (int): Number of output classes per token
        d_model (int): Model dimension. Default: 768
        dropout (float): Dropout probability. Default: 0.1
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [batch, seq_len]
        Output: [batch, seq_len, num_classes]
    """

    def __init__(self, num_classes: int, d_model: int = 768, dropout: float = 0.1, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, **bert_kwargs)
        self.classifier = Sequential(
            Linear(in_features=d_model, out_features=num_classes),
            Dropout(dropout),
            Linear(in_features=num_classes, out_features=num_classes),
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Classify each token.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, num_classes]
        """
        encoded, _ = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(encoded)
        return logits


class BertForMaskedLM(Module):
    """
    BERT for Masked Language Modeling.

    Adds an MLM head that predicts the original tokens at masked positions.
    Used during BERT pre-training.

    Args:
        vocab_size (int): Vocabulary size. Default: 30522
        d_model (int): Model dimension. Default: 768
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [batch, seq_len]
        Output: [batch, seq_len, vocab_size]
    """

    def __init__(self, vocab_size: int = 30522, d_model: int = 768, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, vocab_size=vocab_size, **bert_kwargs)
        self.mlm_head = Sequential(
            Linear(in_features=d_model, out_features=d_model),
            GELU(),
            LayerNorm(d_model),
            Linear(in_features=d_model, out_features=vocab_size),
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict masked tokens.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        encoded, _ = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.mlm_head(encoded)
        return logits


class BertForNextSentencePrediction(Module):
    """
    BERT for Next Sentence Prediction.

    Adds a binary classification head on the pooled [CLS] output.

    Args:
        d_model (int): Model dimension. Default: 768
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  input_ids: [batch, seq_len]
        Output: logits: [batch, 2]
    """

    def __init__(self, d_model: int = 768, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, **bert_kwargs)
        self.nsp_head = Linear(in_features=d_model, out_features=2)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict if sentence B follows sentence A.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, 2]
        """
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.nsp_head(pooled)
        return logits


class BertForPreTraining(Module):
    """
    BERT for Pre-Training (combined MLM + NSP).

    Adds both an MLM head and an NSP head on top of BertModel.
    Returns both sets of logits for joint training.

    Args:
        vocab_size (int): Vocabulary size. Default: 30522
        d_model (int): Model dimension. Default: 768
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  input_ids: [batch, seq_len]
        Output: (mlm_logits: [batch, seq_len, vocab_size],
                 nsp_logits: [batch, 2])
    """

    def __init__(self, vocab_size: int = 30522, d_model: int = 768, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, vocab_size=vocab_size, **bert_kwargs)
        self.mlm_head = Sequential(
            Linear(in_features=d_model, out_features=d_model),
            GELU(),
            LayerNorm(d_model),
            Linear(in_features=d_model, out_features=vocab_size),
        )
        self.nsp_head = Linear(in_features=d_model, out_features=2)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Joint MLM + NSP forward pass.

        Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            mlm_logits: [batch, seq_len, vocab_size]
            nsp_logits: [batch, 2]
        """
        encoded, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        mlm_logits = self.mlm_head(encoded)
        nsp_logits = self.nsp_head(pooled)
        return mlm_logits, nsp_logits


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

BERT_TINY_CONFIG = {
    "vocab_size": 30522,
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 4,
    "d_ff": 1024,
    "max_seq_len": 128,
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
