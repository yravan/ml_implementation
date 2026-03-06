"""
PyTorch BERT Implementation.

Port of python/sequence/transformers/bert.py to torch.nn.
Reuses EncoderLayer from pytorch/sequence/transformers/encoder.py.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .encoder import EncoderLayer


class BertEmbeddings(nn.Module):
    """
    BERT Embedding Layer combining token, position, and segment embeddings.

    Args:
        vocab_size: Vocabulary size. Default: 30522
        d_model: Model dimension. Default: 768
        max_seq_len: Maximum sequence length. Default: 512
        num_segments: Number of segment types. Default: 2
        dropout: Dropout probability. Default: 0.1
        eps: LayerNorm epsilon. Default: 1e-12

    Shape:
        Input:  token_ids: [B, L], token_type_ids: [B, L]
        Output: [B, L, d_model]
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
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.segment_embeddings = nn.Embedding(num_segments, d_model)
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, L, dtype=torch.long, device=input_ids.device)

        x = self.token_embeddings(input_ids) + \
            self.position_embeddings(position_ids) + \
            self.segment_embeddings(token_type_ids)
        x = self.norm(x)
        x = self.drop(x)
        return x


class BertModel(nn.Module):
    """
    Complete BERT Model.

    Args:
        vocab_size: Vocabulary size. Default: 30522
        d_model: Model dimension. Default: 768
        num_heads: Number of attention heads. Default: 12
        num_layers: Number of transformer layers. Default: 12
        d_ff: Feed-forward dimension. Default: 4 * d_model
        max_seq_len: Maximum sequence length. Default: 512
        dropout: Dropout probability. Default: 0.1
        activation: Activation in FFN. Default: 'gelu'

    Shape:
        Input:  input_ids: [B, L]
        Output: hidden_states: [B, L, d_model], pooled_output: [B, d_model]
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
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert attention_mask (1=keep, 0=pad) to key_padding_mask (True=ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        x = self.embedding(input_ids, token_type_ids)
        for layer in self.encoder:
            x = layer(x, key_padding_mask=key_padding_mask)

        hidden_states = x
        pooled = self.pooler(x[:, 0, :])
        return hidden_states, pooled


class BertForSequenceClassification(nn.Module):
    """
    BERT for text classification.

    Args:
        num_classes: Number of output classes
        d_model: Model dimension. Default: 768
        dropout: Dropout in classification head. Default: 0.1
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [B, L]
        Output: [B, num_classes]
    """

    def __init__(self, num_classes: int, d_model: int = 768, dropout: float = 0.1, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, **bert_kwargs)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        return self.classifier(pooled)


class BertForTokenClassification(nn.Module):
    """
    BERT for token-level classification (NER, POS tagging).

    Args:
        num_classes: Number of output classes per token
        d_model: Model dimension. Default: 768
        dropout: Dropout probability. Default: 0.1
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [B, L]
        Output: [B, L, num_classes]
    """

    def __init__(self, num_classes: int, d_model: int = 768, dropout: float = 0.1, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, **bert_kwargs)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded, _ = self.bert(input_ids, token_type_ids, attention_mask)
        return self.classifier(encoded)


class BertForMaskedLM(nn.Module):
    """
    BERT for Masked Language Modeling.

    Args:
        vocab_size: Vocabulary size. Default: 30522
        d_model: Model dimension. Default: 768
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [B, L]
        Output: [B, L, vocab_size]
    """

    def __init__(self, vocab_size: int = 30522, d_model: int = 768, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, vocab_size=vocab_size, **bert_kwargs)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded, _ = self.bert(input_ids, token_type_ids, attention_mask)
        return self.mlm_head(encoded)


class BertForNextSentencePrediction(nn.Module):
    """
    BERT for Next Sentence Prediction.

    Args:
        d_model: Model dimension. Default: 768
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [B, L]
        Output: [B, 2]
    """

    def __init__(self, d_model: int = 768, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, **bert_kwargs)
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        return self.nsp_head(pooled)


class BertForPreTraining(nn.Module):
    """
    BERT for Pre-Training (combined MLM + NSP).

    Args:
        vocab_size: Vocabulary size. Default: 30522
        d_model: Model dimension. Default: 768
        **bert_kwargs: Arguments for BertModel

    Shape:
        Input:  [B, L]
        Output: (mlm_logits: [B, L, vocab_size], nsp_logits: [B, 2])
    """

    def __init__(self, vocab_size: int = 30522, d_model: int = 768, **bert_kwargs):
        super().__init__()
        self.bert = BertModel(d_model=d_model, vocab_size=vocab_size, **bert_kwargs)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
