"""
Masked Language Modeling (MLM): Foundation of BERT-style Pretraining

Masked Language Modeling is a self-supervised learning objective where:
1. Randomly mask some tokens in a sequence
2. Train model to predict the masked tokens
3. Model learns contextual representations without any labels

This approach powers BERT, RoBERTa, ALBERT, and many modern NLP models.

Paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
       https://arxiv.org/abs/1810.04805
       Devlin et al. (Google), 2018

Theory:
========
Masked Language Modeling (MLM) is based on the Cloze task from psycholinguistics:
given "The capital of France is _____", predict the missing word.

Key Insight: Bidirectional Context
====================================

Unlike traditional language models (which predict next token from left):
  P(w_i | w_{i-1}, w_{i-2}, ...) [unidirectional/causal]

MLM uses bidirectional context:
  P(w_i | w_1, ..., w_{i-1}, w_{i+1}, ..., w_n) [bidirectional]

This allows the model to use both left and right context for prediction.
In BERT, this is achieved through a Transformer encoder (not decoder).

Architecture:
==============

Standard BERT-like Architecture:

Input: "The cat sat on the mat"
         1   2   3  4   5   6

Tokenization: [CLS] The cat sat on the mat [SEP]

Token Embedding: Each token → embedding vector

Positional Embedding: Each position gets embedding
  (helps model learn about word order)

Segment Embedding: Which sentence token belongs to
  (helps with multi-sentence inputs)

[Embedding] = [Token Embedding] + [Positional Embedding] + [Segment Embedding]

Transformer Encoder:
  [Embedding] → [Self-Attention + FFN]^12 → [Hidden States]

Output: Hidden state for each token position
  Shape: [sequence_length, hidden_dim]

Key Components:
===============

1. **Token Masking**:
   - Randomly select 15% of tokens to mask
   - Replace with special [MASK] token
   - Example:
     Original: "The cat sat on the mat"
     Masked:   "The [MASK] sat on [MASK] mat"

2. **Masking Strategy**:
   For each masked token (15%):
     - 80% of time: Replace with [MASK]
       "The cat sat" → "The [MASK] sat"
     - 10% of time: Replace with random token
       "The cat sat" → "The dog sat"
     - 10% of time: Keep original
       "The cat sat" → "The cat sat"

   Why three strategies?
   - [MASK] only: Model exploits [MASK] signal, not learning representation
   - Random replacement: Forces model to reason about context
   - Keep original: Makes prediction harder (easier to cheat with identity)

   This mixture prevents model from gaming the system.

3. **Prediction Objective**:
   - Only predict tokens that were masked (15% of sequence)
   - Use softmax over vocabulary size (30,000 - 50,000 tokens)
   - Loss: Cross-entropy between predicted and actual token

   Formula:
     Loss = CrossEntropy(logits, masked_token_ids)
     logits[i] = MLM_Head(hidden_state[i])

4. **MLM Head**:
   - Applied to hidden state of each position
   - Architecture: Linear(hidden_dim) → Linear(vocab_size)
   - Actually: Linear → Activation → LayerNorm → Linear
   - Full formula: logits = LayerNorm(GELU(Linear1(hidden_state)))

Mathematical Formulation:
=========================

Given sequence x = [x_1, x_2, ..., x_n] and mask m = [m_1, m_2, ..., m_n]
where m_i ∈ {0, 1} (1 if position i is masked):

1. Replace masked tokens:
   x̃_i = MASK_token if m_i=1, else x_i

2. Encode:
   h = Transformer([embedding(x̃_1), ..., embedding(x̃_n)])

3. Predict:
   logits_i = MLMHead(h_i)
   pred_i = softmax(logits_i)

4. Loss (only on masked positions):
   L = -Σ_{i: m_i=1} log(pred_i[x_i])

5. Total loss (with other objectives):
   L_total = L_MLM + L_NSP + λ*L_regularization

Training Procedure:
====================

Data preprocessing:
  1. Tokenize text into subword tokens (WordPiece, BPE, SentencePiece)
  2. Construct sequences of length 512 (BERT)
  3. Randomly sample two adjacent sentences (for NSP task)
  4. Create [CLS] token_seq1 [SEP] token_seq2 [SEP]

For each training example:
  1. Randomly select 15% of token positions
  2. For each selected position:
     - 80%: Replace with [MASK]
     - 10%: Replace with random token from vocabulary
     - 10%: Keep original token
  3. Forward pass through Transformer
  4. Compute MLM loss on masked positions
  5. Backprop and update weights

Masking Considerations:
=======================

1. **Masking Rate** (typically 15%):
   - Too low (< 10%): Not enough learning signal
   - Too high (> 20%): Too much context corrupted
   - 15% found empirically optimal

2. **Sequence Length**:
   - BERT uses 512 tokens
   - Longer: More computation, capture longer dependencies
   - Shorter: Faster training, may miss long-range context
   - Trade-off between compute and performance

3. **Position Matters**:
   - Random masking treats all positions equally
   - Some positions more informative than others
   - Empirically, random masking works well in practice

4. **Vocabulary Size**:
   - Typical: 30,000-50,000 tokens
   - Larger vocabulary: More expressiveness, more parameters
   - WordPiece tokenization: Common for BERT

Variants and Extensions:
========================

1. **Token Replacement Strategies**:
   BERT (original):
     80% [MASK], 10% random, 10% keep

   RoBERTa:
     100% [MASK] (no random replacement)
     Simpler, slightly better performance

   ELECTRA:
     Replace with plausible tokens (from generator)
     More realistic masking

2. **Masking Scope**:
   Whole Word Masking (WWM):
     If token is part of word, mask entire word
     "New York" (tokens: "New", "York") → mask both
     More linguistically motivated

   Span Masking:
     Mask contiguous spans of tokens
     Example: mask 3-5 consecutive tokens
     Predicts longer context

   N-gram Masking:
     Mask complete n-grams
     More challenging prediction task

3. **Prediction Task Variations**:
   Standard MLM:
     Predict exact token

   Contrastive MLM:
     Predict token from candidates
     Easier than predicting from full vocabulary

   Span Prediction:
     Predict entire masked span
     Predict length of span + tokens

Downstream Evaluation:
======================

1. **Linear Probing**:
   - Freeze BERT, train linear classifier on top
   - Measure accuracy on downstream tasks

2. **Fine-tuning**:
   - Entire BERT updated on downstream task
   - Usually better than linear probing
   - Requires task-specific head

3. **Transfer Performance**:
   - BERT representations transfer well
   - Few-shot learning improved
   - Useful for low-resource languages

Advantages of MLM:
==================
  + Simple self-supervised objective
  + Works with unlabeled text corpus
  + Learns bidirectional representations
  + Effective for transfer learning
  + Language-agnostic (works in any language)
  + Naturally captures semantic meaning

Disadvantages of MLM:
=====================
  - Requires large corpus and compute
  - Limited to text modality (originally)
  - Masking strategy somewhat arbitrary
  - Doesn't capture sequential nature well
  - Special tokens like [MASK] not in natural text

Theoretical Understanding:
==========================

Why does MLM work?

1. **Information Bottleneck**:
   - Masking forces information to flow through representations
   - Hidden states must capture enough info to predict masked tokens
   - Forces model to learn structure

2. **Context Sensitivity**:
   - Different contexts → different predictions for masked token
   - "bank deposit" vs "river bank"
   - Model learns contextual embeddings

3. **Linguistic Structure**:
   - Implicitly learns grammar, syntax, semantics
   - No explicit supervision
   - Emerges from objective

4. **Word Frequency Balancing**:
   - Common words: Harder to predict (more context needed)
   - Rare words: Easier to predict (more distinctive)
   - Creates balanced learning signal

Connection to Autoencoders:
===========================

MLM is similar to denoising autoencoders:
  1. Corrupt input (mask tokens)
  2. Encode to representation
  3. Decode (predict original)
  4. Minimize reconstruction error

Key difference from autoencoder:
  - Only reconstruct masked positions
  - Use full bidirectional context
  - Output space is discrete (tokens)

Modern Developments:
====================

Vision MLM (MAE):
  - Apply MLM to vision (Masked Autoencoders)
  - Mask patches of image
  - Predict masked patches
  - Surprisingly effective for vision

Multimodal MLM:
  - Mask tokens in both text and image
  - Predict across modalities
  - UNITER, ALBEF, etc.

Unified Objectives:
  - Combine MLM with other losses
  - MLM + contrastive + alignment
  - State-of-the-art multimodal models

Practical Implementation Notes:
==============================

1. **Efficient Masking**:
   - Don't mask [CLS] and [SEP] tokens
   - Mask only in middle (for efficiency)
   - Or mask randomly across entire sequence

2. **Label Smoothing**:
   - Optional regularization
   - Reduces overfitting on prediction task
   - Not always helpful for MLM

3. **Vocabulary Knowledge**:
   - Rare words harder to predict
   - May underrepresent in training signal
   - Can use importance sampling to balance

4. **Batch Size**:
   - Larger batches help with gradient noise
   - Typical: 32-256 depending on hardware
   - Gradient accumulation if needed

5. **Learning Rate**:
   - Careful tuning important
   - Typical: 1e-4 to 1e-3
   - Warmup helps with stability
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from python.nn_core import Module, Parameter


class MaskingStrategy:
    """
    Implements BERT-style masking strategy.

    For each selected token position:
    - 80% of time: Replace with [MASK] token
    - 10% of time: Replace with random token
    - 10% of time: Keep original token
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        mask_probability: float = 0.15,
        mask_ratio: float = 0.8,
        random_ratio: float = 0.1,
        keep_ratio: float = 0.1
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            mask_token_id: Token ID for [MASK]
            mask_probability: Probability of masking token (typically 0.15)
            mask_ratio: Of masked tokens, ratio to replace with [MASK] (0.8)
            random_ratio: Ratio to replace with random (0.1)
            keep_ratio: Ratio to keep original (0.1)
        """
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.mask_probability = mask_probability
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.keep_ratio = keep_ratio

        # Verify ratios sum to 1
        total = mask_ratio + random_ratio + keep_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1, got {total}"

    def __call__(
        self,
        input_ids: np.ndarray,
        special_tokens: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply masking to input sequence.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            special_tokens: List of token IDs to NOT mask (e.g., [CLS], [SEP])

        Returns:
            masked_input_ids: Masked version of input
            mlm_labels: Token IDs of masked positions (or -100 for non-masked)

        Implementation:
        1. Create mask: randomly select mask_probability fraction
        2. Exclude special tokens from masking
        3. For each masked position:
           - 80% replace with [MASK]
           - 10% replace with random token
           - 10% keep original
        4. Return masked input and labels
        """
        raise NotImplementedError(
            "Implement masking strategy:\n"
            "1. Create mask: mask = torch.rand(input_ids.shape) < self.mask_probability\n"
            "2. Exclude special tokens: mask[input_ids in special_tokens] = False\n"
            "3. Split masked positions:\n"
            "   - mask_positions: Replace with [MASK]\n"
            "   - random_positions: Replace with random token\n"
            "   - keep_positions: Keep original\n"
            "4. Create output:\n"
            "   - masked_input: Apply replacements\n"
            "   - labels: Original tokens at masked positions, -100 elsewhere\n"
            "5. Return masked_input, labels"
        )


class MLMHead(Module):
    """
    Masked Language Modeling prediction head.

    Predicts the original token at masked positions.

    Architecture:
      hidden_state → Linear → LayerNorm → GELU → Linear → logits

    Args:
        hidden_size: Dimension of hidden states (e.g., 768)
        vocab_size: Size of vocabulary to predict over
    """

    def __init__(self, hidden_size: int = 768, vocab_size: int = 30522):
        """
        Args:
            hidden_size: Hidden dimension of model
            vocab_size: Size of vocabulary
        """
        super().__init__()
        raise NotImplementedError(
            "Implement MLM head:\n"
            "1. Dense layer: hidden_size → hidden_size\n"
            "2. LayerNorm\n"
            "3. GELU activation\n"
            "4. Dense layer: hidden_size → vocab_size\n"
            "Formula: logits = Dense2(GELU(LayerNorm(Dense1(x))))"
        )

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits.

        Args:
            hidden_states: [batch_size, seq_length, hidden_size]

        Returns:
            logits: [batch_size, seq_length, vocab_size]
        """
        raise NotImplementedError()


class MLMModel(Module):
    """
    BERT-like model for masked language modeling.

    Combines:
    - Token embedding
    - Positional embedding
    - Segment embedding
    - Transformer encoder
    - MLM head

    Usage:
        model = MLMModel(vocab_size=30522)
        logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        num_token_type_ids: int = 2
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden dimension
            num_hidden_layers: Number of Transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Intermediate size in FFN
            hidden_dropout_prob: Dropout probability
            attention_probs_dropout_prob: Attention dropout
            max_position_embeddings: Maximum sequence length
            num_token_type_ids: Number of segment types
        """
        super().__init__()
        raise NotImplementedError(
            "Implement MLM model:\n"
            "1. Token embedding layer\n"
            "2. Positional embedding layer (max_position_embeddings)\n"
            "3. Segment embedding layer (num_token_type_ids)\n"
            "4. Dropout layer\n"
            "5. Stack of Transformer blocks\n"
            "6. MLM head\n"
            "Note: Use Module base class for components"
        )

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        token_type_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass to get prediction logits.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Mask for padding [batch_size, seq_length]
            token_type_ids: Segment IDs [batch_size, seq_length]

        Returns:
            logits: [batch_size, seq_length, vocab_size]

        Implementation:
        1. Get embeddings: token + positional + segment
        2. Apply dropout
        3. Pass through Transformer blocks
        4. Apply MLM head
        5. Return logits
        """
        raise NotImplementedError()


class MLMLoss(Module):
    """
    Loss function for masked language modeling.

    Only computes loss for masked positions (-100 label = ignore).

    L = CrossEntropy(logits[masked_pos], labels[masked_pos])
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute MLM loss.

        Args:
            logits: [batch_size, seq_length, vocab_size]
            labels: [batch_size, seq_length], with -100 for non-masked

        Returns:
            Scalar loss value

        Implementation:
        1. Reshape logits to [batch_size*seq_length, vocab_size]
        2. Reshape labels to [batch_size*seq_length]
        3. Apply cross-entropy (ignores -100 labels)
        4. Return loss
        """
        raise NotImplementedError(
            "Implement MLM loss:\n"
            "1. Reshape: logits_flat = logits.view(-1, logits.size(-1))\n"
            "2. Reshape: labels_flat = labels.view(-1)\n"
            "3. Loss: loss = self.criterion(logits_flat, labels_flat)\n"
            "4. Return loss"
        )


class MLMDataset:
    """
    Dataset for masked language modeling.

    Assumes input is list of text documents.

    Example:
        texts = ["The cat sat on the mat.", "Hello world!"]
        dataset = MLMDataset(texts, tokenizer, max_length=512)
        loader = DataLoader(dataset, batch_size=32)
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        masking_strategy: MaskingStrategy,
        max_length: int = 512,
        special_tokens: Optional[List[int]] = None
    ):
        """
        Args:
            texts: List of text documents
            tokenizer: Tokenizer with encode method
            masking_strategy: MaskingStrategy instance
            max_length: Maximum sequence length
            special_tokens: Token IDs to NOT mask
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.masking_strategy = masking_strategy
        self.max_length = max_length
        self.special_tokens = special_tokens or []

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get dataset item.

        Returns:
            {
                'input_ids': masked_input_ids,
                'attention_mask': attention_mask,
                'labels': mlm_labels
            }

        Implementation:
        1. Get text
        2. Tokenize to input_ids
        3. Pad/truncate to max_length
        4. Create attention_mask
        5. Apply masking strategy
        6. Return dict with input_ids, attention_mask, labels
        """
        raise NotImplementedError(
            "Implement __getitem__:\n"
            "1. text = self.texts[idx]\n"
            "2. encoded = self.tokenizer.encode(text)\n"
            "3. Pad or truncate to max_length\n"
            "4. Create attention_mask (1 for real tokens, 0 for padding)\n"
            "5. Apply masking: masked_ids, labels = masking_strategy(input_ids)\n"
            "6. Return {'input_ids': ..., 'attention_mask': ..., 'labels': ...}"
        )


class MLMTrainer:
    """
    Trainer for masked language modeling.

    Handles:
    - Loading and preprocessing data
    - Training loop with gradient updates
    - Loss computation with masking
    - Checkpoint saving/loading

    Usage:
        model = MLMModel()
        trainer = MLMTrainer(model, train_loader, device='cuda')
        for epoch in range(epochs):
            train_loss = trainer.train_epoch()
    """

    def __init__(
        self,
        model: MLMModel,
        optimizer,
        train_loader,
        loss_fn: MLMLoss,
        device: str = 'cpu',
        val_loader = None
    ):
        """
        Args:
            model: MLMModel instance
            optimizer: Optimizer (AdamW recommended)
            train_loader: Training data loader
            loss_fn: MLMLoss instance
            device: 'cpu' (no GPU support in custom Module system)
            val_loader: Optional validation data loader
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss

        Training Loop:
        1. Set model to training mode
        2. For each batch:
           a. Move batch to device
           b. Forward pass: logits = model(input_ids, attention_mask)
           c. Compute loss: loss = loss_fn(logits, labels)
           d. Backward and optimizer step
           e. Track running loss
        3. Return average loss
        """
        raise NotImplementedError(
            "Implement training loop:\n"
            "1. self.model.train()\n"
            "2. For each batch:\n"
            "   a. Move batch to device\n"
            "   b. logits = self.model(batch['input_ids'], batch['attention_mask'])\n"
            "   c. loss = self.loss_fn(logits, batch['labels'])\n"
            "   d. loss.backward()\n"
            "   e. self.optimizer.step()\n"
            "   f. Track loss\n"
            "3. Return average loss"
        )

    def evaluate(self) -> float:
        """Evaluate on validation set."""
        raise NotImplementedError(
            "Implement evaluation loop (similar to train_epoch but no backprop)"
        )

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        raise NotImplementedError()

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        raise NotImplementedError()


# ============================================================================
# Understanding Masked Language Modeling
# ============================================================================

"""
Key Insights about MLM:

1. **Why Masking Works**:
   - Corrupting input creates learning signal
   - Model must reconstruct from context
   - Forces learning of contextual representations

2. **Masking Strategy Details**:
   - 80% [MASK]: Prevents learning to exploit masking
   - 10% random: Forces robust representation learning
   - 10% keep: Makes task harder, encourages deep learning

3. **Bidirectional Context**:
   - Different from autoregressive models (GPT)
   - Can use future and past context
   - Better for understanding tasks

4. **Why Bidirectional is Better**:
   - Unidirectional: "cat _____" could be anything
   - Bidirectional: "The ___ sat on the mat" clearly "cat"
   - Bidirectional representations more informative

5. **Relationship to Other Methods**:
   - Autoencoder: Corrupt → encode → decode
   - MLM: Mask → encode → predict
   - Contrastive: Compare representations
   - MLM is simplest self-supervised for NLP

Recent Variants:

1. RoBERTa:
   - No random/keep strategies (100% mask replacement)
   - Simpler but slightly better
   - Longer training helps

2. ELECTRA:
   - Replace with plausible alternatives
   - Generator produces replacements
   - More realistic masking

3. Span MLM (SpanBERT):
   - Mask entire spans (phrases)
   - Predict span from context
   - Better for NLI and coreference tasks

4. Contrastive MLM:
   - Combine MLM with contrastive learning
   - Multiple objectives combined

Why MLM Became Standard:

1. Simple and effective objective
2. Works with any unlabeled text
3. Learns meaningful representations
4. Easy to adapt to downstream tasks
5. No manual annotations needed
"""


# Alias for common naming
MaskedLanguageModel = MLMModel

