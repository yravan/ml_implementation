"""
Sequence-to-Sequence (Seq2Seq) Architecture with Attention

End-to-end learning for variable-length input-output sequences with
attention mechanism for improved alignment and long-range dependencies.

Theory:
========
Seq2Seq tackles the problem of mapping variable-length sequences to
variable-length output sequences. Classical example: machine translation
where input is English sentence and output is French sentence.

Architecture Components:

1. Encoder (LSTM/GRU):
   - Processes entire input sequence
   - Produces context vector (final hidden state or attention-weighted sum)
   - Compresses input information into fixed-size vector

2. Context Vector:
   - Initial hidden state for decoder
   - Usually final hidden state: c = h_T (encoder)
   - Can be attention-weighted sum of all encoder states

3. Decoder (LSTM/GRU):
   - Autoregressively generates output tokens one at a time
   - Uses previous token as input (teacher forcing during training)
   - Hidden state initialized from encoder context

Basic Seq2Seq Equations:

Encoder forward pass:
    h_t = LSTM(x_t, h_{t-1})  for t = 1...T
    context = h_T  (or attention mechanism)

Decoder forward pass:
    s_t = LSTM(y_{t-1}, s_{t-1})  for t = 1...T'
    p(y_t | y_1...y_{t-1}, x) = softmax(W @ s_t + b)

Where:
- x_1...x_T: source sequence (English)
- y_1...y_T': target sequence (French)
- h_t: encoder hidden state
- s_t: decoder hidden state
- context: encoder final state used to initialize decoder

Fundamental Limitation: Information Bottleneck
==============================================
The context vector must compress ALL information from entire source sequence
into a fixed-size vector. For long sequences (100+ tokens), this causes:
- Information loss: cannot encode all important source details
- Attention problems: decoder cannot easily access specific source tokens
- Gradient flow issues: context vector gradient must flow through entire sequence

Solution: Attention Mechanism (Bahdanau et al., 2014)

Additive Attention (also called Bahdanau attention):

For each decoder step t:
    1. Compute attention scores:
        e_{t,i} = v^T tanh(W_q @ s_{t-1} + W_k @ h_i + b)

    2. Normalize scores (softmax):
        α_{t,i} = exp(e_{t,i}) / Σ_j exp(e_{t,j})

    3. Compute context vector as attention-weighted sum:
        c_t = Σ_i α_{t,i} h_i

    4. Decoder uses both previous state and context:
        s_t = LSTM([y_{t-1}; c_t], s_{t-1})

Where:
- s_{t-1}: decoder hidden state (query)
- h_i: encoder hidden state (key/value)
- e_{t,i}: unnormalized attention score
- α_{t,i}: normalized attention weight (how much to attend to position i)
- c_t: context vector (dynamic, computed for each decoder step)

Multiplicative Attention (Luong et al., 2015):

Simplification that uses dot product instead of neural network:
    e_{t,i} = (s_{t-1}^T h_i) / sqrt(d_k)

This is more efficient and works equally well in practice.
Also called scaled dot-product attention (foundation of Transformers).

Why Attention Works:
====================
1. Alignment: lets decoder "look back" at specific source positions
2. Dynamic context: different context for each decoder step
3. Interpretability: attention weights show model's focus
4. Gradient flow: provides direct paths from decoder to all encoder states

Attention enables:
- Better handling of long sequences (no information bottleneck)
- Word alignment interpretability
- Improved performance on many NMT benchmarks

Mathematical Motivation:
=======================
Without attention, decoder gradient must flow through single context vector:
    dL/dh_i passes through context ← bottleneck

With attention, decoder has direct paths to each encoder state:
    dL/dh_i flows directly from attention gradients ← multiple paths

This enables learning of fine-grained alignment between source and target.

Key Papers:
===========
1. "Sequence to Sequence Learning with Neural Networks"
   (Sutskever et al., 2014)
   - Original seq2seq paper (without attention)
   - Shows RNNs can learn variable-length mappings
   - Trick: reverse input sequence order

2. "Neural Machine Translation by Jointly Learning to Align and Translate"
   (Bahdanau et al., 2014)
   - Introduces attention mechanism (additive)
   - Shows attention learns meaningful alignments
   - Significant improvement over basic seq2seq

3. "Attention Is All You Need" (Vaswani et al., 2017)
   - Introduces Transformer (scaled dot-product attention)
   - Multi-head attention architecture
   - Replaces RNNs entirely, enables parallelization

4. "Effective Approaches to Attention-based Neural Machine Translation"
   (Luong et al., 2015)
   - Simplifies attention (multiplicative vs additive)
   - Various attention mechanisms compared
   - Concatenation of context and hidden state

5. "Learning to Generate Reviews and Discovering Sentiment"
   (Radford et al., 2017)
   - Shows seq2seq with attention for sentiment analysis
   - Demonstrates attention's generality beyond translation

Architecture Details:
====================

1. Encoder:
   - LSTM or GRU processing source sequence
   - Bidirectional for better context (concatenated hidden states)
   - Outputs: all hidden states + final hidden state
   - Final state becomes decoder initialization

2. Attention Module:
   - Takes: decoder hidden state (query), encoder hidden states (keys/values)
   - Computes: attention weights + context vector
   - Output dimension: same as encoder hidden dim

3. Decoder:
   - LSTM or GRU conditioned on context
   - At each step: take previous token + context vector
   - Outputs: logits over vocabulary
   - Autoregressively generates one token at a time

4. Training procedure (Teacher Forcing):
   - Encoder processes entire source sequence
   - Decoder fed ground truth target tokens during training
   - Speeds up training (parallel within sequence)
   - But creates exposure bias: train on gold, test on model predictions

5. Inference procedure:
   - Encoder processes source
   - Decoder generates tokens one at a time
   - Uses own previous predictions as input (no teacher)
   - Can use beam search for better quality

Implementation Strategy:
=======================

Essential components:
- Encoder: LSTM/GRU taking (batch, seq_len, input_dim)
  * Returns: (batch, seq_len, hidden_dim) and (batch, hidden_dim)
  * Can be bidirectional for better context

- Attention layer: computes attention-weighted context
  * Input: query (batch, hidden_dim), keys/values (batch, seq_len, hidden_dim)
  * Computes: scores, weights, context
  * Output: (batch, hidden_dim) context vector

- Decoder: LSTM/GRU with attention-augmented input
  * Input: previous token embedding + context vector
  * Maintains hidden state
  * Outputs: logits over vocabulary

Training:
- Use teacher forcing: feed true target tokens to decoder
- Compute cross-entropy loss on predicted logits
- Backprop through entire encoder and decoder
- Gradient clipping essential

Inference:
- Generate tokens autoregressively
- First token: usually special <START> token
- Stop when generating <END> token
- Beam search for better quality (keeps K hypotheses)

Common Issues:
- Exposure bias: train with teacher forcing, test without
  * Solution: scheduled sampling (gradually use model predictions during training)
- Insufficient attention: attention doesn't align well
  * Solution: coverage mechanism or auxiliary loss on attention
- Degeneracy: model ignores attention early in training
  * Solution: appropriate initialization and regularization

Performance:
============
- Time complexity: O(T_x * H^2 + T_y * T_x * H^2) where T_x/T_y are seq lengths
- Encoder sequential: O(T_x)
- Decoder sequential: O(T_y)
- Attention: O(T_y * T_x) (cross-product of encoder and decoder positions)
- Cannot parallelize across time due to RNN

When sequence lengths are long (1000+), attention becomes bottleneck.
Transformers use efficient attention variants (sparse, linear, etc.)

Variants and Extensions:
=======================

1. Bidirectional encoder:
   - Forward and backward LSTMs, concatenate hidden states
   - Allows each encoder position to see full context
   - Better representations

2. Multi-layer encoder/decoder:
   - Stack multiple LSTM layers
   - Deeper models for more complex patterns
   - Need more training data and computation

3. Attention variations:
   - Multi-head attention (learn multiple alignment patterns)
   - Self-attention (attend to own sequence positions)
   - Coverage mechanism (track which source positions attended to)

4. Copy mechanism:
   - Allow decoder to copy tokens from source directly
   - Useful for tasks with rare words or structured output
   - Augments softmax with pointer network

5. Scheduled sampling (Bengio et al., 2015):
   - During training: gradually transition from teacher forcing to model predictions
   - Reduces exposure bias
   - Improves generalization at inference time

Comparison with Transformers:
=============================
Seq2Seq (RNN-based):
- Pros: Simple, good for streaming, handles variable lengths naturally
- Cons: Sequential (slow), limited context window

Transformer (attention-only):
- Pros: Parallelizable, can handle long sequences, SOTA results
- Cons: Quadratic memory (O(T^2)), cannot process streaming

For modern tasks: Transformers are generally preferred due to SOTA results.
But seq2seq still useful for: streaming, low-latency requirements, limited computation.
"""

from typing import Optional, Tuple, List
import numpy as np


class Encoder:
    """
    Seq2Seq Encoder: processes source sequence into context representation.

    Uses bidirectional LSTM/GRU to process entire source sequence
    and produce context vector(s) for decoder initialization.
    """

    def __init__(self, input_dim: int, hidden_dim: int, vocab_size: int,
                 bidirectional: bool = True, num_layers: int = 1,
                 embedding_dim: Optional[int] = None):
        """
        Initialize Encoder.

        Args:
            input_dim: vocabulary size of source language
            hidden_dim: dimensionality of LSTM hidden states
            vocab_size: source vocabulary size
            bidirectional: whether to use bidirectional LSTM
            num_layers: number of stacked LSTM layers
            embedding_dim: dimensionality of token embeddings (default: hidden_dim)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim or hidden_dim

        # TODO: Initialize embedding layer
        # Maps token IDs (integers) to dense vectors
        self.embedding = None  # (vocab_size, embedding_dim)

        # TODO: Initialize LSTM cells (forward and possibly backward)
        self.lstm_cells = []

    def forward(self, source_ids: np.ndarray, lengths: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode source sequence into context representations.

        Args:
            source_ids: source token IDs, shape (batch_size, src_seq_len)
            lengths: actual lengths of source sequences for masking

        Returns:
            encoder_outputs: all encoder hidden states (batch_size, src_seq_len, hidden_dim)
            context: final context vector for decoder (batch_size, hidden_dim)
                    if bidirectional: concatenates forward and backward final states
        """
        # TODO: Embed source tokens
        # TODO: Process through bidirectional LSTM
        # TODO: If bidirectional:
        #   - Run forward LSTM left-to-right
        #   - Run backward LSTM right-to-left
        #   - Concatenate hidden states at each position
        #   - Return concatenated outputs and concatenated final states
        # TODO: Return encoder outputs and context vector

        pass

    def backward(self, dcontext: np.ndarray, learn_rate: float = 0.01) -> None:
        """
        Backward pass through encoder.

        Args:
            dcontext: gradient w.r.t. context vector
            learn_rate: learning rate for weight updates
        """
        # TODO: Implement BPTT through encoder
        # TODO: Accumulate gradients and update weights

        pass


class AttentionLayer:
    """
    Attention mechanism computing context-dependent alignment weights.

    Implements additive (Bahdanau) attention:
        e_{t,i} = v^T tanh(W_q @ s_{t-1} + W_k @ h_i + b)
        α_{t,i} = softmax(e_{t,i})
        c_t = Σ_i α_{t,i} h_i
    """

    def __init__(self, hidden_dim: int, attention_dim: int):
        """
        Initialize Attention Layer.

        Args:
            hidden_dim: dimensionality of encoder/decoder hidden states
            attention_dim: dimensionality of attention mechanism
                          (internal dimension for computing scores)
        """
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # TODO: Initialize attention weights
        # Query projection: decoder hidden state -> attention_dim
        self.W_q = None  # (hidden_dim, attention_dim)
        self.b_q = None  # (attention_dim,)

        # Key projection: encoder hidden state -> attention_dim
        self.W_k = None  # (hidden_dim, attention_dim)

        # Value vector: attention_dim -> 1 (computes final score)
        self.v = None  # (attention_dim,)

        # Gradients
        self.dW_q = None
        self.dW_k = None
        self.dv = None
        self.db_q = None

    def forward(self, query: np.ndarray, keys: np.ndarray, values: np.ndarray,
                mask: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention weights and context vector.

        Args:
            query: decoder hidden state (batch_size, hidden_dim)
            keys: encoder hidden states (batch_size, src_seq_len, hidden_dim)
            values: encoder hidden states (same as keys for additive attention)
            mask: optional mask for padding positions (batch_size, src_seq_len)
                 set padded positions to -inf before softmax

        Returns:
            context: attention-weighted sum of values (batch_size, hidden_dim)
            weights: normalized attention weights (batch_size, src_seq_len)
        """
        # TODO: Compute query projection
        # q = query @ self.W_q + self.b_q  # (batch, attention_dim)

        # TODO: Compute key projections
        # k = keys @ self.W_k  # (batch, src_seq_len, attention_dim)

        # TODO: Compute attention scores (additive)
        # scores = tanh(q.unsqueeze(1) + k) @ self.v  # (batch, src_seq_len)
        # Or: scores = (q @ self.W_k @ keys.T) / sqrt(attention_dim) for mult. attention

        # TODO: Apply mask if provided (set padding to -inf)
        # scores[mask == 0] = -np.inf

        # TODO: Softmax to get attention weights
        # weights = softmax(scores)  # (batch, src_seq_len)

        # TODO: Compute weighted sum of values (context)
        # context = (weights @ values)  # (batch, hidden_dim)

        # TODO: Store cache for backward pass
        # TODO: Return context and weights

        pass

    def backward(self, dcontext: np.ndarray, cache: dict) -> np.ndarray:
        """
        Backward pass through attention.

        Args:
            dcontext: gradient w.r.t. context vector
            cache: cache from forward pass

        Returns:
            dquery: gradient w.r.t. query (decoder hidden state)
        """
        # TODO: Compute gradients w.r.t. query, keys, values, and parameters
        # TODO: Return gradient w.r.t. query for decoder backprop

        pass


class Decoder:
    """
    Seq2Seq Decoder: generates target sequence with attention to source.

    Uses attention-augmented LSTM to generate target tokens one at a time,
    conditioned on source sequence through attention mechanism.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, embedding_dim: Optional[int] = None):
        """
        Initialize Decoder.

        Args:
            vocab_size: target vocabulary size
            hidden_dim: dimensionality of LSTM hidden states
            embedding_dim: dimensionality of token embeddings (default: hidden_dim)
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim or hidden_dim

        # TODO: Initialize embedding layer
        self.embedding = None  # (vocab_size, embedding_dim)

        # TODO: Initialize LSTM cell (single layer, no bidirectional)
        self.lstm_cell = None

        # TODO: Initialize attention layer
        self.attention = None

        # TODO: Initialize output projection
        # Projects hidden state to vocabulary logits
        self.W_out = None  # (hidden_dim, vocab_size)
        self.b_out = None  # (vocab_size,)

    def forward_step(self, prev_token_id: int, hidden_state: np.ndarray,
                     cell_state: np.ndarray, encoder_outputs: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Single decoder step: generate one target token.

        Args:
            prev_token_id: previous target token ID (int, for single sample)
            hidden_state: decoder LSTM hidden state (hidden_dim,)
            cell_state: decoder LSTM cell state (hidden_dim,)
            encoder_outputs: all encoder hidden states (src_seq_len, hidden_dim)

        Returns:
            logits: prediction logits over vocabulary (vocab_size,)
            hidden_state: new decoder hidden state
            cell_state: new decoder cell state
            attention_weights: attention weights over source sequence
        """
        # TODO: Embed previous token
        # token_emb = self.embedding[prev_token_id]  # (embedding_dim,)

        # TODO: Compute attention context using current hidden state as query
        # context, weights = self.attention.forward(
        #     hidden_state, encoder_outputs, encoder_outputs
        # )

        # TODO: Concatenate token embedding and context
        # decoder_input = np.concatenate([token_emb, context])

        # TODO: LSTM step
        # hidden_state, cell_state = self.lstm_cell.forward(decoder_input, hidden_state, cell_state)

        # TODO: Project to vocabulary
        # logits = hidden_state @ self.W_out + self.b_out

        # TODO: Return logits, states, and attention weights
        pass

    def forward(self, target_ids: np.ndarray, encoder_outputs: np.ndarray,
                initial_hidden: np.ndarray, initial_cell: np.ndarray,
                teacher_forcing: bool = True) \
            -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Decode entire target sequence.

        Args:
            target_ids: target token IDs (batch_size, tgt_seq_len)
            encoder_outputs: encoder outputs (batch_size, src_seq_len, hidden_dim)
            initial_hidden: decoder initial hidden state (batch_size, hidden_dim)
            initial_cell: decoder initial cell state (batch_size, hidden_dim)
            teacher_forcing: whether to use ground truth tokens as input

        Returns:
            logits: predictions for all timesteps (batch_size, tgt_seq_len, vocab_size)
            attention_weights: attention weights for all steps (tgt_seq_len, batch_size, src_seq_len)
        """
        # TODO: Initialize list for outputs
        # TODO: Set decoder initial states from encoder
        # TODO: For each target timestep:
        #   1. Determine input token (true token if teacher_forcing, else model prediction)
        #   2. Compute attention context
        #   3. LSTM step
        #   4. Project to vocabulary
        #   5. Store logits and attention weights
        # TODO: Return stacked logits and attention weights

        pass

    def backward(self, dlogits: np.ndarray, learn_rate: float = 0.01) -> None:
        """
        Backward pass through decoder.

        Args:
            dlogits: gradient w.r.t. all logits
            learn_rate: learning rate for weight updates
        """
        # TODO: Implement BPTT through decoder with attention
        # TODO: Backprop through attention at each step
        # TODO: Accumulate gradients and update weights

        pass


class Seq2Seq:
    """
    Complete Sequence-to-Sequence model with encoder, decoder, and attention.

    End-to-end trainable model for mapping variable-length input sequences
    to variable-length output sequences.
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 hidden_dim: int, embedding_dim: Optional[int] = None,
                 attention_dim: Optional[int] = None,
                 num_encoder_layers: int = 1,
                 bidirectional_encoder: bool = True):
        """
        Initialize Seq2Seq model.

        Args:
            src_vocab_size: source language vocabulary size
            tgt_vocab_size: target language vocabulary size
            hidden_dim: dimensionality of LSTM hidden states
            embedding_dim: dimensionality of embeddings (default: hidden_dim)
            attention_dim: dimensionality of attention (default: hidden_dim)
            num_encoder_layers: number of stacked encoder layers
            bidirectional_encoder: whether encoder is bidirectional
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim or hidden_dim
        self.attention_dim = attention_dim or hidden_dim

        # TODO: Initialize encoder
        self.encoder = Encoder(src_vocab_size, hidden_dim, src_vocab_size,
                              bidirectional=bidirectional_encoder,
                              num_layers=num_encoder_layers,
                              embedding_dim=self.embedding_dim)

        # TODO: Initialize decoder
        self.decoder = Decoder(tgt_vocab_size, hidden_dim,
                              embedding_dim=self.embedding_dim)

    def forward(self, source_ids: np.ndarray, target_ids: np.ndarray,
                teacher_forcing: bool = True) \
            -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through entire seq2seq model.

        Args:
            source_ids: source language token IDs (batch_size, src_seq_len)
            target_ids: target language token IDs (batch_size, tgt_seq_len)
            teacher_forcing: whether decoder uses ground truth tokens

        Returns:
            logits: decoder predictions (batch_size, tgt_seq_len, tgt_vocab_size)
            attention_weights: attention visualization (tgt_seq_len, batch_size, src_seq_len)
        """
        # TODO: Encode source sequence
        # encoder_outputs, context = self.encoder.forward(source_ids)

        # TODO: Decode with attention
        # logits, attention = self.decoder.forward(
        #     target_ids, encoder_outputs, context[0], context[1],
        #     teacher_forcing=teacher_forcing
        # )

        # TODO: Return logits and attention for visualization
        pass

    def backward(self, dlogits: np.ndarray, learn_rate: float = 0.01) -> None:
        """
        Backward pass through entire model.

        Args:
            dlogits: gradient w.r.t. decoder logits
            learn_rate: learning rate
        """
        # TODO: Backprop through decoder (includes attention backprop)
        # TODO: Backprop through encoder
        # TODO: Update all parameters

        pass

    def generate(self, source_ids: np.ndarray, max_length: int = 50,
                 beam_width: int = 1) -> np.ndarray:
        """
        Generate target sequence from source (inference mode).

        Args:
            source_ids: source language token IDs (batch_size, src_seq_len)
            max_length: maximum length of generated sequence
            beam_width: beam width for beam search (1 = greedy)

        Returns:
            generated_ids: generated target token IDs (batch_size, tgt_seq_len)
        """
        # TODO: Encode source
        # TODO: Initialize decoder with START token
        # TODO: For each timestep up to max_length:
        #   1. Predict next token (greedy or beam search)
        #   2. If prediction is END token, stop
        #   3. Otherwise, add to output and continue
        # TODO: Return generated sequence

        pass


if __name__ == "__main__":
    # Test seq2seq model
    src_vocab_size, tgt_vocab_size = 10000, 10000
    hidden_dim, embedding_dim = 256, 128

    # TODO: Create model
    # seq2seq = Seq2Seq(src_vocab_size, tgt_vocab_size, hidden_dim, embedding_dim)

    # TODO: Create sample data
    # batch_size, src_len, tgt_len = 32, 15, 18
    # source_ids = np.random.randint(0, src_vocab_size, (batch_size, src_len))
    # target_ids = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # TODO: Forward pass
    # logits, attention = seq2seq.forward(source_ids, target_ids, teacher_forcing=True)
    # print(f"Logits shape: {logits.shape}")
    # print(f"Attention shape: {len(attention)}")
