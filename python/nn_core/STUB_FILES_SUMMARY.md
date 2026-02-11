# Comprehensive Stub Files: Attention and Recurrent Modules

Created comprehensive stub files with full theory, mathematics, and implementation hints for deep learning core modules.

## Attention Module (`/nn_core/attention/`)

### 1. scaled_dot_product.py (197 lines)
- **Core Concept**: Fundamental attention mechanism `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Key Components**:
  - ScaledDotProductAttention class with forward() and backward()
  - softmax() helper with numerical stability
- **Math Included**:
  - Attention equation with all components
  - Forward pass algorithm with shapes
  - BPTT backward derivation with softmax gradient
  - Why √d_k scaling prevents gradient issues
- **References**: "Attention Is All You Need" (Vaswani et al., 2017)

### 2. multihead.py (289 lines)
- **Core Concept**: Multi-head attention with h parallel attention heads
- **Architecture**: `MultiHead(Q,K,V) = Concat(head_1,...,head_h) @ W^O`
- **Key Components**:
  - MultiHeadAttention class with learnable projections
  - split_heads() and combine_heads() helper functions
  - Per-head dimension d_k = d_model / num_heads
- **Math Included**:
  - Projection equations for all heads
  - Shape transformations [B,L,d] → [B,h,L,d_k]
  - Gradient flow through concatenation
- **Implementation Hints**: Covers reshape/transpose operations for head separation

### 3. multi_query.py (269 lines)
- **Core Concept**: Shared K/V across heads - h-fold KV cache reduction
- **Why It Matters**: 
  - Standard MHA: O(h × seq_len) KV cache
  - MQA: O(seq_len) KV cache (h-fold improvement!)
- **Key Components**:
  - MultiQueryAttention with per-head Q projections
  - Shared W_k and W_v (single projection for all heads)
  - get_kv_cache_size() comparison method
- **Trade-offs**: Modest quality reduction (~5-20%) for substantial inference speedup
- **Applications**: GPT-3, large LLMs, production serving

### 4. grouped_query.py (350 lines)
- **Core Concept**: Compromise between MHA (h groups) and MQA (1 group)
- **Tunable**: g KV groups where 1 ≤ g ≤ h
- **Key Components**:
  - GroupedQueryAttention with configurable num_kv_groups
  - convert_from_mha() for converting trained MHA to GQA
  - get_kv_cache_stats() for memory analysis
- **Practical Configs**:
  - g = h: Standard MHA, no reduction
  - g = h/2, h/4: Moderate reduction with minimal quality loss
  - g = 1: Full MQA, maximum reduction
- **Real-world**: Used in LLaMA 2, recent production LLMs

### 5. causal_mask.py (375 lines)
- **Core Concept**: Lower triangular masks for autoregressive models
- **Key Methods**:
  - create_causal_mask(): Basic causal masking [seq_len, seq_len]
  - create_padding_mask(): Variable-length sequence handling
  - create_causal_padding_mask(): Combined mask
  - create_sliding_window_mask(): Recent tokens only (efficiency)
  - create_local_attention_mask(): Sparse attention patterns
  - apply_causal_mask(): Apply mask before softmax
- **Math Included**:
  - Mask value semantics (0/-inf for float, True/False for bool)
  - Broadcasting rules for efficient masking
  - Local attention patterns for long sequences
- **Applications**: GPT (autoregressive), efficient transformers (Longformer, BigBird)

### 6. cross_attention.py (373 lines)
- **Core Concept**: Decoder attends to encoder outputs (encoder-decoder models)
- **Key Components**:
  - CrossAttention extends MultiHeadAttention
  - Different query source (decoder) vs key/value source (encoder)
  - CachedCrossAttention with K/V caching for inference
  - MultimodalCrossAttention for vision-language models
- **Shapes**:
  - Query: [B, tgt_len, d_model] (decoder)
  - Key/Value: [B, src_len, d_model] (encoder)
  - Attention matrix: [B, num_heads, tgt_len, src_len]
- **Inference**: K/V computed once, reused across decoder steps (tgt_len × speedup)
- **Applications**: Machine translation, summarization, image captioning, QA

## Recurrent Module (`/nn_core/recurrent/`)

### 1. rnn_cell.py (382 lines)
- **Core Concept**: Vanilla RNN `h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)`
- **Key Components**:
  - RNNCell with forward() and backward() (BPTT)
  - BidirectionalRNNCell for both directions
- **Math Included**:
  - Forward pass with pre-activation and tanh
  - BPTT gradient computation through hidden states
  - Critical issue: Vanishing/exploding gradients
  - Why tanh bounds output to [-1, 1]
- **Problem**: Gradients vanish/explode over long sequences
  - Gradient: ∂L/∂h_0 = ∏_t(W_hh^T * tanh'(...))
  - Repeated multiplication shrinks gradient exponentially
- **Solution**: LSTM/GRU with gating mechanisms

### 2. lstm_cell.py (508 lines) - VERY DETAILED
- **Core Concept**: LSTM solves vanishing gradients with gating + cell state
- **Gates**: Forget, Input, Output (3 gates) + Cell candidate
- **Equations** (COMPREHENSIVE):
  ```
  i_t = σ(W_ii @ x_t + W_hi @ h_{t-1} + b_i)     [Input gate]
  f_t = σ(W_if @ x_t + W_hf @ h_{t-1} + b_f)     [Forget gate]
  c̃_t = tanh(W_ic @ x_t + W_hc @ h_{t-1} + b_c) [Cell candidate]
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t              [Cell state UPDATE (CRITICAL)]
  o_t = σ(W_io @ x_t + W_ho @ h_{t-1} + b_o)     [Output gate]
  h_t = o_t ⊙ tanh(c_t)                           [Hidden state]
  ```
- **Why LSTM Works**:
  - Additive cell state gradient: ∂c_t/∂c_{t-1} = f_t (can ≈ 1!)
  - vs RNN: ∂h_t/∂h_{t-1} = W_hh^T * tanh'(z) (often << 1)
  - Forget gate learns: f_t ≈ 1 preserves, f_t ≈ 0 forgets
- **Backward Pass** (EXTREMELY DETAILED):
  - Critical: grad_c has TWO sources
    1. Direct from output: grad_y through o_t
    2. From next cell: grad_c_next through f_{t+1}
  - Gate gradients depend on cell state
  - Multiplicative interactions in backward pass
- **Key Components**:
  - LSTMCell: Single time step
  - StackedLSTM: Multiple layers
- **Applications**: Universal for RNN tasks (NMT, speech, time series)

### 3. gru_cell.py (394 lines)
- **Core Concept**: Simplified LSTM with 2 gates (reset, update)
- **Trade-offs**: ~2/3 parameters of LSTM, comparable performance
- **Gates**:
  ```
  r_t = σ(W_ir @ x_t + W_hr @ h_{t-1} + b_r)     [Reset gate]
  z_t = σ(W_iz @ x_t + W_hz @ h_{t-1} + b_z)     [Update gate]
  h̃_t = tanh(W_ih @ x_t + W_hh @ (r_t ⊙ h_{t-1}) + b_h) [Candidate]
  h_t = (1-z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}          [Update (interpolation)]
  ```
- **Interpretation**:
  - r_t: What to reset (affects candidate computation)
  - z_t: What to update (interpolates between candidate and previous)
  - No separate cell state (hidden state is the state)
- **When to use**:
  - Limited data → fewer parameters prevent overfitting
  - Moderate dependencies → matches LSTM on most tasks
  - Faster training → ~20% speedup vs LSTM
- **Key Components**:
  - GRUCell: Single time step
  - BidirectionalGRU: Both directions
- **Backward Pass**: Complex due to multiplicative interactions in reset gate

### 4. bidirectional.py (421 lines)
- **Core Concept**: Processes sequence forward AND backward, concatenates
- **Architecture**:
  ```
  h_f_t = RNN_forward(x_t, h_f_{t-1})  [left-to-right]
  h_b_t = RNN_backward(x_t, h_b_{t+1}) [right-to-left]
  y_t = [h_f_t; h_b_t]                 [concatenated 2*d_h]
  ```
- **Key Components**:
  - BidirectionalWrapper: Wraps any RNN cell
  - StackedBidirectional: Multiple bidirectional layers
  - Support for variable-length sequences with masks
- **Gradient Flow**:
  - Each position gets gradients from both directions
  - ∂L/∂x_t = ∂L through forward + ∂L through backward
- **Trade-offs**:
  - Better representations (future context available)
  - 2x memory and computation
  - Cannot be used for streaming/online
- **Applications**: BERT, ELMo, tagging tasks (NER, POS), BLSTM
- **Complexity**:
  - Time: O(2T × d_h × (d_in + d_h))
  - Memory: O(2T × d_h) for BPTT

## Summary Statistics

| Module | Files | Total Lines | Key Classes |
|--------|-------|-------------|------------|
| Attention | 6 | 1,853 | ScaledDotProduct, MultiHead, MQA, GQA, CausalMask, CrossAttention |
| Recurrent | 4 | 1,705 | RNNCell, LSTMCell, GRUCell, Bidirectional |
| **Total** | **10** | **3,558** | **10 main classes** |

## Documentation Coverage

Each file includes:

1. **Theory Section** (comprehensive)
   - Motivation and why the component matters
   - Mathematical equations with LaTeX notation
   - Gradient flow analysis
   - Computational complexity

2. **Architecture** (detailed)
   - Forward pass algorithm with steps
   - Shapes at each transformation
   - How information flows
   - Variants and alternatives

3. **Backward Pass** (BPTT)
   - Gradient derivations
   - Chain rule through gates/computations
   - Matrix dimension tracking
   - Implementation notes for stability

4. **Hyperparameters & Tips**
   - Initialization strategies
   - Gradient clipping guidance
   - Learning rate considerations
   - When to use each variant

5. **Applications**
   - Real models using these components
   - Performance characteristics
   - Production considerations

6. **Implementation Hints**
   - Specific numpy operations needed
   - Shape transformations
   - Efficiency tips
   - Common pitfalls

## Implementation Status

All files contain:
- ✅ Full module docstrings with theory
- ✅ Complete class definitions
- ✅ Method signatures with detailed docstrings
- ✅ Shape documentation
- ✅ Mathematics in comments
- ✅ References to papers and resources
- ❌ NotImplementedError with specific implementation hints
- ❌ Code bodies (stubs ready for implementation)

Each NotImplementedError includes:
- Step-by-step algorithm outline
- Specific numpy operations to use
- Shape transformations needed
- Common pitfalls to avoid
- Optimization tips

## Usage

To implement any component:
1. Open the corresponding file
2. Find the method with NotImplementedError
3. Follow the numbered hints with specific operations
4. Reference the math and shapes in the docstring
5. Test with the documented shapes

Example:
```python
# For ScaledDotProductAttention.forward():
# 1. scores = (query @ key.swapaxes(-2, -1)) / sqrt(d_k)
# 2. Apply mask: np.where(mask, scores, -1e9)
# 3. weights = softmax(scores)
# 4. output = weights @ value
```
