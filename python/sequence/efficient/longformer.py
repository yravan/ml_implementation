"""
Longformer: Efficient Transformer for Long Documents

Combines local (sliding window) attention and global attention to process
long sequences efficiently while maintaining full document awareness.

Theory:
========
The quadratic complexity of standard Transformer attention O(n^2) makes it
impractical for documents longer than ~512 tokens. Longformer introduces
a hybrid attention pattern:

1. Local attention: each token attends only to nearby tokens in a window
2. Global attention: selected tokens can attend to entire sequence

This reduces complexity from O(n^2) to approximately O(n * w) where w is window size,
while maintaining the ability for important tokens to see full context.

Attention Pattern:

Standard (Dense) Attention:
    α_ij = softmax((q_i · k_j) / sqrt(d))  for all i,j
    Complexity: O(n^2)
    Memory: O(n^2)

Sliding Window Attention:
    α_ij = softmax((q_i · k_j) / sqrt(d))  for |i - j| ≤ w/2
    Where w is window size
    Complexity: O(n * w)
    Memory: O(n * w)

Example with window size 4:
    Position 0 can attend to: [0, 1, 2]
    Position 1 can attend to: [0, 1, 2, 3]
    Position 2 can attend to: [0, 1, 2, 3, 4]
    ...
    Position n can attend to: [n-2, n-1, n]

Hybrid Attention (Longformer):
    α_ij = softmax((q_i · k_j) / sqrt(d))
    for: |i - j| ≤ w/2 OR i in global_positions OR j in global_positions

Global positions might be: [0, 512, 1024, ...] or task-specific tokens

This allows:
- Most tokens: local context (fast)
- Key tokens: full context (expressive)
- Balance: good speed and accuracy

Mathematical Formulation:
=======================

Sliding window attention for position i:
    output_i = Σ_{j: |i-j| ≤ w/2} α_ij * v_j

Where α_ij = exp((q_i · k_j) / sqrt(d)) / Σ_k exp((q_i · k_k) / sqrt(d))

Efficient computation via reshape:
1. Transpose sequence into windows (reshape trick)
   - For window size w: reshape (n, d) → (n, w, d) with sliding window
   - This enables fast batched matrix multiplication

2. Compute attention within each window independently
   - All windows computed in parallel (efficient on GPUs)
   - Window size w (e.g., 64) is manageable

3. Transpose back to get output

Global attention adds conditional logic:
    If i is global: α_ij = softmax(...) for all j (expensive, but few globals)
    If j is global: include in all position's attention (efficient broadcast)

Complexity Analysis:
===================

Standard Transformer:
    - Layers: L
    - Sequence length: n
    - Embedding dimension: d
    - Attention complexity per layer: O(n^2 * d)
    - Total: O(L * n^2 * d)

Longformer with window w and global tokens g:
    - Local attention: O(L * n * w * d)
    - Global attention: O(L * g * n * d) + O(L * n * g * d)
    - Total: O(L * n * (w + g) * d)

For w = 512, g = 4, n = 4096:
    - Standard: O(L * 4096^2 * d) = O(16M * L * d)
    - Longformer: O(L * 4096 * 516 * d) = O(2M * L * d)
    - Speedup: 8x!

For even longer documents (n = 16384):
    - Standard: O(256M * L * d) - infeasible
    - Longformer: O(8M * L * d) - practical

Key Papers:
===========
1. "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
   - Original Longformer paper
   - Introduces local + global attention pattern
   - SOTA on long document tasks
   - Efficient implementation details

2. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
   (Dai et al., 2019)
   - Segment-level recurrence for longer context
   - Relative positional embeddings
   - Can be combined with Longformer

3. "ETC: An Episode Transformer with Context Compression" (Sperlich et al., 2021)
   - Similar local-global approach for long documents
   - Adds compression mechanism
   - Better than Longformer on some tasks

4. "Big Bird: Transformers for Longer Sequences" (Zaheer et al., 2020)
   - Similar pattern: local + global + random attention
   - Adds random attention tokens
   - Theoretical analysis of expressiveness

Architecture Details:
====================

1. Local Attention Module:
   - Window size: typically 64-512
   - Can be expanded with "dilated" windows (sparse attention)
   - Computed efficiently via reshape trick

2. Global Attention Module:
   - Can mark specific positions as global (e.g., [CLS] token, section headers)
   - Or learn which positions should be global during training
   - Global positions computed separately (slower but few)

3. Positional Embeddings:
   - Standard absolute embeddings may not work well for long sequences
   - Relative positional embeddings recommended (Transformer-XL style)
   - Can use rotary embeddings for efficiency

4. Mixed Precision:
   - Local attention: can use float16 (fast, sufficient local context)
   - Global attention: use float32 (need precision for full document)
   - Attention weights: float32 for numerical stability

Implementation Strategy:
=======================

Efficient Local Attention Computation:

1. Create sliding window representation:
   ```
   # For window size w, position i attends to [i-w/2, ..., i+w/2]
   # Reshape sequence into windows
   # Input: (batch, n, d)
   # Window output: (batch, n, w, d)
   ```

2. Compute attention within windows:
   ```
   # All windows are independent
   # Batch matrix multiplication: (batch*n, w, d)
   # Compute QK^T for all windows in parallel
   # This is O(n*w^2*d) instead of O(n^2*d)
   ```

3. Handle window boundaries:
   ```
   # First w/2 positions: truncated window
   # Last w/2 positions: truncated window
   # Middle positions: full window
   # OR: pad sequence with padding tokens
   ```

4. Transpose back to sequence format

Efficient Global Attention:

1. Identify global positions (fixed or learned)
2. For global positions:
   - Compute full QK^T with all positions (expensive but few)
3. For other positions:
   - Compute QK^T only with global positions (cheap broadcast)
4. Merge with local attention via masking

Common Pitfalls:
- Memory alignment for reshape operations
- Boundary effects at window edges
- Numerical stability in softmax over wide windows
- Communication of global information across layers

Performance Characteristics:
============================
Memory:
- Standard 12-layer BERT at length 4096: ~48GB (infeasible)
- Longformer-base at length 4096: ~6GB (on high-end GPU)
- Speedup: 8x faster, 8x less memory

Training:
- Longformer slower than BERT on short sequences (<512)
- Competitive on medium sequences (512-2048)
- Much faster on long sequences (>2048)

Inference:
- Can process very long documents in one pass
- Good for single-pass reading comprehension
- Streaming: can process document in chunks with sliding window

Practical Considerations:
=======================

Window Size Selection:
- Larger window: more context, slower computation
- Smaller window: less context, faster computation
- Typical range: 64-512
- For most tasks: 256-512 good balance

Global Token Selection:
- Task-specific: e.g., [CLS] for classification, headers for long documents
- Can learn which tokens should be global
- Usually 4-16 global tokens per layer
- More globals = more computation but better understanding

Pre-training:
- Longformer requires special pre-training or adaptation from BERT
- Cannot directly use BERT weights (attention patterns don't match)
- Need to interpolate position embeddings to longer sequences
- Can fine-tune from BERT with modified attention

Variants and Extensions:
=======================

1. Dilated Attention:
   - Instead of contiguous window [i-w/2, ..., i+w/2]
   - Use sparse pattern: [i-2w, i-w, i, i+w, i+2w]
   - Larger effective receptive field, same computation

2. Learned Attention Patterns:
   - Dynamically learn which positions to attend to
   - Sparse attention determined by learned scores
   - More flexible but harder to implement efficiently

3. Hierarchical Attention:
   - Multiple levels with different window sizes
   - Lower layers: small windows (local patterns)
   - Higher layers: large windows (document structure)

4. Task-Specific Attention:
   - Question answering: global attention on question tokens
   - Summarization: global attention on sentence starters
   - Can be learned or fixed based on task

Comparison with Other Approaches:
==================================

Longformer vs Linear Attention:
- Linear attention: O(n) but approximate (doesn't compute exact softmax)
- Longformer: O(n*w) but exact local attention
- Linear: better for very long sequences (10K+)
- Longformer: better for moderate length with exact attention

Longformer vs Sparse Transformers:
- Both use sparse attention patterns
- Sparse Transformer: learned patterns (very flexible but complex)
- Longformer: fixed patterns (simpler, more efficient)
- Longformer faster for most practical tasks

Longformer vs Hierarchical Transformers:
- Hierarchical: compress document into chunks, then attention
- Longformer: direct attention with local windows
- Longformer: better for tasks needing position-level detail
- Hierarchical: better for very long documents (20K+)

When to Use:
============
Use Longformer when:
- Document length 1K-16K tokens
- Need exact attention computation (not approximation)
- Have compute resources for moderate sequences
- Task requires fine-grained positional understanding

Use Linear Attention when:
- Document length >20K tokens
- Approximate attention acceptable
- Memory-constrained environment

Use Hierarchical Transformers when:
- Document length >100K tokens
- Can afford chunking document
- Task allows summary-level processing
"""

from typing import Optional, Tuple
import numpy as np


class LocalAttention:
    """
    Sliding window local attention for efficient sequence processing.

    Each position attends only to nearby positions within a window.
    Enables O(n*w) complexity instead of O(n^2).
    """

    def __init__(self, dim: int, window_size: int, num_heads: int = 1):
        """
        Initialize Local Attention.

        Args:
            dim: dimensionality of attention
            window_size: size of attention window
                        each position attends to [i-w/2, ..., i+w/2]
            num_heads: number of attention heads
        """
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # TODO: Initialize projection matrices
        self.W_q = None  # (dim, dim)
        self.W_k = None  # (dim, dim)
        self.W_v = None  # (dim, dim)
        self.W_out = None  # (dim, dim)

    def _create_sliding_windows(self, x: np.ndarray) -> np.ndarray:
        """
        Create sliding window representation for efficient computation.

        Args:
            x: input, shape (batch, seq_len, dim)

        Returns:
            windows: windowed representation (batch, seq_len, window_size, dim)
        """
        # TODO: Implement sliding window creation
        # For each position i, create window [i-w/2, ..., i+w/2]
        # Use stride tricks or explicit indexing
        # Handle boundary cases (padding)

        pass

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass with local attention.

        Complexity: O(seq_len * window_size^2 * dim)

        Args:
            query: (batch_size, seq_len, dim)
            key: (batch_size, seq_len, dim)
            value: (batch_size, seq_len, dim)
            mask: optional mask for padding

        Returns:
            output: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = query.shape

        # TODO: Project input
        # TODO: Create sliding windows
        # TODO: Compute attention within each window
        # TODO: Handle boundary effects (truncated windows at edges)
        # TODO: Reshape back to sequence format
        # TODO: Return output

        pass


class GlobalAttention:
    """
    Global attention for selected "important" tokens.

    Allows certain tokens to attend to entire sequence while other tokens
    attend locally. Enables efficient long document understanding.
    """

    def __init__(self, dim: int, num_global_tokens: int = 4):
        """
        Initialize Global Attention.

        Args:
            dim: dimensionality of attention
            num_global_tokens: number of positions marked as global
        """
        self.dim = dim
        self.num_global_tokens = num_global_tokens

        # TODO: Initialize projection matrices
        self.W_q = None  # (dim, dim)
        self.W_k = None  # (dim, dim)
        self.W_v = None  # (dim, dim)

    def forward(self, x: np.ndarray, global_indices: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with global attention.

        Args:
            x: input sequence, shape (batch_size, seq_len, dim)
            global_indices: indices of global tokens (batch_size, num_global)
                           if None, treat first token as global

        Returns:
            output: attention output, shape (batch_size, seq_len, dim)
            attention_weights: for visualization
        """
        # TODO: Compute full attention for global tokens
        # TODO: For non-global tokens, attend to global tokens only
        # TODO: Combine into output
        # TODO: Return output and weights

        pass


class Longformer:
    """
    Longformer model: combines local and global attention for long documents.

    Enables efficient processing of sequences up to 4096 tokens and beyond
    while maintaining document-level understanding through global attention.
    """

    def __init__(self, dim: int, window_size: int = 256, num_global: int = 4,
                 num_layers: int = 12, num_heads: int = 12, ffn_dim: int = 3072,
                 vocab_size: int = 30522):
        """
        Initialize Longformer.

        Args:
            dim: embedding dimensionality
            window_size: local attention window size
            num_global: number of global tokens per layer
            num_layers: number of transformer layers
            num_heads: number of attention heads
            ffn_dim: feed-forward network hidden dimension
            vocab_size: vocabulary size for embeddings
        """
        self.dim = dim
        self.window_size = window_size
        self.num_global = num_global
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.vocab_size = vocab_size

        # TODO: Initialize embedding layer
        self.embedding = None  # (vocab_size, dim)
        self.position_embedding = None  # (max_seq_len, dim)

        # TODO: Initialize layers
        self.layers = []  # List of (LocalAttention + GlobalAttention) pairs

        # TODO: Initialize projection heads for downstream tasks
        self.classification_head = None  # (dim, num_classes)

    def forward(self, input_ids: np.ndarray, global_token_mask: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through Longformer.

        Args:
            input_ids: token IDs, shape (batch_size, seq_len)
            global_token_mask: which tokens are global (batch_size, seq_len)
                              1 for global, 0 for local
                              if None, only [CLS] token is global

        Returns:
            hidden_states: final hidden states (batch_size, seq_len, dim)
            cls_output: representation of [CLS] token for classification (batch_size, dim)
        """
        # TODO: Embed tokens
        # TODO: Add position embeddings
        # TODO: For each layer:
        #   1. Apply local attention to local tokens
        #   2. Apply global attention to global tokens
        #   3. Combine results
        #   4. Apply feed-forward network
        # TODO: Return final hidden states and CLS representation

        pass

    def forward_for_classification(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass for classification task.

        Args:
            input_ids: token IDs (batch_size, seq_len)

        Returns:
            logits: classification logits (batch_size, num_classes)
        """
        # TODO: Forward through Longformer
        # TODO: Extract CLS token representation (first token)
        # TODO: Project through classification head
        # TODO: Return logits

        pass

    def forward_for_qa(self, input_ids: np.ndarray, token_type_ids: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for question-answering task.

        Args:
            input_ids: token IDs (batch_size, seq_len)
            token_type_ids: segment IDs distinguishing question and context

        Returns:
            start_logits: answer span start position logits
            end_logits: answer span end position logits
        """
        # TODO: Embed with token type information
        # TODO: Mark question tokens as global (important for QA)
        # TODO: Forward through Longformer
        # TODO: Project hidden states to start/end logits
        # TODO: Return logits for start and end positions

        pass


class DilatedAttention:
    """
    Dilated/sparse attention pattern for larger receptive field.

    Instead of contiguous window [i-w/2, ..., i+w/2], uses sparse pattern
    like [i-2w, i-w, i, i+w, i+2w] for larger effective context.
    """

    def __init__(self, dim: int, window_size: int, dilation_rate: int = 2,
                 num_heads: int = 1):
        """
        Initialize Dilated Attention.

        Args:
            dim: dimensionality
            window_size: base window size
            dilation_rate: sparsity factor (how many positions to skip)
            num_heads: number of attention heads
        """
        self.dim = dim
        self.window_size = window_size
        self.dilation_rate = dilation_rate
        self.num_heads = num_heads

        # TODO: Initialize attention parameters

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) \
            -> np.ndarray:
        """
        Forward pass with dilated attention pattern.

        Args:
            query: (batch_size, seq_len, dim)
            key: (batch_size, seq_len, dim)
            value: (batch_size, seq_len, dim)

        Returns:
            output: (batch_size, seq_len, dim)
        """
        # TODO: Create dilated window pattern
        # TODO: Apply attention to sparse positions only
        # TODO: Merge results
        # TODO: Return output

        pass


if __name__ == "__main__":
    # Test Longformer
    batch_size, seq_len, dim = 4, 1024, 768

    # TODO: Create model
    # longformer = Longformer(dim=dim, window_size=256, num_global=4, num_layers=12)

    # TODO: Create sample input
    # input_ids = np.random.randint(0, 30522, (batch_size, seq_len))

    # TODO: Forward pass
    # hidden, cls_out = longformer.forward(input_ids)
    # print(f"Hidden states shape: {hidden.shape}")
    # print(f"CLS output shape: {cls_out.shape}")
