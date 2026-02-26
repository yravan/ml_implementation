"""
Linear Attention Mechanism: O(n) Complexity

Efficient attention that approximates standard softmax attention in O(n) time
and O(n) space by using kernel methods to approximate softmax.

Theory:
========
Standard Scaled Dot-Product Attention has O(n^2) complexity:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

The softmax(QK^T) is a dense matrix of size (n × n) where n is sequence length.
For long sequences (1000+), this becomes prohibitively expensive in both
time and memory.

Linear Attention approximates this using the kernel trick:

    softmax(x) ≈ φ(x)  (feature map approximation)

Where φ(x) is a kernel feature map. Then:

    Attention ≈ (Φ(Q) @ (Φ(K)^T @ V)) / (Φ(Q) @ Σ_i Φ(K_i))

This can be computed in O(n) time by reorganizing:

    output_i = Σ_j (Φ(Q_i) · Φ(K_j)) V_j / Σ_j (Φ(Q_i) · Φ(K_j))

Notice that the denominator is just normalizing the weighted sum.
Using associativity of matrix multiplication:

    numerator = Φ(Q) @ (Φ(K)^T @ V)  [O(n*d) instead of O(n^2*d)]
    denominator = Φ(Q) @ (Φ(K)^T @ 1)  [just sum of Φ(K)]

This reorganization is the key: instead of computing (n × n) matrix,
we compute (d × d) intermediate matrix where d = feature dimension.

Kernel Feature Maps:
===================

1. ELU + 1 kernel (Tsai et al., 2019):
    φ(x) = elu(x) + 1

   Simple, effective, but may be negative
   Solution: use max(φ(x), ε) to ensure numerical stability

2. Exponential kernel (via Taylor expansion):
    φ(x) = exp(x)

   Direct approximation, but numerically unstable
   Mitigate by subtracting maximum: exp(x - max(x))

3. Polynomial kernels:
    φ(x) = (1 + x)^p

   Stable, interpretable, but may accumulate error

4. Cosine kernel (Rethinking Attention with Performers, Choromanski et al., 2020):
    φ_i(x) = cos(w_i^T x + b_i)  for random w_i, b_i

   Uses random projections to approximate exponential kernel
   Reduces to Performer architecture

Mathematical Justification:
===========================

Standard attention:
    α_ij = exp(x_ij) / Σ_k exp(x_ik)  where x_ij = (q_i · k_j) / sqrt(d)

Approximation via kernel features:
    α_ij ≈ (φ(x_ij)) / Σ_k φ(x_ik)

Why this works: softmax is "almost" linear on its domain. By using a feature
map φ that captures the non-linearity, we can approximate softmax.

Key insight: if φ(x) = exp(x), then α_ij = exp(x_ij) / Σ_k exp(x_ik) exactly!
But exp is expensive to compute stably, so we use cheaper approximations.

Error Analysis:
===============

The approximation error depends on:
1. Choice of kernel (how well φ approximates exp)
2. Numerical stability (avoiding overflow/underflow)
3. Normalization term (denominator computation)

For ELU + 1 kernel with feature dimension d_f:
    Approximation error: O(d^{-1/4}) (diminishing with feature dimension)

This means: more feature dimensions → better approximation → slower but more accurate

Trade-off: increase feature_dim from d (attention dim) to higher value for better
approximation, at cost of computation.

Theoretical Analysis (Tsai et al., 2019):
- Linear attention with ELU + 1 kernel can express any function in limit
- With sufficient feature dimensions, can approximate standard attention arbitrarily well
- In practice: d_f = d or d_f = 2d gives good results

Implementation Tricks:
====================

1. Numerical Stability:
   - Subtract maximum from query-key products before applying kernel
   - Subtract maximum from normalizer separately to prevent NaN
   - Use log-space computation for stability

2. Feature Computation:
   - Batch compute φ(Q) and φ(K) for entire sequence at once
   - Precompute and cache kernel features if repeated

3. Normalization:
   - Compute denominator once for each query position
   - Can accumulate as we go in streaming scenarios

4. Memory Efficiency:
   - Instead of storing (n × n) attention matrix, store only:
     * Query features: (batch, n, d_f)
     * Key features: (batch, n, d_f)
     * KV products: (batch, d_f, d_v)
   - Total: O(n*d_f) instead of O(n^2)

Connection to RNNs:
==================

Linear attention can be viewed as an RNN:

    state_t = state_{t-1} + φ(k_t) ⊙ v_t^T
    output_t = φ(q_t)^T @ state_t / φ(q_t)^T @ sum_t

Where ⊙ is outer product. This is a linear RNN!

Advantages over standard RNNs:
- Parallelizable (compute all states in parallel, unlike RNN)
- Handles long sequences better (no hidden state bottleneck)

Key Papers:
===========
1. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
   (Katharopoulos et al., 2020)
   - First linear attention paper
   - Shows equivalence to RNNs
   - ELU + 1 kernel

2. "Rethinking Attention with Performers" (Choromanski et al., 2020)
   - Performer architecture
   - Random feature projection for exponential kernel
   - Achieves O(n log n) complexity with random features

3. "Linear Transformers Are Secretly Fast Weight Memory Systems"
   (Schlag et al., 2021)
   - Theoretical analysis of linear attention
   - Connection to fast weight algorithms
   - Interpretation of state transitions

4. "Hyperbolic Attention with Hyperbolic Bilinear Layers"
   (Yang et al., 2021)
   - Non-Euclidean variant of linear attention
   - Improved approximation of softmax

Performance:
============
Time complexity:
- Standard attention: O(n^2 * d)
- Linear attention: O(n * d^2) or O(n * d_f * d) where d_f = feature_dim
- Practical: linear is faster for n > 1000-2000

Space complexity:
- Standard attention: O(n^2)
- Linear attention: O(n * d_f)

For long documents (n=10000), linear attention is 100x faster!

Limitations:
============
1. Approximation error: doesn't exactly compute softmax attention
2. Loss of some expressiveness: may struggle with specific attention patterns
3. Feature dimension trade-off: more features = better but slower
4. Causal masking harder: must ensure causality in feature map
5. May underperform on tasks with strong positional dependencies

When to use:
============
- Long sequences (n > 1000)
- Limited memory budgets
- Real-time inference (latency critical)
- Tasks where approximate attention is acceptable

When NOT to use:
- Short sequences (n < 100): standard attention fine
- Tasks needing exact softmax attention patterns
- Machine translation (spatial awareness important)
"""

from typing import Optional, Tuple
import numpy as np


class LinearAttention:
    """
    Linear attention mechanism with O(n) complexity.

    Uses kernel features to approximate softmax attention efficiently.
    """

    def __init__(self, dim: int, kernel_type: str = "elu_plus_one",
                 feature_dim: Optional[int] = None):
        """
        Initialize Linear Attention.

        Args:
            dim: dimensionality of queries, keys, values
            kernel_type: type of kernel approximation
                        "elu_plus_one": elu(x) + 1 (Katharopoulos et al.)
                        "sigmoid": sigmoid(x) approximation
                        "random_features": use random projections
            feature_dim: dimensionality of feature map
                        (default: dim, can increase for better approximation)
        """
        self.dim = dim
        self.kernel_type = kernel_type
        self.feature_dim = feature_dim or dim

        # TODO: For random features kernel, initialize random projection matrix
        if kernel_type == "random_features":
            # TODO: Random projection: (dim, feature_dim)
            self.random_proj = None  # (dim, feature_dim)
            self.random_bias = None  # (feature_dim,)

        # TODO: Initialize projection matrices for Q, K, V if needed
        # For efficient computation, may want learnable projections
        self.W_q = None
        self.W_k = None
        self.W_v = None

    def kernel_feature_map(self, x: np.ndarray) -> np.ndarray:
        """
        Compute kernel feature map φ(x).

        Args:
            x: input scores, shape (batch, seq_len, feature_dim or dim)

        Returns:
            features: kernel features, shape (batch, seq_len, feature_dim)
        """
        if self.kernel_type == "elu_plus_one":
            # TODO: Implement elu(x) + 1
            # Ensure strictly positive: max(elu(x) + 1, eps)
            pass

        elif self.kernel_type == "sigmoid":
            # TODO: Implement sigmoid(x)
            # Sigmoid is always in (0, 1), naturally bounded
            pass

        elif self.kernel_type == "random_features":
            # TODO: Project input to random features
            # cos and sin for approximating exp kernel
            pass

        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for linear attention.

        Linear complexity: O(n * d^2) instead of O(n^2 * d)

        Args:
            query: queries, shape (batch_size, n_q, dim)
            key: keys, shape (batch_size, n_k, dim)
            value: values, shape (batch_size, n_v, dim)
            mask: optional mask, shape (batch_size, n_k)

        Returns:
            output: attention output, shape (batch_size, n_q, dim)
            features: kernel features for analysis
        """
        batch_size = query.shape[0]
        n_q = query.shape[1]
        n_k = key.shape[1]

        # TODO: Compute scaled query-key scores
        # scores = (query @ key.T) / sqrt(dim)
        # shape: (batch, n_q, n_k)

        # TODO: Apply kernel feature map to scores
        # query_features = kernel_feature_map(query @ key.T / sqrt(d))
        # But more efficient: compute features directly
        # φ(q_i)·φ(k_j) ≈ φ(q_i · k_j)
        # Actually: we want φ(q_i^T @ k_j / sqrt(d))
        # For efficiency, compute φ(key) first, then φ(query @ key)

        # TODO: Compute key features: φ(K)
        # key_features = kernel_feature_map(...)  # (batch, n_k, feature_dim)

        # TODO: Compute query features: φ(Q)
        # query_features = kernel_feature_map(...)  # (batch, n_q, feature_dim)

        # TODO: Efficient computation:
        # Instead of (n_q, n_k) @ (n_k, d_v) = (n_q, d_v)
        # Compute: (n_q, f_d) @ (f_d, f_d) @ (f_d, d_v) = (n_q, d_v)
        # where f_d = feature_dim

        # TODO: Compute KV product: K^T @ V
        # kv = (batch, feature_dim, feature_dim) = key_features.T @ value
        # This is (n_k, feature_dim)^T @ (n_k, d_v) = (feature_dim, d_v)

        # TODO: Compute normalizer: K^T @ 1 (sum of features over keys)
        # normalizer = (batch, feature_dim) = sum(key_features, axis=1)

        # TODO: Compute output: Q @ KV / (Q @ 1)
        # output = (query_features @ kv) / (query_features @ normalizer + eps)

        # TODO: Handle masking if provided
        # Set masked key positions to 0 before computing KV product

        # TODO: Return output and features (for visualization/analysis)
        pass

    def backward(self, doutput: np.ndarray, cache: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for linear attention.

        Args:
            doutput: gradient w.r.t. output
            cache: cache from forward pass

        Returns:
            dquery: gradient w.r.t. query
            dkey: gradient w.r.t. key
            dvalue: gradient w.r.t. value
        """
        # TODO: Implement backprop through linear attention
        # Key insight: gradients flow through accumulated KV product

        pass


class CausalLinearAttention:
    """
    Linear attention with causality constraint (for autoregressive models).

    Ensures decoder cannot attend to future tokens in sequence-to-sequence or
    language modeling tasks.
    """

    def __init__(self, dim: int, kernel_type: str = "elu_plus_one",
                 feature_dim: Optional[int] = None):
        """
        Initialize Causal Linear Attention.

        Args:
            dim: dimensionality of attention
            kernel_type: kernel type (see LinearAttention)
            feature_dim: feature dimension for kernel approximation
        """
        self.dim = dim
        self.kernel_type = kernel_type
        self.feature_dim = feature_dim or dim

        # TODO: Reuse LinearAttention implementation

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with causal masking.

        Args:
            query: queries, shape (batch_size, seq_len, dim)
            key: keys, shape (batch_size, seq_len, dim)
            value: values, shape (batch_size, seq_len, dim)

        Returns:
            output: attention output, shape (batch_size, seq_len, dim)
            features: kernel features
        """
        # TODO: Enforce causality: for position i, can only attend to j <= i
        # Can be done via:
        # 1. Masked computation (explicit masking in each step)
        # 2. Causal kernel (kernel that is 0 for i < j)
        # 3. Cumulative computation (maintain running KV product)

        # Most efficient: cumulative/online computation
        # For each position i:
        #   state_i = state_{i-1} + φ(k_i) ⊙ v_i^T
        #   output_i = φ(q_i)^T @ state_i / (φ(q_i)^T @ sum_j≤i φ(k_j))

        pass


class MultiHeadLinearAttention:
    """
    Multi-head variant of linear attention.

    Processes multiple "attention heads" in parallel for richer representations.
    """

    def __init__(self, dim: int, num_heads: int, kernel_type: str = "elu_plus_one",
                 feature_dim: Optional[int] = None):
        """
        Initialize Multi-Head Linear Attention.

        Args:
            dim: total dimensionality (must be divisible by num_heads)
            num_heads: number of attention heads
            kernel_type: kernel type for each head
            feature_dim: feature dimension per head
        """
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_type = kernel_type
        self.feature_dim = feature_dim or self.head_dim

        # TODO: Initialize projection matrices
        self.W_q = None  # (dim, dim)
        self.W_k = None  # (dim, dim)
        self.W_v = None  # (dim, dim)
        self.W_out = None  # (dim, dim)

        # TODO: Initialize individual heads
        self.heads = []  # num_heads LinearAttention instances

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass with multiple attention heads.

        Args:
            query: (batch_size, seq_len, dim)
            key: (batch_size, seq_len, dim)
            value: (batch_size, seq_len, dim)
            mask: optional mask

        Returns:
            output: (batch_size, seq_len, dim)
        """
        # TODO: Project input
        # TODO: Split into heads
        # TODO: Apply linear attention to each head in parallel
        # TODO: Concatenate head outputs
        # TODO: Project output
        # TODO: Return final output

        pass


if __name__ == "__main__":
    # Test linear attention
    batch_size, seq_len, dim = 4, 100, 64

    # TODO: Create sample data
    # query = np.random.randn(batch_size, seq_len, dim)
    # key = np.random.randn(batch_size, seq_len, dim)
    # value = np.random.randn(batch_size, seq_len, dim)

    # TODO: Apply linear attention
    # attn = LinearAttention(dim, kernel_type="elu_plus_one")
    # output, features = attn.forward(query, key, value)
    # print(f"Output shape: {output.shape}")
    # print(f"Features shape: {features.shape}")
