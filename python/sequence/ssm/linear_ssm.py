"""
Linear State Space Models (SSM) for Sequence Processing

Fundamental theory of state space models and their application to sequence
learning. Foundation for modern SSMs like S4 and Mamba.

Theory:
========
State space models describe dynamical systems via:

Continuous-time state space model:
    ẋ(t) = A x(t) + B u(t)
    y(t) = C x(t) + D u(t)

Where:
- x(t): hidden state (latent dynamics)
- u(t): input (observation)
- y(t): output
- A: state transition matrix
- B: input-to-state matrix
- C: state-to-output matrix
- D: feedthrough/residual connection

Discretization:
To apply to sequences, we discretize time using step size Δ:

    x_k = A_d x_{k-1} + B_d u_k
    y_k = C x_k + D u_k

Where discretized matrices depend on continuous A, B, C and step size Δ.

Common discretization: Bilinear (zero-order hold):
    A_d = (I + A*Δ/2) (I - A*Δ/2)^{-1}
    B_d = (I - A*Δ/2)^{-1} B_d

Alternative: Exponential:
    A_d = exp(A*Δ)
    B_d = A^{-1}(A_d - I) B

Mathematical Properties:
========================

1. Linearity:
   The system is entirely linear! No nonlinearity except input embedding/output projection.
   This enables several advantages:
   - Easy to analyze theoretically
   - Can use linear algebra tricks
   - Connections to linear algebra and signal processing

2. State dimension vs sequence length:
   - State x_k has fixed dimension (e.g., 256)
   - Doesn't depend on sequence length
   - Enables linear-time processing (key advantage!)

3. Stability:
   For stability, eigenvalues of A must have negative real parts:
   eigenvalues(A) ⊂ {z : Re(z) < 0}

   After discretization:
   eigenvalues(A_d) must have magnitude < 1

   Stable systems: state doesn't explode, gradients don't blow up

4. Impulse response:
   System's response to single input pulse determines all behavior
   h_k = C A_d^k B

   If decay: ||A_d^k|| → 0 exponentially
   Can represent long-range dependencies through matrix powers

Relationship to RNNs:
====================

SSM forward pass:
    x_k = A_d x_{k-1} + B_d u_k
    y_k = C x_k

RNN forward pass:
    h_k = f(W_h h_{k-1} + W_u u_k + b)

Similarities:
- Both maintain recurrent state
- Both process sequences sequentially
- Both can model long-range dependencies

Differences:
- SSM: linear (except in embedding/output)
- RNN: nonlinear throughout
- SSM: A_d fixed, no activation
- RNN: W_h learned per-layer with activation

Connection to Transformers:
===========================

Transformer processes sequence in parallel using self-attention.
SSM processes sequence sequentially using state recurrence.

SSM advantages:
- Linear time: O(n) per forward pass (vs O(n^2) for attention)
- Linear space: O(1) per step (vs O(n) for attention)
- Can parallelize via convolution (HiPPO paper)

Transformer advantages:
- Full context: each position attends to all others
- Flexible routing: learned attention patterns
- Better for certain tasks (translation, language modeling)

Why SSMs didn't work traditionally:
1. Information bottleneck: state is small, must compress entire history
2. Gradient flow: long sequences, exponential decay
3. Learning difficulty: A matrices hard to initialize

Recent breakthroughs (S4, Mamba):
1. Better parameterization of A (structured matrices)
2. Better initialization (HiPPO)
3. Fast algorithms (kernel trick for conv)
4. Selective state transitions (input-dependent)

Practical Implementation:
=========================

Efficient computation via convolution:

Forward pass as matrix multiplication (naive):
    [x_1]   [0    ]       [B_d]
    [x_2] = [A_d  ] x_0 + [A_d B_d     ] [u_1]
    [x_3]   [A_d^2]       [A_d^2 B_d   ] [u_2]
                          [...]         [u_3]

Output: y = Cx (can parallelize this)

But this is still O(n) space. Instead, compute as convolution:

    y_k = Σ_{j=0}^k C A_d^{k-j} B_d u_j

Which is convolution of:
    h = [CB_d, CA_d B_d, CA_d^2 B_d, ...]  (impulse response)
    u = [u_1, u_2, u_3, ...]

So: y = h * u (convolution)

This enables efficient computation via FFT:
- Compute impulse response: O(n)
- FFT of h and u: O(n log n)
- Multiply: O(n)
- Inverse FFT: O(n log n)
Total: O(n log n) per layer!

During training:
- Can use convolution formulation: fast
- Or use RNN formulation: gives gradients naturally

Key Papers:
===========
1. "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
   (Gu et al., 2020)
   - Principled way to initialize A matrices
   - Orthogonal polynomial basis for history
   - Enables long-range dependencies

2. "Efficiently Modeling Long Sequences with Structured State Spaces"
   (Gu et al., 2021) - The S4 paper
   - Structured parameterization of A
   - Fast algorithms via kernel trick
   - SOTA on long-range dependencies

3. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   (Gu & Dao, 2023)
   - Input-selective state transitions
   - Modern variant with input-dependent behavior
   - Competitive with Transformers

4. "The Effectiveness of State Space Models on Vision"
   (Zhu et al., 2023)
   - Shows SSMs work for vision tasks
   - Demonstrates generality beyond language

Architecture Details:
====================

1. State dimension:
   Typically 256, 512, 1024 for language tasks
   Larger state = more expressive but slower

2. Parameterization:
   Can use: Dense A, diagonal A, triangular A, etc.
   S4: uses structured matrices for efficiency

3. Nonlinearity:
   Input projection: u_emb = relu(W_u * u)
   Output projection: out = W_o * (y + residual)
   Gating: similar to gating in RNNs

4. Multiple layers:
   Stack multiple SSM layers
   Use residual connections between layers
   Similar to transformer stacking

Efficiency Comparison:
====================

Time complexity (per forward pass):
- RNN: O(n * h^2) where h = state dim, n = seq len
- Transformer: O(n^2 * d) where d = embedding dim
- SSM (naive): O(n * h)
- SSM (conv): O(n log n)

Space complexity:
- RNN: O(n * h) [store all hidden states]
- Transformer: O(n * d) [store all attention matrices]
- SSM: O(n) [only store impulse response of length n]

For n=1000, h=256, d=768:
- RNN: O(256K)
- Transformer: O(768M) ← much larger
- SSM: O(n) ← smallest

Advantages of SSMs:
==================
1. Linear time and space with respect to sequence length
2. Principled initialization (HiPPO)
3. Can be parallelized via convolution
4. Gradients stable for long sequences
5. Interpretable as differential equations

Disadvantages:
==============
1. Smaller state than attention window (information bottleneck)
2. Can't model attention patterns as flexibly
3. Harder to train initially (needs good initialization)
4. Less research/adoption than Transformers
5. Harder to parallelize training across many devices

When to Use:
============
Use SSMs when:
- Sequence length very long (10K+ tokens)
- Model needs to process efficiently online/streaming
- Long-range dependencies important
- Memory is constrained

Use Transformers when:
- Sequence length moderate (512-2048)
- Have compute resources
- Task benefits from flexible attention
- Need maximum performance (better tuned)
"""

from typing import Optional, Tuple
import numpy as np


class LinearSSM:
    """
    Basic Linear State Space Model for sequence processing.

    Implements discrete-time linear state space model:
        x_k = A_d x_{k-1} + B_d u_k
        y_k = C x_k + D u_k
    """

    def __init__(self, state_dim: int, input_dim: int, output_dim: int,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        """
        Initialize Linear SSM.

        Args:
            state_dim: dimensionality of hidden state
            input_dim: dimensionality of input
            output_dim: dimensionality of output
            dt_min: minimum discretization step size
            dt_max: maximum discretization step size
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt_min = dt_min
        self.dt_max = dt_max

        # TODO: Initialize continuous-time state space matrices
        # A: state transition matrix (state_dim, state_dim)
        # B: input-to-state matrix (state_dim, input_dim)
        # C: state-to-output matrix (output_dim, state_dim)
        # D: feedthrough (output_dim, input_dim)

        self.A = None  # (state_dim, state_dim)
        self.B = None  # (state_dim, input_dim)
        self.C = None  # (output_dim, state_dim)
        self.D = None  # (output_dim, input_dim)

        # TODO: Initialize discretization step size
        # Typically learned parameter or fixed
        self.dt = None  # scalar or (input_dim,) for input-dependent

        # Cached discretized matrices
        self.A_d = None
        self.B_d = None

    def discretize(self, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize continuous-time SSM to discrete-time using bilinear transform.

        Args:
            dt: discretization step size (default: self.dt)

        Returns:
            A_d: discretized state transition matrix
            B_d: discretized input matrix
        """
        if dt is None:
            dt = self.dt

        # TODO: Implement bilinear discretization
        # A_d = (I + A*dt/2) (I - A*dt/2)^{-1}
        # B_d = (I - A*dt/2)^{-1} B_d

        # Alternative: exponential discretization
        # A_d = exp(A*dt)
        # B_d = A^{-1}(A_d - I) B_d

        # TODO: Cache discretized matrices
        self.A_d = None  # TODO
        self.B_d = None  # TODO

        # TODO: Return discretized matrices
        pass

    def forward(self, u: np.ndarray, x0: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through SSM (sequential computation).

        Args:
            u: input sequence, shape (batch_size, seq_len, input_dim)
            x0: initial state, shape (batch_size, state_dim)

        Returns:
            y: output sequence, shape (batch_size, seq_len, output_dim)
            x_states: all hidden states, shape (batch_size, seq_len, state_dim)
            h_response: impulse response for convolution view
        """
        batch_size, seq_len, _ = u.shape

        if x0 is None:
            x0 = np.zeros((batch_size, self.state_dim))

        # TODO: Discretize if not already done
        # TODO: Initialize outputs and state tracking
        # TODO: For each timestep:
        #   x_k = A_d @ x_{k-1} + B_d @ u_k
        #   y_k = C @ x_k + D @ u_k
        # TODO: Compute impulse response h for convolution interpretation
        # h_k = C @ A_d^{k-1} @ B_d
        # TODO: Return outputs, states, and impulse response

        pass

    def forward_conv(self, u: np.ndarray) -> np.ndarray:
        """
        Forward pass using convolution formulation (parallel computation).

        Faster than sequential RNN-style forward pass.

        Args:
            u: input sequence, shape (batch_size, seq_len, input_dim)

        Returns:
            y: output sequence, shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = u.shape

        # TODO: Compute impulse response
        # h_k = C @ A_d^{k-1} @ B_d  for k = 0, 1, ..., seq_len-1

        # TODO: Reshape for convolution
        # Input u: (batch, seq_len, input_dim)
        # Impulse response h: (seq_len, output_dim, input_dim)
        # Need to apply per-input-dim convolution

        # TODO: Efficient option: use FFT convolution
        # 1. Compute impulse response: O(seq_len)
        # 2. FFT(h): O(seq_len * log(seq_len))
        # 3. FFT(u): O(seq_len * log(seq_len))
        # 4. Multiply: O(seq_len)
        # 5. IFFT: O(seq_len * log(seq_len))

        # TODO: Apply output transformation
        # y = C @ y_internal + D @ u

        # TODO: Return output

        pass

    def backward(self, dy: np.ndarray, cache: dict) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through SSM.

        Args:
            dy: gradient w.r.t. output
            cache: cache from forward pass

        Returns:
            du: gradient w.r.t. input
            dx0: gradient w.r.t. initial state
        """
        # TODO: Backward through C and D projections
        # TODO: BPTT through recurrence (backward in time)
        # TODO: Compute gradients w.r.t. A, B, C, D matrices

        pass


class HiPPOSSM(LinearSSM):
    """
    SSM with HiPPO (High-order Polynomial Projection Operators) initialization.

    Uses principled initialization of A matrix based on orthogonal polynomials
    for better long-range dependency modeling.
    """

    def __init__(self, state_dim: int, input_dim: int, output_dim: int,
                 polynomial_order: str = "legendre", dt_min: float = 0.001,
                 dt_max: float = 0.1):
        """
        Initialize HiPPO SSM.

        Args:
            state_dim: dimensionality of hidden state
            input_dim: dimensionality of input
            output_dim: dimensionality of output
            polynomial_order: type of orthogonal polynomial
                             "legendre": Legendre polynomials
                             "laguerre": Laguerre polynomials
                             "fourier": Fourier basis
            dt_min: minimum step size
            dt_max: maximum step size
        """
        super().__init__(state_dim, input_dim, output_dim, dt_min, dt_max)

        self.polynomial_order = polynomial_order

        # TODO: Initialize A matrix using HiPPO framework
        # A is structured based on orthogonal polynomial projections
        # This enables the SSM to maintain a compressed history of the input

    def initialize_hippo_matrix(self) -> np.ndarray:
        """
        Initialize A matrix using HiPPO framework.

        Returns:
            A: (state_dim, state_dim) HiPPO-structured matrix
        """
        # TODO: Implement HiPPO matrix initialization
        # For Legendre polynomials:
        #   A_{ij} = (2i+1)^{1/2} (2j+1)^{1/2} / 2  if i > j
        #          = i + 1  if i == j
        #          = 0  otherwise

        # This ensures:
        # 1. Stable (eigenvalues have negative real part)
        # 2. Can represent continuous functions well
        # 3. Avoids information bottleneck

        # TODO: Return initialized A matrix

        pass


class StackedSSM:
    """
    Multiple stacked SSM layers for deeper models.
    """

    def __init__(self, state_dim: int, input_dim: int, output_dim: int,
                 num_layers: int, hidden_dim: Optional[int] = None):
        """
        Initialize Stacked SSM.

        Args:
            state_dim: state dimensionality for each layer
            input_dim: input dimensionality
            output_dim: output dimensionality
            num_layers: number of SSM layers to stack
            hidden_dim: hidden dimension for nonlinear projections
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # TODO: Initialize SSM layers
        self.layers = []

        # TODO: Initialize nonlinear projections
        # Input projection: embed input before SSM
        self.W_in = None

        # Layer-wise projections
        self.W_layers = []

        # Output projection
        self.W_out = None

    def forward(self, u: np.ndarray) -> np.ndarray:
        """
        Forward pass through stacked SSMs.

        Args:
            u: input sequence, shape (batch_size, seq_len, input_dim)

        Returns:
            y: output, shape (batch_size, seq_len, output_dim)
        """
        # TODO: Project input
        # TODO: For each layer:
        #   1. Apply SSM forward pass
        #   2. Apply residual connection
        #   3. Apply nonlinear activation
        # TODO: Project output
        # TODO: Return output

        pass


if __name__ == "__main__":
    # Test SSM
    batch_size, seq_len = 4, 100
    state_dim, input_dim, output_dim = 256, 128, 64

    # TODO: Create SSM
    # ssm = LinearSSM(state_dim, input_dim, output_dim)

    # TODO: Create sample input
    # u = np.random.randn(batch_size, seq_len, input_dim)

    # TODO: Forward pass
    # y, x_states, h_response = ssm.forward(u)
    # print(f"Output shape: {y.shape}")
    # print(f"State shape: {x_states.shape}")

    # TODO: Test convolution forward pass
    # y_conv = ssm.forward_conv(u)
    # print(f"Conv output shape: {y_conv.shape}")
