"""
Temporal Convolutional Networks (TCN) for Sequence Modeling

Applies causal convolutions with dilations to capture multi-scale temporal patterns.
Offers alternative to RNNs with better parallelization and gradient flow.

Theory:
========
Traditional Approach (RNN):
    Process sequence sequentially: x_1, x_2, ..., x_T
    Hidden state maintains context: h_t = f(h_{t-1}, x_t)
    Advantage: handles variable-length sequences naturally
    Disadvantage: sequential computation, gradient vanishing over time

Convolutional Approach (TCN):
    Apply causal convolutions to past tokens
    y_t = W * [x_{t-1}, x_{t-2}, ..., x_{t-k}]
    Advantage: parallelizable, stable gradients
    Disadvantage: fixed receptive field per layer

Why Causal?
    Standard convolution looks at future timesteps (non-causal)
    Causal convolution: y_t depends only on [x_1, ..., x_t]
    Essential for autoregressive tasks (generation, prediction)

Dilated Convolutions:
    Standard convolution kernel size k: covers k timesteps
    Dilated convolution with dilation d: covers k*d timesteps
    Exponential expansion: d=1,2,4,8,... gives exponential receptive field

Mathematical Formulation:
========================

Standard Convolution:
    y_t = Σ_{i=0}^{k-1} w_i * x_{t-i}
    Covers range: [t-k+1, t]

Dilated Convolution (dilation d):
    y_t = Σ_{i=0}^{k-1} w_i * x_{t-d*i}
    Covers range: [t-d*(k-1), t]
    Exponential: with d=2, covers 2^n timesteps with log(n) layers

Example with k=2, d=1,2,4:
    Layer 1 (d=1): y_t = w_0*x_t + w_1*x_{t-1}
    Layer 2 (d=2): y_t = w_0*y_t^(1) + w_1*y_{t-2}^(1)
    Layer 3 (d=4): y_t = w_0*y_t^(2) + w_1*y_{t-4}^(2)

Receptive field growth: 2 -> 4 -> 8 tokens with 3 layers!

Residual Connections:
    TCN uses skip connections like ResNets:
    y_t^(l) = f(y_t^(l-1)) + y_t^(l-1)

    Benefits:
    - Better gradient flow
    - Allows deeper networks
    - Enables training of 100+ layer networks
    - Better optimization landscape

Advantages of TCN:
==================

1. Parallelization:
   - RNN: must process x_1, then x_2, etc. (sequential)
   - TCN: can process all timesteps in parallel (parallelizable)
   - For training: significant speedup (10-100x on modern hardware)
   - For inference: RNN may still be faster (single step at a time)

2. Gradient Flow:
   - RNN: gradients propagate through h_t → h_{t-1} → ... → h_1
            Can vanish exponentially: ||grad|| ~ λ^T where |λ| < 1
   - TCN: gradients from y_t directly to past timesteps
           Multiplicative factor: product of weight norms
           No exponential decay if well-initialized

3. Variable-Length Sequences:
   - RNN: natural handling (recurrence naturally variable)
   - TCN: fixed receptive field, can handle via masking
           Some timesteps may have receptive field beyond sequence start
           Handle via zero-padding or masking

4. Computational Efficiency:
   - RNN: O(T * H^2) where T = seq_len, H = hidden_dim
   - TCN: O(T * K * H^2) where K = kernel_size (usually small)
          More efficient per-layer if K < H

5. Easy Parallelization:
   - RNN: sequential (hard to parallelize across time)
   - TCN: easy to parallelize across time dimension
          Similar to image convolutions (well-optimized)

Disadvantages:
==============

1. Fixed Receptive Field:
   - Each layer covers fixed number of timesteps
   - Must use deep networks for long sequences
   - May need many layers to model long dependencies

2. Position Bias:
   - Output y_t depends more on recent timesteps
   - x_t has larger influence than x_{t-1000}
   - Unlike attention which can weight any position equally
   - For many tasks, position-invariant processing better

3. Memory for Training:
   - Need to store activations for all timesteps
   - For backprop, need activation history
   - O(T * H * L) memory for T timesteps, H hidden, L layers
   - RNN can recompute forward in backward pass (less memory)

4. Sequence Length Dependence:
   - Must process all timesteps at once
   - Can't easily do streaming/online processing
   - RNN can process one step at a time

Key Papers:
===========
1. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for
   Sequence Modeling" (Bai et al., 2018)
   - Original TCN paper
   - Comprehensive comparison with RNNs
   - Shows TCN can outperform RNNs
   - Introduces residual causal convolutions

2. "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)
   - Uses dilated causal convolutions for audio generation
   - Exponential dilation pattern
   - Inspired TCN architecture

3. "Dilated Residual Networks" (Yu & Koltun, 2016)
   - Introduces dilated convolutions for semantic segmentation
   - Exponential dilation pattern
   - Foundation for TCN architecture

4. "Very Deep Convolutional Networks for Large-Scale Image Recognition"
   (Simonyan & Zisserman, 2014)
   - Deep residual networks
   - Shows deep networks can work with proper initialization
   - Skip connections enable training

Architecture:
==============

Basic TCN Block:
1. Causal convolution: expands receptive field
2. Activation: ReLU or other nonlinearity
3. Dropout: regularization
4. Repeat for depth
5. Residual connection: skip across block

Residual Block:
    x → Conv(k, d) → ReLU → Dropout → Conv(k, d) → ReLU → Dropout → + → y
    └────────────────────────────────────────────────────────────────┘

Stack blocks with increasing dilation:
    Input → TCN(d=1) → TCN(d=2) → TCN(d=4) → ... → TCN(d=2^L)

Growing receptive field exponentially!

Implementation Details:
=====================

Causal Padding:
    Convolution y[i] = Σ_j w[j] * x[i+j]
    For causality: need x[i-k+1, ..., i] available
    Solution: zero-pad input on left with (k-1) zeros

    For dilated with dilation d:
    Need to pad with (k-1)*d zeros on left

Efficient Implementation:
    # Instead of padding explicitly
    Conv1d with padding=0, manually pad input
    x_padded = zero_pad(x, left=(k-1)*d, right=0)
    y = Conv1d(x_padded, kernel_size=k, dilation=d)

    Or use PyTorch's padding parameter directly:
    Conv1d(padding=(k-1)*d, dilation=d)
    Then remove right padding: y = y[:, :(original_length)]

Gradient Flow:
    Backprop through residual connection enables:
    ∇x = ∇y (identity) + ∇f(x) (function gradients)

    With proper initialization, ||∇f(x)|| ≈ 0.1
    Result: ||∇x|| ≈ 1 (not vanishing!)

    Compare with RNN: ||∇h_0|| ~ λ^T with |λ| < 1 (exponential decay)

Performance Comparison:
======================

Accuracy (benchmark tasks):
- TCN: competitive or better than RNN/LSTM on many tasks
- Task-dependent: some favor RNN, some TCN

Training Speed:
- TCN: 10-100x faster on GPUs (parallelizable)
- RNN: slower due to sequential nature
- Hardware matters: TCN needs proper GPU utilization

Inference Speed:
- TCN: slow if need output for all timesteps (compute all at once)
- RNN: can be faster for single token (one step per forward pass)
- Streaming: RNN natural, TCN not ideal

Memory:
- TCN: O(T) space for storing activations
- RNN: can use gradient checkpointing (less memory, slower backprop)
- Very long sequences: RNN might use less memory

When to Use TCN:
================

Use TCN when:
1. Parallelization important (training efficiency)
2. Standard NLP tasks (mostly look at recent context)
3. Long sequence length not extreme
4. Have powerful GPUs to parallelize
5. Position bias acceptable or expected

Use RNN when:
1. Streaming/online processing needed
2. Variable-length sequences in batch
3. Long-range dependencies important
4. Model needs to "pay attention" to distant past
5. Hardware-constrained (CPU inference)

Use Transformer when:
1. Need to model all-pair relationships
2. Sequence length not too long (<2048)
3. Task benefits from flexible routing
4. Have compute for O(n^2)

Modern Trends:
==============

After Transformers (2017):
- Transformers became dominant in NLP
- TCN less popular
- But still useful for:
  * Audio/speech tasks (WaveNet-style)
  * Time series prediction
  * Tasks with position bias
  * Low-latency requirements

Recent: Return to SSMs/Mamba:
- Even better than Transformers for efficiency
- Also handles long sequences well
- May replace both RNN and TCN in future

But TCN remains useful:
- Simpler than attention (easier to understand)
- Good for domain-specific applications
- Still competitive on some benchmarks
- Well-understood theory and practice
"""

from typing import Optional, Tuple, List
import numpy as np


class CausalConv1D:
    """
    1D Causal Convolution for temporal sequences.

    Ensures no information flows from future to past timesteps.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1):
        """
        Initialize Causal Convolution.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: size of convolution kernel
            dilation: dilation rate for dilated convolution
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        # TODO: Calculate padding needed for causality
        # For dilated conv with dilation d and kernel k:
        # Padding = (k-1) * d (pad on left only)
        self.padding_left = (kernel_size - 1) * dilation
        self.padding_right = 0

        # TODO: Initialize weight and bias
        self.W = None  # (out_channels, in_channels, kernel_size)
        self.b = None  # (out_channels,)

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass of causal convolution.

        Args:
            x: input sequence, shape (batch_size, in_channels, seq_len)

        Returns:
            y: output, shape (batch_size, out_channels, seq_len)
            cache: for backward pass
        """
        batch_size, _, seq_len = x.shape

        # TODO: Apply causal padding (left only)
        # x_padded = zero_pad(x, left=self.padding_left, right=0)

        # TODO: Perform 1D convolution
        # For each position i:
        #   y[i] = Σ_k w[k] * x[i-dilation*k]

        # TODO: Add bias
        # TODO: Cache for backward
        # TODO: Return output and cache

        pass

    def backward(self, dy: np.ndarray, cache: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through causal convolution.

        Args:
            dy: gradient w.r.t. output
            cache: cache from forward pass

        Returns:
            dx: gradient w.r.t. input
            dw: gradient w.r.t. weights
        """
        # TODO: Implement backward through convolution
        # TODO: Handle dilation and causality in backward
        # TODO: Compute weight gradients
        # TODO: Return input gradient and weight gradient

        pass


class ResidualBlock:
    """
    Residual block for TCN with causal convolutions.

    Combines two causal convolutions with nonlinearity and residual connection.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.5):
        """
        Initialize Residual Block.

        Args:
            channels: number of channels (in and out)
            kernel_size: kernel size for convolutions
            dilation: dilation rate
            dropout: dropout probability
        """
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout

        # TODO: Initialize two causal convolutions
        self.conv1 = CausalConv1D(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1D(channels, channels, kernel_size, dilation)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through residual block.

        Args:
            x: input, shape (batch_size, channels, seq_len)

        Returns:
            y: output, shape (batch_size, channels, seq_len)
            cache: for backward pass
        """
        # TODO: First convolution
        # y = self.conv1.forward(x)

        # TODO: Activation (ReLU)
        # y = relu(y)

        # TODO: Dropout for regularization
        # TODO: Second convolution

        # TODO: Activation

        # TODO: Dropout

        # TODO: Residual connection
        # y = y + x  [skip connection]

        # TODO: Cache and return

        pass

    def backward(self, dy: np.ndarray, cache: dict) -> np.ndarray:
        """
        Backward pass through residual block.

        Args:
            dy: gradient w.r.t. output
            cache: cache from forward pass

        Returns:
            dx: gradient w.r.t. input
        """
        # TODO: Backprop through residual addition
        # dx = dy  [from skip connection]

        # TODO: Backprop through second conv
        # TODO: Backprop through dropout
        # TODO: Backprop through activation (ReLU)

        # TODO: Backprop through first conv
        # TODO: Backprop through dropout
        # TODO: Backprop through activation

        # TODO: Combine with skip gradient
        # TODO: Return input gradient

        pass


class TemporalConvolutionalNetwork:
    """
    Temporal Convolutional Network (TCN) for sequence modeling.

    Stacks residual blocks with increasing dilation for exponential receptive field.
    """

    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 4,
                 kernel_size: int = 5, dropout: float = 0.5):
        """
        Initialize TCN.

        Args:
            in_channels: input dimensionality
            out_channels: output dimensionality
            num_levels: number of residual blocks (levels)
            kernel_size: convolution kernel size
            dropout: dropout probability
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.kernel_size = kernel_size

        # TODO: Initialize residual blocks
        # Each level has exponential dilation: 2^0, 2^1, 2^2, ...
        self.blocks = []
        for level in range(num_levels):
            dilation = 2 ** level
            # Input to first block: in_channels
            # Input to subsequent blocks: in_channels
            block = ResidualBlock(in_channels, kernel_size, dilation, dropout)
            self.blocks.append(block)

        # TODO: Output projection
        self.W_out = None  # (out_channels, in_channels)
        self.b_out = None  # (out_channels,)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through TCN.

        Args:
            x: input sequence, shape (batch_size, seq_len, in_channels)
               Note: may need to transpose to (batch, channels, seq_len)

        Returns:
            y: output, shape (batch_size, seq_len, out_channels)
        """
        batch_size, seq_len, _ = x.shape

        # TODO: Transpose to (batch, channels, seq_len)
        # x = x.transpose(0, 2, 1)

        # TODO: Pass through each residual block
        # y = x
        # for block in self.blocks:
        #     y, cache = block.forward(y)

        # TODO: Project to output dimension
        # y = project(y, self.W_out, self.b_out)

        # TODO: Transpose back to (batch, seq_len, channels)
        # y = y.transpose(0, 2, 1)

        # TODO: Return output

        pass

    def backward(self, dy: np.ndarray, learn_rate: float = 0.01) -> np.ndarray:
        """
        Backward pass through TCN.

        Args:
            dy: gradient w.r.t. output
            learn_rate: learning rate for weight updates

        Returns:
            dx: gradient w.r.t. input
        """
        # TODO: Transpose to (batch, channels, seq_len)
        # TODO: Backprop through output projection
        # TODO: Backprop through each residual block (in reverse order)
        # TODO: Update weights
        # TODO: Transpose back and return

        pass


class DilatedTCN:
    """
    TCN variant with explicit dilation control.

    Allows for non-exponential dilation patterns if desired.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dilations: List[int], kernel_size: int = 3,
                 dropout: float = 0.5):
        """
        Initialize Dilated TCN.

        Args:
            in_channels: input dimension
            out_channels: output dimension
            dilations: list of dilation rates per block
                      e.g., [1, 2, 4, 8] for exponential
                      or custom like [1, 2, 3, 5] for custom pattern
            kernel_size: kernel size for all convolutions
            dropout: dropout probability
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.kernel_size = kernel_size

        # TODO: Initialize blocks with specified dilations
        self.blocks = []
        for dilation in dilations:
            block = ResidualBlock(in_channels, kernel_size, dilation, dropout)
            self.blocks.append(block)

        # TODO: Output projection
        self.W_out = None
        self.b_out = None

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass with custom dilations.

        Args:
            x: input (batch_size, seq_len, in_channels)

        Returns:
            y: output (batch_size, seq_len, out_channels)
            cache: for backward pass
        """
        # TODO: Similar to TemporalConvolutionalNetwork.forward
        # but using custom dilations

        pass


class GatedTCN:
    """
    TCN with gating mechanism (inspired by highway networks and gated RNNs).

    May improve expressiveness compared to standard TCN.
    """

    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 4,
                 kernel_size: int = 5, dropout: float = 0.5):
        """
        Initialize Gated TCN.

        Args:
            in_channels: input dimension
            out_channels: output dimension
            num_levels: number of levels
            kernel_size: kernel size
            dropout: dropout probability
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels

        # TODO: Initialize main pathway (similar to standard TCN)
        # TODO: Initialize gating pathway
        # Gate learns to blend identity with transformation:
        # y = gate * transform(x) + (1 - gate) * x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with gating.

        Args:
            x: input (batch_size, seq_len, in_channels)

        Returns:
            y: output (batch_size, seq_len, out_channels)
        """
        # TODO: Compute transformation pathway
        # TODO: Compute gating pathway
        # TODO: Blend using gate: y = gate * transform + (1 - gate) * x
        # TODO: Return output

        pass


if __name__ == "__main__":
    # Test TCN
    batch_size, seq_len, in_channels, out_channels = 32, 100, 50, 64

    # TODO: Create TCN
    # tcn = TemporalConvolutionalNetwork(in_channels, out_channels, num_levels=4)

    # TODO: Create sample input
    # x = np.random.randn(batch_size, seq_len, in_channels)

    # TODO: Forward pass
    # y, cache = tcn.forward(x)
    # print(f"Output shape: {y.shape}")

    # TODO: Test dilated TCN with custom dilations
    # dilated_tcn = DilatedTCN(in_channels, out_channels,
    #                          dilations=[1, 2, 4, 8],
    #                          kernel_size=3)
    # y, cache = dilated_tcn.forward(x)
    # print(f"Dilated TCN output shape: {y.shape}")
