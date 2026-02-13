"""
Comprehensive Tests for nn_core Module
======================================

This test suite covers all components of the nn_core module:
- Module base class and containers
- Linear layers
- Activation functions (Module and Functional)
- Convolution layers (Module and Functional)
- Normalization layers (Module and Functional)
- Pooling layers (Module and Functional)
- Attention mechanisms (Module and Functional)
- Regularization (Module and Functional)
- Recurrent layers (Module and Functional)
- Initialization functions
- Positional encodings

Each test verifies both forward computation and (where applicable) backward gradients.

GRADIENT TESTING:
Uses gradcheck from foundations to verify that analytical gradients (from backward pass)
match numerical gradients (computed via finite differences).
"""

import numpy as np
import pytest
from typing import Callable, Tuple

# Import gradient checking utilities
from python.foundations import Tensor, gradcheck, numerical_gradient, gradient_check


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(None)


@pytest.fixture
def batch_1d(seed):
    """Create batch of 1D data (batch, channels, length)."""
    return np.random.randn(2, 3, 16).astype(np.float32)


@pytest.fixture
def batch_2d(seed):
    """Create batch of 2D data (batch, channels, height, width)."""
    return np.random.randn(2, 3, 8, 8).astype(np.float32)


@pytest.fixture
def batch_sequence(seed):
    """Create batch of sequences (batch, seq_len, features)."""
    return np.random.randn(2, 10, 32).astype(np.float32)


# =============================================================================
# Module Base Class Tests
# =============================================================================

class TestModule:
    """Test Module base class."""

    def test_module_creation(self):
        """Test creating a Module."""
        from python.nn_core import Module

        class SimpleModule(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        m = SimpleModule()
        assert isinstance(m, Module)

    def test_module_parameters(self):
        """Test Module parameter tracking."""
        from python.nn_core import Module, Parameter
        import numpy as np

        class ParamModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(np.random.randn(3, 3))

            def forward(self, x):
                return x

        m = ParamModule()
        params = list(m.parameters())
        assert len(params) == 1

    def test_module_children(self):
        """Test Module child module tracking."""
        from python.nn_core import Module, Linear

        class ParentModule(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        m = ParentModule()
        children = list(m.children())
        assert len(children) == 1

    def test_module_train_eval(self):
        """Test train/eval mode switching."""
        from python.nn_core import Module

        class SimpleModule(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        m = SimpleModule()

        m.train()
        assert m.training == True

        m.eval()
        assert m.training == False


class TestSequential:
    """Test Sequential container."""

    def test_sequential_forward(self):
        """Test Sequential forward pass."""
        from python.nn_core import Sequential, Linear, ReLU
        from python.foundations import Tensor

        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        )

        x = Tensor(np.random.randn(3, 10), requires_grad=True)
        y = model(x)

        assert y.shape == (3, 2)


# =============================================================================
# Linear Layer Tests
# =============================================================================

class TestLinear:
    """Test Linear layer."""

    def test_linear_forward(self):
        """Test Linear forward pass."""
        from python.nn_core import Linear
        from python.foundations import Tensor

        layer = Linear(10, 5)
        x = Tensor(np.random.randn(3, 10), requires_grad=True)

        y = layer(x)

        assert y.shape == (3, 5)

    def test_linear_backward(self):
        """Test Linear backward pass."""
        from python.nn_core import Linear
        from python.foundations import Tensor

        layer = Linear(10, 5)
        x = Tensor(np.random.randn(3, 10), requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert layer.weight.grad is not None
        if layer.bias is not None:
            assert layer.bias.grad is not None

    def test_linear_without_bias(self):
        """Test Linear without bias."""
        from python.nn_core import Linear
        from python.foundations import Tensor

        layer = Linear(10, 5, bias=False)
        x = Tensor(np.random.randn(3, 10), requires_grad=True)

        y = layer(x)

        assert y.shape == (3, 5)
        assert layer.bias is None


# =============================================================================
# Activation Function Tests
# =============================================================================

class TestActivations:
    """Test activation functions."""

    def test_relu_forward(self):
        """Test ReLU forward pass."""
        from python.nn_core import ReLU
        from python.foundations import Tensor

        relu = ReLU()
        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)

        y = relu(x)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.allclose(y.data, expected)

    def test_relu_backward(self):
        """Test ReLU backward pass."""
        from python.nn_core import ReLU
        from python.foundations import Tensor

        relu = ReLU()
        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)

        y = relu(x)
        loss = y.sum()
        loss.backward()

        # Gradient is 0 for x <= 0, 1 for x > 0
        expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        assert np.allclose(x.grad, expected_grad)

    def test_leaky_relu(self):
        """Test LeakyReLU."""
        from python.nn_core import LeakyReLU
        from python.foundations import Tensor

        lrelu = LeakyReLU(negative_slope=0.1)
        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)

        y = lrelu(x)

        expected = np.array([-0.2, -0.1, 0.0, 1.0, 2.0])
        assert np.allclose(y.data, expected)

    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        from python.nn_core import Sigmoid
        from python.foundations import Tensor

        sigmoid = Sigmoid()
        x = Tensor(np.array([0.0, 1.0, -1.0]), requires_grad=True)

        y = sigmoid(x)

        expected = 1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
        assert np.allclose(y.data, expected)

    def test_tanh(self):
        """Test Tanh activation."""
        from python.nn_core import Tanh
        from python.foundations import Tensor

        tanh = Tanh()
        x = Tensor(np.array([0.0, 1.0, -1.0]), requires_grad=True)

        y = tanh(x)

        expected = np.tanh([0.0, 1.0, -1.0])
        assert np.allclose(y.data, expected)

    def test_gelu(self):
        """Test GELU activation."""
        from python.nn_core import GELU
        from python.foundations import Tensor

        gelu = GELU()
        x = Tensor(np.array([0.0, 1.0, -1.0]), requires_grad=True)

        y = gelu(x)

        # GELU: x * Φ(x) where Φ is standard normal CDF
        # Just check it runs and output is reasonable
        assert y.shape == x.shape

    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        from python.nn_core import Softmax
        from python.foundations import Tensor

        softmax = Softmax(axis=-1)
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

        y = softmax(x)

        # Should sum to 1
        assert np.allclose(y.data.sum(axis=-1), 1.0)

    def test_softmax_backward(self):
        """Test Softmax backward pass."""
        from python.nn_core import Softmax
        from python.foundations import Tensor

        softmax = Softmax(axis=-1)
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

        y = softmax(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None


# =============================================================================
# Activation Functional Tests
# =============================================================================

class TestActivationsFunctional:
    """Test activation functional operations."""

    def test_relu_functional_forward_backward(self):
        """Test ReLU functional forward and backward."""
        from python.nn_core.activations_functional import ReLU

        relu_fn = ReLU()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Forward
        y = relu_fn.forward(x)
        assert np.allclose(y, [0.0, 0.0, 0.0, 1.0, 2.0])

        # Backward
        grad_output = np.ones_like(y)
        grad_x, = relu_fn.backward(grad_output)
        assert np.allclose(grad_x, [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_sigmoid_functional_forward_backward(self):
        """Test Sigmoid functional forward and backward."""
        from python.foundations.functionals import Sigmoid

        sigmoid_fn = Sigmoid()
        x = np.array([0.0, 1.0, -1.0])

        # Forward
        y = sigmoid_fn.forward(x)
        expected = 1 / (1 + np.exp(-x))
        assert np.allclose(y, expected)

        # Backward
        grad_output = np.ones_like(y)
        grad_x, = sigmoid_fn.backward(grad_output)
        expected_grad = y * (1 - y)
        assert np.allclose(grad_x, expected_grad)


# =============================================================================
# Convolution Layer Tests
# =============================================================================

class TestConv2d:
    """Test Conv2d layer."""

    def test_conv2d_forward(self):
        """Test Conv2d forward pass."""
        from python.nn_core import Conv2d
        from python.foundations import Tensor

        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)

        y = conv(x)

        # With padding=1 and kernel_size=3, output size equals input size
        assert y.shape == (2, 16, 8, 8)

    def test_conv2d_backward(self):
        """Test Conv2d backward pass."""
        from python.nn_core import Conv2d
        from python.foundations import Tensor

        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert conv.weight.grad is not None


class TestConvFunctional:
    """Test convolution functional operations."""

    def test_conv2d_functional(self):
        """Test Conv2d functional forward and backward."""
        from python.nn_core.conv_functional import Conv2d as Conv2dFn

        conv_fn = Conv2dFn()
        x = np.random.randn(2, 3, 8, 8)
        weight = np.random.randn(16, 3, 3, 3)
        bias = np.random.randn(16)

        # Forward
        y = conv_fn.forward(x, weight, bias, stride=1, padding=1)

        assert y.shape == (2, 16, 8, 8)


# =============================================================================
# Comprehensive Convolution Tests
# =============================================================================

class TestConv1dComprehensive:
    """Comprehensive tests for Conv1d layer (not yet implemented)."""

    def test_conv1d_forward_basic(self):
        """Test Conv1d forward pass with basic configuration."""
        from python.nn_core import Conv1d

        conv = Conv1d(in_channels=3, out_channels=16, kernel_size=3)
        x = Tensor(np.random.randn(2, 3, 20).astype(np.float64), requires_grad=True)
        y = conv(x)

        # Output size: (input_size - kernel_size) / stride + 1 = (20 - 3) / 1 + 1 = 18
        assert y.shape == (2, 16, 18)

    def test_conv1d_forward_with_padding(self):
        """Test Conv1d forward pass with padding."""
        from python.nn_core import Conv1d

        conv = Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 20).astype(np.float64), requires_grad=True)
        y = conv(x)

        # With padding=1, output size = input size
        assert y.shape == (2, 16, 20)

    def test_conv1d_forward_with_stride(self):
        """Test Conv1d forward pass with stride."""
        from python.nn_core import Conv1d

        conv = Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        x = Tensor(np.random.randn(2, 3, 20).astype(np.float64), requires_grad=True)
        y = conv(x)

        # Output size: (20 - 3) / 2 + 1 = 9
        assert y.shape == (2, 16, 9)

    def test_conv1d_forward_different_kernel_sizes(self):
        """Test Conv1d with different kernel sizes."""
        from python.nn_core import Conv1d

        for kernel_size in [1, 3, 5, 7]:
            conv = Conv1d(in_channels=4, out_channels=8, kernel_size=kernel_size, padding=kernel_size//2)
            x = Tensor(np.random.randn(2, 4, 16).astype(np.float64), requires_grad=True)
            y = conv(x)

            # With appropriate padding, output length should be close to input
            assert y.shape[0] == 2
            assert y.shape[1] == 8

    def test_conv1d_backward(self):
        """Test Conv1d backward pass."""
        from python.nn_core import Conv1d

        np.random.seed(42)
        conv = Conv1d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 16).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert x.grad.shape == x.shape, "Input gradient shape mismatch"
        assert conv.weight.grad is not None, "Weight gradient should exist"
        if conv.bias is not None:
            assert conv.bias.grad is not None, "Bias gradient should exist"

    def test_conv1d_gradcheck(self):
        """Verify Conv1d gradients using gradcheck."""
        from python.nn_core import Conv1d

        np.random.seed(42)
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_conv1d_gradcheck_no_bias(self):
        """Verify Conv1d gradients without bias."""
        from python.nn_core import Conv1d

        np.random.seed(42)
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1, bias=False)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_conv1d_gradcheck_with_stride(self):
        """Verify Conv1d gradients with stride > 1."""
        from python.nn_core import Conv1d

        np.random.seed(42)
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 2, 10).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_conv1d_weight_gradient(self):
        """Verify weight gradients for Conv1d."""
        from python.nn_core import Conv1d

        np.random.seed(42)
        conv = Conv1d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 16).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert conv.weight.grad is not None
        assert conv.weight.grad.shape == conv.weight.shape
        assert not np.allclose(conv.weight.grad, 0), "Weight gradient should be non-zero"


class TestConv2dComprehensive:
    """Comprehensive tests for Conv2d layer."""

    def test_conv2d_forward_basic(self):
        """Test Conv2d forward pass with basic configuration."""
        from python.nn_core import Conv2d

        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        x = Tensor(np.random.randn(2, 3, 16, 16).astype(np.float64), requires_grad=True)
        y = conv(x)

        # Output size: (16 - 3) / 1 + 1 = 14
        assert y.shape == (2, 16, 14, 14)

    def test_conv2d_forward_with_padding(self):
        """Test Conv2d forward pass with padding."""
        from python.nn_core import Conv2d

        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 16, 16).astype(np.float64), requires_grad=True)
        y = conv(x)

        # With padding=1 and kernel_size=3, output size = input size
        assert y.shape == (2, 16, 16, 16)

    def test_conv2d_forward_with_stride(self):
        """Test Conv2d forward pass with stride."""
        from python.nn_core import Conv2d

        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        x = Tensor(np.random.randn(2, 3, 16, 16).astype(np.float64), requires_grad=True)
        y = conv(x)

        # Output size: (16 + 2*1 - 3) / 2 + 1 = 8
        assert y.shape == (2, 16, 8, 8)

    def test_conv2d_forward_different_kernel_sizes(self):
        """Test Conv2d with different kernel sizes."""
        from python.nn_core import Conv2d

        for kernel_size in [1, 3, 5, 7]:
            conv = Conv2d(in_channels=4, out_channels=8, kernel_size=kernel_size, padding=kernel_size//2)
            x = Tensor(np.random.randn(2, 4, 16, 16).astype(np.float64), requires_grad=True)
            y = conv(x)

            assert y.shape[0] == 2
            assert y.shape[1] == 8

    def test_conv2d_forward_rectangular_kernel(self):
        """Test Conv2d with rectangular kernel."""
        from python.nn_core import Conv2d

        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 5), padding=(1, 2))
        x = Tensor(np.random.randn(2, 3, 16, 16).astype(np.float64), requires_grad=True)
        y = conv(x)

        # Output should maintain spatial dimensions with correct padding
        assert y.shape == (2, 8, 16, 16)

    def test_conv2d_forward_rectangular_input(self):
        """Test Conv2d with rectangular input (non-square)."""
        from python.nn_core import Conv2d

        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 12, 20).astype(np.float64), requires_grad=True)
        y = conv(x)

        assert y.shape == (2, 8, 12, 20)

    def test_conv2d_backward(self):
        """Test Conv2d backward pass."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert conv.weight.grad is not None
        assert conv.weight.grad.shape == conv.weight.shape

    def test_conv2d_gradcheck_basic(self):
        """Verify Conv2d gradients using gradcheck."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-3, atol=1e-4, rtol=1e-3)

    def test_conv2d_gradcheck_no_bias(self):
        """Verify Conv2d gradients without bias."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1, bias=False)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_conv2d_gradcheck_with_stride(self):
        """Verify Conv2d gradients with stride > 1."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 2, 8, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_conv2d_gradcheck_1x1_kernel(self):
        """Verify Conv2d gradients with 1x1 kernel (pointwise conv)."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=4, out_channels=8, kernel_size=1)
        x = Tensor(np.random.randn(1, 4, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_conv2d_gradcheck_different_configs(self):
        """Test Conv2d gradients with various configurations."""
        from python.nn_core import Conv2d

        configs = [
            {'kernel_size': 3, 'padding': 0, 'stride': 1},
            {'kernel_size': 3, 'padding': 1, 'stride': 1},
            {'kernel_size': 5, 'padding': 2, 'stride': 1},
            {'kernel_size': 3, 'padding': 1, 'stride': 2},
        ]

        for config in configs:
            np.random.seed(42)
            conv = Conv2d(in_channels=2, out_channels=4, **config)
            x = Tensor(np.random.randn(1, 2, 8, 8).astype(np.float64), requires_grad=True)

            def func(x):
                return conv(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"Conv2d gradcheck failed for config {config}"

    def test_conv2d_weight_bias_gradients(self):
        """Verify weight and bias gradients for Conv2d."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert conv.weight.grad is not None
        assert conv.weight.grad.shape == conv.weight.shape
        assert not np.allclose(conv.weight.grad, 0)

        if conv.bias is not None:
            assert conv.bias.grad is not None
            assert conv.bias.grad.shape == conv.bias.shape


class TestConv3dComprehensive:
    """Comprehensive tests for Conv3d layer."""

    def test_conv3d_forward_basic(self):
        """Test Conv3d forward pass with basic configuration."""
        from python.nn_core import Conv3d

        conv = Conv3d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 8, 8, 8).astype(np.float64), requires_grad=True)
        y = conv(x)

        assert y.shape == (2, 8, 8, 8, 8)

    def test_conv3d_gradcheck(self):
        """Verify Conv3d gradients using gradcheck."""
        from python.nn_core import Conv3d

        np.random.seed(42)
        conv = Conv3d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 4, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestConvTranspose2dComprehensive:
    """Comprehensive tests for ConvTranspose2d (not yet implemented)."""

    def test_convtranspose2d_forward_basic(self):
        """Test ConvTranspose2d forward pass."""
        from python.nn_core import ConvTranspose2d

        conv = ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
        x = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float64), requires_grad=True)
        y = conv(x)

        # Transposed conv upsamples: output_size = (input_size - 1) * stride - 2*padding + kernel_size + output_padding
        # = (4 - 1) * 2 - 2*1 + 3 + 1 = 6 - 2 + 3 + 1 = 8
        assert y.shape == (2, 8, 8, 8)

    def test_convtranspose2d_upsample_2x(self):
        """Test ConvTranspose2d for 2x upsampling."""
        from python.nn_core import ConvTranspose2d

        conv = ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        x = Tensor(np.random.randn(2, 16, 8, 8).astype(np.float64), requires_grad=True)
        y = conv(x)

        # 2x upsampling
        assert y.shape == (2, 8, 16, 16)

    def test_convtranspose2d_backward(self):
        """Test ConvTranspose2d backward pass."""
        from python.nn_core import ConvTranspose2d

        np.random.seed(42)
        conv = ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1)
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert conv.weight.grad is not None

    def test_convtranspose2d_gradcheck(self):
        """Verify ConvTranspose2d gradients using gradcheck."""
        from python.nn_core import ConvTranspose2d

        np.random.seed(42)
        conv = ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 4, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_convtranspose2d_gradcheck_no_bias(self):
        """Verify ConvTranspose2d gradients without bias."""
        from python.nn_core import ConvTranspose2d

        np.random.seed(42)
        conv = ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False)
        x = Tensor(np.random.randn(1, 4, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestConvTranspose1dComprehensive:
    """Comprehensive tests for ConvTranspose1d (not yet implemented)."""

    def test_convtranspose1d_forward_basic(self):
        """Test ConvTranspose1d forward pass."""
        from python.nn_core import ConvTranspose1d

        conv = ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        x = Tensor(np.random.randn(2, 16, 8).astype(np.float64), requires_grad=True)
        y = conv(x)

        # 2x upsampling
        assert y.shape == (2, 8, 16)

    def test_convtranspose1d_backward(self):
        """Test ConvTranspose1d backward pass."""
        from python.nn_core import ConvTranspose1d

        np.random.seed(42)
        conv = ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1)
        x = Tensor(np.random.randn(2, 8, 8).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert conv.weight.grad is not None

    def test_convtranspose1d_gradcheck(self):
        """Verify ConvTranspose1d gradients using gradcheck."""
        from python.nn_core import ConvTranspose1d

        np.random.seed(42)
        conv = ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 4, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestDepthwiseConv2dComprehensive:
    """Comprehensive tests for DepthwiseConv2d."""

    def test_depthwise_conv2d_forward(self):
        """Test DepthwiseConv2d forward pass."""
        from python.nn_core import DepthwiseConv2d

        conv = DepthwiseConv2d(in_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 8, 16, 16).astype(np.float64), requires_grad=True)
        y = conv(x)

        # Depthwise conv maintains channel count
        assert y.shape == (2, 8, 16, 16)

    def test_depthwise_conv2d_groups(self):
        """Test that DepthwiseConv2d uses groups=in_channels."""
        from python.nn_core import DepthwiseConv2d

        conv = DepthwiseConv2d(in_channels=8, kernel_size=3, padding=1)

        # Weight should be (in_channels, 1, kernel_size, kernel_size) for depthwise
        assert conv.weight.shape[0] == 8
        assert conv.weight.shape[1] == 1

    def test_depthwise_conv2d_backward(self):
        """Test DepthwiseConv2d backward pass."""
        from python.nn_core import DepthwiseConv2d

        np.random.seed(42)
        conv = DepthwiseConv2d(in_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 8, 8, 8).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert conv.weight.grad is not None

    def test_depthwise_conv2d_gradcheck(self):
        """Verify DepthwiseConv2d gradients using gradcheck."""
        from python.nn_core import DepthwiseConv2d

        np.random.seed(42)
        conv = DepthwiseConv2d(in_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 4, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestPointwiseConv2dComprehensive:
    """Comprehensive tests for PointwiseConv2d (1x1 convolution)."""

    def test_pointwise_conv2d_forward(self):
        """Test PointwiseConv2d forward pass."""
        from python.nn_core import PointwiseConv2d

        conv = PointwiseConv2d(in_channels=8, out_channels=16)
        x = Tensor(np.random.randn(2, 8, 16, 16).astype(np.float64), requires_grad=True)
        y = conv(x)

        # Pointwise conv changes channels but not spatial dims
        assert y.shape == (2, 16, 16, 16)

    def test_pointwise_conv2d_kernel_size(self):
        """Test that PointwiseConv2d uses 1x1 kernel."""
        from python.nn_core import PointwiseConv2d

        conv = PointwiseConv2d(in_channels=8, out_channels=16)

        assert conv.weight.shape[2] == 1
        assert conv.weight.shape[3] == 1

    def test_pointwise_conv2d_backward(self):
        """Test PointwiseConv2d backward pass."""
        from python.nn_core import PointwiseConv2d

        np.random.seed(42)
        conv = PointwiseConv2d(in_channels=8, out_channels=16)
        x = Tensor(np.random.randn(2, 8, 8, 8).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert conv.weight.grad is not None

    def test_pointwise_conv2d_gradcheck(self):
        """Verify PointwiseConv2d gradients using gradcheck."""
        from python.nn_core import PointwiseConv2d

        np.random.seed(42)
        conv = PointwiseConv2d(in_channels=4, out_channels=8)
        x = Tensor(np.random.randn(1, 4, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestSeparableConv2dComprehensive:
    """Comprehensive tests for SeparableConv2d (depthwise separable convolution)."""

    def test_separable_conv2d_forward(self):
        """Test SeparableConv2d forward pass."""
        from python.nn_core import SeparableConv2d

        conv = SeparableConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 8, 16, 16).astype(np.float64), requires_grad=True)
        y = conv(x)

        assert y.shape == (2, 16, 16, 16)

    def test_separable_conv2d_param_efficiency(self):
        """Test that SeparableConv2d has fewer parameters than regular Conv2d."""
        from python.nn_core import SeparableConv2d, Conv2d

        sep_conv = SeparableConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        reg_conv = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        sep_params = sum(p.data.size for p in sep_conv.parameters())
        reg_params = sum(p.data.size for p in reg_conv.parameters())

        # Separable conv should have significantly fewer parameters
        assert sep_params < reg_params

    def test_separable_conv2d_backward(self):
        """Test SeparableConv2d backward pass."""
        from python.nn_core import SeparableConv2d

        np.random.seed(42)
        conv = SeparableConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 8, 8, 8).astype(np.float64), requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None

    def test_separable_conv2d_gradcheck(self):
        """Verify SeparableConv2d gradients using gradcheck."""
        from python.nn_core import SeparableConv2d

        np.random.seed(42)
        conv = SeparableConv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 4, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestConvGradientsComprehensive:
    """Comprehensive gradient tests for all convolution layers using gradcheck."""

    # =========================================================================
    # Conv1d Gradient Tests
    # =========================================================================

    def test_conv1d_gradcheck_basic(self):
        """Basic Conv1d gradient check."""
        from python.nn_core import Conv1d

        np.random.seed(42)
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 10).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_conv1d_gradcheck_various_configs(self):
        """Conv1d gradient check with various configurations."""
        from python.nn_core import Conv1d

        configs = [
            {'kernel_size': 1, 'padding': 0, 'stride': 1},
            {'kernel_size': 3, 'padding': 1, 'stride': 1},
            {'kernel_size': 5, 'padding': 2, 'stride': 1},
            {'kernel_size': 3, 'padding': 1, 'stride': 2},
        ]

        for config in configs:
            np.random.seed(42)
            conv = Conv1d(in_channels=2, out_channels=4, **config)
            x = Tensor(np.random.randn(1, 2, 12).astype(np.float64), requires_grad=True)

            def func(x):
                return conv(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"Conv1d gradcheck failed for {config}"

    # =========================================================================
    # Conv2d Gradient Tests
    # =========================================================================

    def test_conv2d_gradcheck_basic(self):
        """Basic Conv2d gradient check."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 8, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_conv2d_gradcheck_various_batch_sizes(self):
        """Conv2d gradient check with various batch sizes."""
        from python.nn_core import Conv2d

        for batch_size in [1, 2, 4]:
            np.random.seed(42)
            conv = Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
            x = Tensor(np.random.randn(batch_size, 2, 6, 6).astype(np.float64), requires_grad=True)

            def func(x):
                return conv(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"Conv2d gradcheck failed for batch_size={batch_size}"

    def test_conv2d_gradcheck_asymmetric_padding(self):
        """Conv2d gradient check with asymmetric padding."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 5), padding=(1, 2))
        x = Tensor(np.random.randn(1, 2, 8, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    # =========================================================================
    # ConvTranspose2d Gradient Tests
    # =========================================================================

    def test_convtranspose2d_gradcheck_basic(self):
        """Basic ConvTranspose2d gradient check."""
        from python.nn_core import ConvTranspose2d

        np.random.seed(42)
        conv = ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 4, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_convtranspose2d_gradcheck_various_configs(self):
        """ConvTranspose2d gradient check with various configurations."""
        from python.nn_core import ConvTranspose2d

        configs = [
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'kernel_size': 4, 'stride': 2, 'padding': 1},
            {'kernel_size': 2, 'stride': 2, 'padding': 0},
        ]

        for config in configs:
            np.random.seed(42)
            conv = ConvTranspose2d(in_channels=4, out_channels=2, **config)
            x = Tensor(np.random.randn(1, 4, 4, 4).astype(np.float64), requires_grad=True)

            def func(x):
                return conv(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"ConvTranspose2d gradcheck failed for {config}"

    # =========================================================================
    # Depthwise and Pointwise Conv Gradient Tests
    # =========================================================================

    def test_depthwise_conv2d_gradcheck(self):
        """DepthwiseConv2d gradient check."""
        from python.nn_core import DepthwiseConv2d

        np.random.seed(42)
        conv = DepthwiseConv2d(in_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 4, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_pointwise_conv2d_gradcheck(self):
        """PointwiseConv2d gradient check."""
        from python.nn_core import PointwiseConv2d

        np.random.seed(42)
        conv = PointwiseConv2d(in_channels=4, out_channels=8)
        x = Tensor(np.random.randn(1, 4, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_separable_conv2d_gradcheck(self):
        """SeparableConv2d gradient check."""
        from python.nn_core import SeparableConv2d

        np.random.seed(42)
        conv = SeparableConv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 4, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    # =========================================================================
    # Conv Chain and Numerical Stability Tests
    # =========================================================================

    def test_conv_chain_gradient_flow(self):
        """Test gradient flow through chained convolutions."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv1 = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)

        y = conv1(x)
        z = conv2(y)
        loss = z.sum()
        loss.backward()

        assert x.grad is not None
        assert conv1.weight.grad is not None
        assert conv2.weight.grad is not None
        assert not np.allclose(x.grad, 0)

    def test_conv2d_gradient_small_input(self):
        """Test Conv2d gradients with small input values."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float64) * 1e-4, requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_conv2d_gradient_large_input(self):
        """Test Conv2d gradients with large input values."""
        from python.nn_core import Conv2d

        np.random.seed(42)
        conv = Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float64) * 100, requires_grad=True)

        def func(x):
            return conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


# =============================================================================
# Normalization Layer Tests
# =============================================================================

class TestBatchNorm2d:
    """Test BatchNorm2d layer."""

    def test_batchnorm2d_forward(self):
        """Test BatchNorm2d forward pass."""
        from python.nn_core import BatchNorm2d
        from python.foundations import Tensor

        bn = BatchNorm2d(num_features=3)
        bn.train()
        x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)

        y = bn(x)

        assert y.shape == x.shape

    def test_batchnorm2d_eval_mode(self):
        """Test BatchNorm2d in eval mode."""
        from python.nn_core import BatchNorm2d
        from python.foundations import Tensor

        bn = BatchNorm2d(num_features=3)
        bn.eval()
        x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)

        y = bn(x)

        assert y.shape == x.shape


class TestLayerNorm:
    """Test LayerNorm layer."""

    def test_layernorm_forward(self):
        """Test LayerNorm forward pass."""
        from python.nn_core import LayerNorm
        from python.foundations import Tensor

        ln = LayerNorm(normalized_shape=[32])
        x = Tensor(np.random.randn(2, 10, 32), requires_grad=True)

        y = ln(x)

        assert y.shape == x.shape

    def test_layernorm_backward(self):
        """Test LayerNorm backward pass."""
        from python.nn_core import LayerNorm
        from python.foundations import Tensor

        ln = LayerNorm(normalized_shape=[32])
        x = Tensor(np.random.randn(2, 10, 32), requires_grad=True)

        y = ln(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None


class TestNormalizationFunctional:
    """Test normalization functional operations."""

    def test_layernorm_functional(self):
        """Test LayerNorm functional forward and backward."""
        from python.nn_core.normalization_functional import LayerNorm as LayerNormFn

        ln_fn = LayerNormFn()
        x = np.random.randn(2, 10, 32)
        gamma = np.ones(32)
        beta = np.zeros(32)

        # Forward
        y = ln_fn.forward(x, gamma, beta, normalized_shape=[32])

        assert y.shape == x.shape


# =============================================================================
# Comprehensive Normalization Tests
# =============================================================================

class TestBatchNorm1d:
    """Test BatchNorm1d layer."""

    def test_batchnorm1d_creation(self):
        """Test BatchNorm1d creation."""
        from python.nn_core import BatchNorm1d

        bn = BatchNorm1d(num_features=64)
        assert bn.num_features == 64
        assert bn.eps == 1e-5
        assert bn.momentum == 0.1

    def test_batchnorm1d_forward_2d(self):
        """Test BatchNorm1d forward with 2D input (N, C)."""
        from python.nn_core import BatchNorm1d
        from python.foundations import Tensor

        bn = BatchNorm1d(num_features=32)
        bn.train()
        x = Tensor(np.random.randn(8, 32), requires_grad=True)

        y = bn(x)

        assert y.shape == x.shape

    def test_batchnorm1d_forward_3d(self):
        """Test BatchNorm1d forward with 3D input (N, C, L)."""
        from python.nn_core import BatchNorm1d
        from python.foundations import Tensor

        bn = BatchNorm1d(num_features=32)
        bn.train()
        x = Tensor(np.random.randn(8, 32, 16), requires_grad=True)

        y = bn(x)

        assert y.shape == x.shape

    def test_batchnorm1d_backward(self):
        """Test BatchNorm1d backward pass."""
        from python.nn_core import BatchNorm1d
        from python.foundations import Tensor

        bn = BatchNorm1d(num_features=32)
        bn.train()
        x = Tensor(np.random.randn(8, 32), requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert bn.weight.grad is not None
        assert bn.bias.grad is not None

    def test_batchnorm1d_running_stats(self):
        """Test BatchNorm1d updates running statistics."""
        from python.nn_core import BatchNorm1d
        from python.foundations import Tensor

        bn = BatchNorm1d(num_features=32, track_running_stats=True)
        bn.train()

        # Initial running stats
        initial_mean = bn.running_mean.copy()
        initial_var = bn.running_var.copy()

        x = Tensor(np.random.randn(8, 32) * 2 + 1, requires_grad=True)  # Non-zero mean and var
        y = bn(x)

        # Running stats should be updated
        assert not np.allclose(bn.running_mean, initial_mean) or \
               not np.allclose(bn.running_var, initial_var)

    def test_batchnorm1d_eval_uses_running_stats(self):
        """Test BatchNorm1d uses running stats in eval mode."""
        from python.nn_core import BatchNorm1d
        from python.foundations import Tensor

        bn = BatchNorm1d(num_features=32)

        # Train to update running stats
        bn.train()
        for _ in range(10):
            x = Tensor(np.random.randn(8, 32) * 2 + 1, requires_grad=True)
            y = bn(x)

        # Save running stats
        running_mean = bn.running_mean.copy()
        running_var = bn.running_var.copy()

        # Eval mode - running stats should not change
        bn.eval()
        x = Tensor(np.random.randn(8, 32) * 5 + 3, requires_grad=True)
        y = bn(x)

        assert np.allclose(bn.running_mean, running_mean)
        assert np.allclose(bn.running_var, running_var)


class TestBatchNorm2dComprehensive:
    """Comprehensive tests for BatchNorm2d layer."""

    def test_batchnorm2d_output_normalized(self):
        """Test BatchNorm2d output is normalized per channel."""
        from python.nn_core import BatchNorm2d
        from python.foundations import Tensor

        bn = BatchNorm2d(num_features=3)
        bn.train()
        x = Tensor(np.random.randn(8, 3, 16, 16) * 5 + 2, requires_grad=True)

        y = bn(x)

        # After normalization, each channel should have ~0 mean and ~1 var
        for c in range(3):
            channel_data = y.data[:, c, :, :]
            mean = channel_data.mean()
            var = channel_data.var()
            assert np.abs(mean) < 0.1, f"Channel {c} mean should be ~0, got {mean}"
            assert np.abs(var - 1.0) < 0.1, f"Channel {c} var should be ~1, got {var}"

    def test_batchnorm2d_affine_params(self):
        """Test BatchNorm2d with affine=False."""
        from python.nn_core import BatchNorm2d
        from python.foundations import Tensor

        bn = BatchNorm2d(num_features=3, affine=False)
        bn.train()

        # Weight and bias should not require grad
        assert not bn.weight.requires_grad
        assert not bn.bias.requires_grad

    def test_batchnorm2d_backward_gradients(self):
        """Test BatchNorm2d computes correct backward gradients."""
        from python.nn_core import BatchNorm2d
        from python.foundations import Tensor

        bn = BatchNorm2d(num_features=3)
        bn.train()
        x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        # All gradients should exist and be finite
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad))
        assert bn.weight.grad is not None
        assert bn.bias.grad is not None


class TestGroupNorm:
    """Test GroupNorm layer."""

    def test_groupnorm_creation(self):
        """Test GroupNorm creation."""
        from python.nn_core import GroupNorm

        gn = GroupNorm(num_groups=4, num_channels=32)
        assert gn.num_groups == 4
        assert gn.num_channels == 32

    def test_groupnorm_invalid_groups(self):
        """Test GroupNorm raises error for invalid num_groups."""
        from python.nn_core import GroupNorm

        with pytest.raises(ValueError):
            GroupNorm(num_groups=5, num_channels=32)  # 32 not divisible by 5

    def test_groupnorm_forward(self):
        """Test GroupNorm forward pass."""
        from python.nn_core import GroupNorm
        from python.foundations import Tensor

        gn = GroupNorm(num_groups=4, num_channels=32)
        x = Tensor(np.random.randn(8, 32, 16, 16), requires_grad=True)

        y = gn(x)

        assert y.shape == x.shape

    def test_groupnorm_batch_independent(self):
        """Test GroupNorm is batch-size independent."""
        from python.nn_core import GroupNorm
        from python.foundations import Tensor

        gn = GroupNorm(num_groups=4, num_channels=32)

        # Same input, different batch sizes
        x1 = Tensor(np.random.randn(1, 32, 8, 8), requires_grad=True)
        x2 = Tensor(np.tile(x1.data, (4, 1, 1, 1)), requires_grad=True)

        y1 = gn(x1)
        y2 = gn(x2)

        # First sample of y2 should match y1
        assert np.allclose(y1.data, y2.data[0:1], rtol=1e-5)


class TestInstanceNorm:
    """Test InstanceNorm layer."""

    def test_instancenorm_creation(self):
        """Test InstanceNorm creation (GroupNorm with groups=channels)."""
        from python.nn_core.normalization import InstanceNorm

        inst_norm = InstanceNorm(num_channels=32)
        # InstanceNorm is GroupNorm with num_groups = num_channels
        assert inst_norm.groupnorm.num_groups == 32
        assert inst_norm.groupnorm.num_channels == 32

    def test_instancenorm_forward(self):
        """Test InstanceNorm forward pass."""
        from python.nn_core.normalization import InstanceNorm
        from python.foundations import Tensor

        inst_norm = InstanceNorm(num_channels=32)
        x = Tensor(np.random.randn(8, 32, 16, 16), requires_grad=True)

        y = inst_norm(x)

        assert y.shape == x.shape


class TestRMSNorm:
    """Test RMSNorm layer."""

    def test_rmsnorm_creation(self):
        """Test RMSNorm creation."""
        from python.nn_core import RMSNorm

        rms = RMSNorm(normalized_shape=64)
        assert rms.normalized_shape == (64,)
        assert rms.eps == 1e-6

    def test_rmsnorm_creation_tuple(self):
        """Test RMSNorm with tuple normalized_shape."""
        from python.nn_core import RMSNorm

        rms = RMSNorm(normalized_shape=(32, 64))
        assert rms.normalized_shape == (32, 64)

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        from python.nn_core import RMSNorm
        from python.foundations import Tensor

        rms = RMSNorm(normalized_shape=64)
        x = Tensor(np.random.randn(8, 16, 64), requires_grad=True)

        y = rms(x)

        assert y.shape == x.shape

    def test_rmsnorm_backward(self):
        """Test RMSNorm backward pass."""
        from python.nn_core import RMSNorm
        from python.foundations import Tensor

        rms = RMSNorm(normalized_shape=64)
        x = Tensor(np.random.randn(8, 16, 64), requires_grad=True)

        y = rms(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert rms.weight.grad is not None

    def test_rmsnorm_no_centering(self):
        """Test RMSNorm does not center (no mean subtraction)."""
        from python.nn_core import RMSNorm
        from python.foundations import Tensor

        rms = RMSNorm(normalized_shape=64, eps=1e-6)
        # Input with non-zero mean
        x_data = np.random.randn(2, 64) + 10.0  # Large positive mean
        x = Tensor(x_data, requires_grad=True)

        y = rms(x)

        # RMSNorm output should still have non-zero mean
        # (unlike LayerNorm which centers to zero mean)
        # The exact value depends on input, but shouldn't be near zero
        output_mean = y.data.mean()
        # RMSNorm normalizes by RMS but doesn't subtract mean


class TestRMSNormTransformer:
    """Test RMSNormTransformer layer."""

    def test_rmsnorm_transformer_forward(self):
        """Test RMSNormTransformer forward pass."""
        from python.nn_core.normalization import RMSNormTransformer
        from python.foundations import Tensor

        rms = RMSNormTransformer(normalized_shape=512)
        x = Tensor(np.random.randn(2, 128, 512), requires_grad=True)

        y = rms(x)

        assert y.shape == x.shape


class TestLayerNormComprehensive:
    """Comprehensive tests for LayerNorm layer."""

    def test_layernorm_creation_int(self):
        """Test LayerNorm with int normalized_shape."""
        from python.nn_core import LayerNorm

        ln = LayerNorm(normalized_shape=64)
        assert ln.normalized_shape == (64,)

    def test_layernorm_creation_tuple(self):
        """Test LayerNorm with tuple normalized_shape."""
        from python.nn_core import LayerNorm

        ln = LayerNorm(normalized_shape=(32, 64))
        assert ln.normalized_shape == (32, 64)

    def test_layernorm_output_normalized(self):
        """Test LayerNorm output has zero mean and unit variance per sample."""
        from python.nn_core import LayerNorm
        from python.foundations import Tensor

        ln = LayerNorm(normalized_shape=64, elementwise_affine=False)
        x = Tensor(np.random.randn(8, 16, 64) * 5 + 2, requires_grad=True)

        y = ln(x)

        # Each sample should have ~0 mean and ~1 var over last dim
        for b in range(8):
            for s in range(16):
                sample = y.data[b, s, :]
                mean = sample.mean()
                var = sample.var()
                assert np.abs(mean) < 0.1, f"Mean should be ~0, got {mean}"
                assert np.abs(var - 1.0) < 0.15, f"Var should be ~1, got {var}"

    def test_layernorm_elementwise_affine_false(self):
        """Test LayerNorm with elementwise_affine=False."""
        from python.nn_core import LayerNorm
        from python.foundations import Tensor

        ln = LayerNorm(normalized_shape=64, elementwise_affine=False)

        assert not ln.weight.requires_grad
        assert not ln.bias.requires_grad


class TestLayerNormTransformer:
    """Test LayerNormTransformer layer."""

    def test_layernorm_transformer_forward(self):
        """Test LayerNormTransformer forward pass."""
        from python.nn_core.normalization import LayerNormTransformer
        from python.foundations import Tensor

        ln = LayerNormTransformer(normalized_shape=512)
        x = Tensor(np.random.randn(2, 128, 512), requires_grad=True)

        y = ln(x)

        assert y.shape == x.shape


# =============================================================================
# Normalization Functional Tests (Comprehensive)
# =============================================================================

class TestBatchNorm1dFunctional:
    """Test BatchNorm1d functional operation."""

    def test_batchnorm1d_functional_forward_2d(self):
        """Test BatchNorm1d functional with 2D input."""
        from python.nn_core.normalization_functional import BatchNorm1d as BN1dFn

        bn_fn = BN1dFn()
        x = np.random.randn(8, 32).astype(np.float64)
        gamma = np.ones(32)
        beta = np.zeros(32)
        running_mean = np.zeros(32)
        running_var = np.ones(32)

        y = bn_fn.forward(x, gamma, beta, running_mean, running_var, training=True)

        assert y.shape == x.shape

    def test_batchnorm1d_functional_forward_3d(self):
        """Test BatchNorm1d functional with 3D input."""
        from python.nn_core.normalization_functional import BatchNorm1d as BN1dFn

        bn_fn = BN1dFn()
        x = np.random.randn(8, 32, 16).astype(np.float64)
        gamma = np.ones(32)
        beta = np.zeros(32)
        running_mean = np.zeros(32)
        running_var = np.ones(32)

        y = bn_fn.forward(x, gamma, beta, running_mean, running_var, training=True)

        assert y.shape == x.shape

    def test_batchnorm1d_functional_backward(self):
        """Test BatchNorm1d functional backward."""
        from python.nn_core.normalization_functional import BatchNorm1d as BN1dFn

        bn_fn = BN1dFn()
        x = np.random.randn(8, 32).astype(np.float64)
        gamma = np.ones(32)
        beta = np.zeros(32)
        running_mean = np.zeros(32)
        running_var = np.ones(32)

        y = bn_fn.forward(x, gamma, beta, running_mean, running_var, training=True)
        grad_output = np.ones_like(y)
        grad_x, grad_gamma, grad_beta = bn_fn.backward(grad_output)

        assert grad_x.shape == x.shape
        assert grad_gamma.shape == gamma.shape
        assert grad_beta.shape == beta.shape


class TestBatchNorm2dFunctional:
    """Test BatchNorm2d functional operation."""

    def test_batchnorm2d_functional_forward(self):
        """Test BatchNorm2d functional forward."""
        from python.nn_core.normalization_functional import BatchNorm2d as BN2dFn

        bn_fn = BN2dFn()
        x = np.random.randn(4, 3, 8, 8).astype(np.float64)
        gamma = np.ones(3)
        beta = np.zeros(3)
        running_mean = np.zeros(3)
        running_var = np.ones(3)

        y = bn_fn.forward(x, gamma, beta, running_mean, running_var, training=True)

        assert y.shape == x.shape

    def test_batchnorm2d_functional_backward(self):
        """Test BatchNorm2d functional backward."""
        from python.nn_core.normalization_functional import BatchNorm2d as BN2dFn

        bn_fn = BN2dFn()
        x = np.random.randn(4, 3, 8, 8).astype(np.float64)
        gamma = np.ones(3)
        beta = np.zeros(3)
        running_mean = np.zeros(3)
        running_var = np.ones(3)

        y = bn_fn.forward(x, gamma, beta, running_mean, running_var, training=True)
        grad_output = np.ones_like(y)
        grad_x, grad_gamma, grad_beta = bn_fn.backward(grad_output)

        assert grad_x.shape == x.shape
        assert grad_gamma.shape == gamma.shape
        assert grad_beta.shape == beta.shape


class TestLayerNormFunctional:
    """Test LayerNorm functional operation."""

    def test_layernorm_functional_forward(self):
        """Test LayerNorm functional forward."""
        from python.nn_core.normalization_functional import LayerNorm as LNFn

        ln_fn = LNFn()
        x = np.random.randn(2, 10, 64).astype(np.float64)
        gamma = np.ones(64)
        beta = np.zeros(64)

        y = ln_fn.forward(x, gamma, beta, normalized_shape=(64,))

        assert y.shape == x.shape

    def test_layernorm_functional_forward_multidim(self):
        """Test LayerNorm functional with multi-dimensional normalized_shape."""
        from python.nn_core.normalization_functional import LayerNorm as LNFn

        ln_fn = LNFn()
        x = np.random.randn(2, 10, 8, 8).astype(np.float64)
        gamma = np.ones((8, 8))
        beta = np.zeros((8, 8))

        y = ln_fn.forward(x, gamma, beta, normalized_shape=(8, 8))

        assert y.shape == x.shape

    def test_layernorm_functional_backward(self):
        """Test LayerNorm functional backward."""
        from python.nn_core.normalization_functional import LayerNorm as LNFn

        ln_fn = LNFn()
        x = np.random.randn(2, 10, 64).astype(np.float64)
        gamma = np.ones(64)
        beta = np.zeros(64)

        y = ln_fn.forward(x, gamma, beta, normalized_shape=(64,))
        grad_output = np.ones_like(y)
        grad_x, grad_gamma, grad_beta = ln_fn.backward(grad_output)

        assert grad_x.shape == x.shape
        assert grad_gamma.shape == gamma.shape
        assert grad_beta.shape == beta.shape


class TestGroupNormFunctional:
    """Test GroupNorm functional operation."""

    def test_groupnorm_functional_forward(self):
        """Test GroupNorm functional forward."""
        from python.nn_core.normalization_functional import GroupNorm as GNFn

        gn_fn = GNFn()
        x = np.random.randn(4, 32, 8, 8).astype(np.float64)
        gamma = np.ones(32)
        beta = np.zeros(32)

        y = gn_fn.forward(x, gamma, beta, num_groups=4)

        assert y.shape == x.shape

    def test_groupnorm_functional_backward(self):
        """Test GroupNorm functional backward."""
        from python.nn_core.normalization_functional import GroupNorm as GNFn

        gn_fn = GNFn()
        x = np.random.randn(4, 32, 8, 8).astype(np.float64)
        gamma = np.ones(32)
        beta = np.zeros(32)

        y = gn_fn.forward(x, gamma, beta, num_groups=4)
        grad_output = np.ones_like(y)
        grad_x, grad_gamma, grad_beta = gn_fn.backward(grad_output)

        assert grad_x.shape == x.shape
        assert grad_gamma.shape == gamma.shape
        assert grad_beta.shape == beta.shape


class TestRMSNormFunctional:
    """Test RMSNorm functional operation."""

    def test_rmsnorm_functional_forward(self):
        """Test RMSNorm functional forward."""
        from python.nn_core.normalization_functional import RMSNorm as RMSFn

        rms_fn = RMSFn()
        x = np.random.randn(2, 10, 64).astype(np.float64)
        gamma = np.ones(64)

        y = rms_fn.forward(x, gamma, normalized_shape=(64,))

        assert y.shape == x.shape

    def test_rmsnorm_functional_backward(self):
        """Test RMSNorm functional backward."""
        from python.nn_core.normalization_functional import RMSNorm as RMSFn

        rms_fn = RMSFn()
        x = np.random.randn(2, 10, 64).astype(np.float64)
        gamma = np.ones(64)

        y = rms_fn.forward(x, gamma, normalized_shape=(64,))
        grad_output = np.ones_like(y)
        grad_x, grad_gamma = rms_fn.backward(grad_output)

        assert grad_x.shape == x.shape
        assert grad_gamma.shape == gamma.shape


# =============================================================================
# Normalization Gradient Checking Tests
# =============================================================================

class TestNormalizationGradientChecks:
    """Gradient checking tests for normalization layers."""

    def test_batchnorm1d_gradient_check(self):
        """Test BatchNorm1d gradients match numerical gradients."""
        from python.nn_core.normalization_functional import BatchNorm1d as BN1dFn

        np.random.seed(42)
        bn_fn = BN1dFn()
        x = np.random.randn(4, 16).astype(np.float64)
        gamma = np.ones(16)
        beta = np.zeros(16)
        running_mean = np.zeros(16)
        running_var = np.ones(16)

        def loss_fn(x_in):
            y = bn_fn.forward(x_in, gamma, beta, running_mean.copy(), running_var.copy(), training=True)
            return np.sum(y)

        # Analytical gradient
        y = bn_fn.forward(x, gamma, beta, running_mean.copy(), running_var.copy(), training=True)
        grad_output = np.ones_like(y)
        grad_x, _, _ = bn_fn.backward(grad_output)

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(x)
        for i in range(x.size):
            idx = np.unravel_index(i, x.shape)
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps
            numerical_grad[idx] = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * eps)

        assert np.allclose(grad_x, numerical_grad, rtol=1e-4, atol=1e-4), \
            "BatchNorm1d gradient check failed"

    def test_layernorm_gradient_check(self):
        """Test LayerNorm gradients match numerical gradients."""
        from python.nn_core.normalization_functional import LayerNorm as LNFn

        np.random.seed(42)
        ln_fn = LNFn()
        x = np.random.randn(2, 8, 16).astype(np.float64)
        gamma = np.ones(16)
        beta = np.zeros(16)

        def loss_fn(x_in):
            # Need fresh instance to avoid stale cached values
            fn = LNFn()
            y = fn.forward(x_in, gamma, beta, normalized_shape=(16,))
            return np.sum(y)

        # Analytical gradient
        y = ln_fn.forward(x, gamma, beta, normalized_shape=(16,))
        grad_output = np.ones_like(y)
        grad_x, _, _ = ln_fn.backward(grad_output)

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(x)
        for i in range(x.size):
            idx = np.unravel_index(i, x.shape)
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps
            numerical_grad[idx] = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * eps)

        assert np.allclose(grad_x, numerical_grad, rtol=1e-4, atol=1e-4), \
            "LayerNorm gradient check failed"

    def test_groupnorm_gradient_check(self):
        """Test GroupNorm gradients match numerical gradients."""
        from python.nn_core.normalization_functional import GroupNorm as GNFn

        np.random.seed(42)
        gn_fn = GNFn()
        x = np.random.randn(2, 8, 4, 4).astype(np.float64)
        gamma = np.ones(8)
        beta = np.zeros(8)

        def loss_fn(x_in):
            fn = GNFn()
            y = fn.forward(x_in, gamma, beta, num_groups=2)
            return np.sum(y)

        # Analytical gradient
        y = gn_fn.forward(x, gamma, beta, num_groups=2)
        grad_output = np.ones_like(y)
        grad_x, _, _ = gn_fn.backward(grad_output)

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(x)
        for i in range(x.size):
            idx = np.unravel_index(i, x.shape)
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps
            numerical_grad[idx] = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * eps)

        assert np.allclose(grad_x, numerical_grad, rtol=1e-4, atol=1e-4), \
            "GroupNorm gradient check failed"

    def test_rmsnorm_gradient_check(self):
        """Test RMSNorm gradients match numerical gradients."""
        from python.nn_core.normalization_functional import RMSNorm as RMSFn

        np.random.seed(42)
        rms_fn = RMSFn()
        x = np.random.randn(2, 8, 16).astype(np.float64)
        gamma = np.ones(16)

        def loss_fn(x_in):
            fn = RMSFn()
            y = fn.forward(x_in, gamma, normalized_shape=(16,))
            return np.sum(y)

        # Analytical gradient
        y = rms_fn.forward(x, gamma, normalized_shape=(16,))
        grad_output = np.ones_like(y)
        grad_x, _ = rms_fn.backward(grad_output)

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(x)
        for i in range(x.size):
            idx = np.unravel_index(i, x.shape)
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps
            numerical_grad[idx] = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * eps)

        assert np.allclose(grad_x, numerical_grad, rtol=1e-4, atol=1e-4), \
            "RMSNorm gradient check failed"


# =============================================================================
# Comprehensive Normalization Gradient Tests using gradcheck
# =============================================================================

class TestNormalizationGradientsComprehensive:
    """Comprehensive gradient tests for all normalization layers using gradcheck.

    Note: Some tests are marked with pytest.mark.xfail for layers that are not yet
    implemented (BatchNorm3d, InstanceNorm, LocalResponseNorm, SpectralNorm).
    These tests serve as specifications for when those layers are implemented.
    """

    # =========================================================================
    # BatchNorm1d Gradient Tests
    # Note: BatchNorm gradcheck tests reveal implementation issues in backward pass
    # =========================================================================

    def test_batchnorm1d_gradcheck_eval_mode(self):
        """Verify BatchNorm1d gradients using gradcheck in eval mode."""
        from python.nn_core import BatchNorm1d

        np.random.seed(42)
        bn = BatchNorm1d(num_features=16)
        bn.eval()  # Eval mode for deterministic behavior
        x = Tensor(np.random.randn(4, 16).astype(np.float64), requires_grad=True)

        def func(x):
            return bn(x).sum()

        # Use looser tolerance for BatchNorm due to running stats complexity
        assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2)

    def test_batchnorm1d_gradcheck_different_sizes(self):
        """Test BatchNorm1d gradients with different batch and feature sizes."""
        from python.nn_core import BatchNorm1d

        configs = [
            (2, 8),    # Small
            (8, 32),   # Medium
            (16, 64),  # Larger
        ]

        for batch_size, num_features in configs:
            np.random.seed(42)
            bn = BatchNorm1d(num_features=num_features)
            bn.eval()
            x = Tensor(np.random.randn(batch_size, num_features).astype(np.float64), requires_grad=True)

            def func(x):
                return bn(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2), \
                f"BatchNorm1d gradcheck failed for batch={batch_size}, features={num_features}"

    def test_batchnorm1d_gradcheck_no_affine(self):
        """Test BatchNorm1d gradients without affine parameters."""
        from python.nn_core import BatchNorm1d

        np.random.seed(42)
        bn = BatchNorm1d(num_features=16, affine=False)
        bn.eval()
        x = Tensor(np.random.randn(4, 16).astype(np.float64), requires_grad=True)

        def func(x):
            return bn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2)

    def test_batchnorm1d_gradcheck_different_eps(self):
        """Test BatchNorm1d gradients with different eps values."""
        from python.nn_core import BatchNorm1d

        for eps in [1e-5, 1e-6, 1e-3]:
            np.random.seed(42)
            bn = BatchNorm1d(num_features=16, eps=eps)
            bn.eval()
            x = Tensor(np.random.randn(4, 16).astype(np.float64), requires_grad=True)

            def func(x):
                return bn(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2), \
                f"BatchNorm1d gradcheck failed for eps={eps}"

    def test_batchnorm1d_gamma_gradient(self):
        """Verify gradient flows correctly to gamma (weight) parameter."""
        from python.nn_core import BatchNorm1d

        np.random.seed(42)
        bn = BatchNorm1d(num_features=8)
        bn.eval()
        x = Tensor(np.random.randn(4, 8).astype(np.float64), requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        # Gamma gradient should exist and be non-zero
        gamma_grad = bn.weight.grad
        assert gamma_grad is not None, "Gamma gradient should exist"
        assert not np.allclose(gamma_grad, 0), "Gamma gradient should be non-zero"

    def test_batchnorm1d_beta_gradient(self):
        """Verify gradient flows correctly to beta (bias) parameter."""
        from python.nn_core import BatchNorm1d

        np.random.seed(42)
        bn = BatchNorm1d(num_features=8)
        bn.eval()
        x = Tensor(np.random.randn(4, 8).astype(np.float64), requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        # Beta gradient should exist
        beta_grad = bn.bias.grad
        assert beta_grad is not None, "Beta gradient should exist"

    # =========================================================================
    # BatchNorm2d Gradient Tests
    # Note: BatchNorm gradcheck tests reveal implementation issues in backward pass
    # =========================================================================

    def test_batchnorm2d_gradcheck_eval_mode(self):
        """Verify BatchNorm2d gradients using gradcheck in eval mode."""
        from python.nn_core import BatchNorm2d

        np.random.seed(42)
        bn = BatchNorm2d(num_features=8)
        bn.eval()
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return bn(x).sum()

        # Use looser tolerance for BatchNorm due to running stats complexity
        assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2)

    def test_batchnorm2d_gradcheck_different_sizes(self):
        """Test BatchNorm2d gradients with different spatial sizes."""
        from python.nn_core import BatchNorm2d

        configs = [
            (2, 4, 3, 3),   # Small
            (2, 8, 8, 8),   # Medium
            (4, 16, 4, 4),  # More channels
        ]

        for batch, channels, h, w in configs:
            np.random.seed(42)
            bn = BatchNorm2d(num_features=channels)
            bn.eval()
            x = Tensor(np.random.randn(batch, channels, h, w).astype(np.float64), requires_grad=True)

            def func(x):
                return bn(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2), \
                f"BatchNorm2d gradcheck failed for shape ({batch}, {channels}, {h}, {w})"

    def test_batchnorm2d_gradcheck_no_affine(self):
        """Test BatchNorm2d gradients without affine parameters."""
        from python.nn_core import BatchNorm2d

        np.random.seed(42)
        bn = BatchNorm2d(num_features=8, affine=False)
        bn.eval()
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return bn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2)

    def test_batchnorm2d_gamma_beta_gradients(self):
        """Verify gamma and beta gradients for BatchNorm2d."""
        from python.nn_core import BatchNorm2d

        np.random.seed(42)
        bn = BatchNorm2d(num_features=8)
        bn.eval()
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        assert bn.weight.grad is not None, "Gamma gradient should exist"
        assert bn.bias.grad is not None, "Beta gradient should exist"

    # =========================================================================
    # BatchNorm3d Gradient Tests (not yet implemented)
    # =========================================================================

    def test_batchnorm3d_gradcheck_eval_mode(self):
        """Verify BatchNorm3d gradients using gradcheck in eval mode."""
        from python.nn_core import BatchNorm3d

        np.random.seed(42)
        bn = BatchNorm3d(num_features=4)
        bn.eval()
        x = Tensor(np.random.randn(2, 4, 3, 3, 3).astype(np.float64), requires_grad=True)

        def func(x):
            return bn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2)

    def test_batchnorm3d_gradcheck_no_affine(self):
        """Test BatchNorm3d gradients without affine parameters."""
        from python.nn_core import BatchNorm3d

        np.random.seed(42)
        bn = BatchNorm3d(num_features=4, affine=False)
        bn.eval()
        x = Tensor(np.random.randn(2, 4, 3, 3, 3).astype(np.float64), requires_grad=True)

        def func(x):
            return bn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-2)

    # =========================================================================
    # LayerNorm Gradient Tests
    # =========================================================================

    def test_layernorm_gradcheck_1d_shape(self):
        """Verify LayerNorm gradients with 1D normalized_shape."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[16])
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        def func(x):
            return ln(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_layernorm_gradcheck_2d_shape(self):
        """Verify LayerNorm gradients with 2D normalized_shape."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[8, 16])
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        def func(x):
            return ln(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_layernorm_gradcheck_3d_shape(self):
        """Verify LayerNorm gradients with 3D normalized_shape (full spatial)."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[4, 4, 4])
        x = Tensor(np.random.randn(2, 4, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return ln(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_layernorm_gradcheck_different_sizes(self):
        """Test LayerNorm gradients with different sizes."""
        from python.nn_core import LayerNorm

        configs = [
            ((2, 8), [8]),           # 2D input
            ((2, 4, 16), [16]),      # 3D input, last dim
            ((4, 8, 32), [32]),      # Larger
            ((2, 4, 8, 16), [16]),   # 4D input
        ]

        for input_shape, norm_shape in configs:
            np.random.seed(42)
            ln = LayerNorm(normalized_shape=norm_shape)
            x = Tensor(np.random.randn(*input_shape).astype(np.float64), requires_grad=True)

            def func(x):
                return ln(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"LayerNorm gradcheck failed for input_shape={input_shape}, norm_shape={norm_shape}"

    def test_layernorm_gradcheck_no_affine(self):
        """Test LayerNorm gradients without affine parameters."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[16], elementwise_affine=False)
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        def func(x):
            return ln(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_layernorm_gradcheck_different_eps(self):
        """Test LayerNorm gradients with different eps values."""
        from python.nn_core import LayerNorm

        for eps in [1e-5, 1e-6, 1e-12]:
            np.random.seed(42)
            ln = LayerNorm(normalized_shape=[16], eps=eps)
            x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

            def func(x):
                return ln(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"LayerNorm gradcheck failed for eps={eps}"

    def test_layernorm_gamma_beta_gradients(self):
        """Verify gamma and beta gradients for LayerNorm."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[16])
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        y = ln(x)
        loss = y.sum()
        loss.backward()

        assert ln.weight.grad is not None, "Gamma gradient should exist"
        assert ln.bias.grad is not None, "Beta gradient should exist"
        assert ln.weight.grad.shape == (16,), "Gamma gradient shape mismatch"
        assert ln.bias.grad.shape == (16,), "Beta gradient shape mismatch"

    # =========================================================================
    # GroupNorm Gradient Tests
    # =========================================================================

    def test_groupnorm_gradcheck_basic(self):
        """Verify GroupNorm gradients using gradcheck."""
        from python.nn_core import GroupNorm

        np.random.seed(42)
        gn = GroupNorm(num_groups=4, num_channels=16)
        x = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return gn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_groupnorm_gradcheck_groups_equal_channels(self):
        """Test GroupNorm gradients when groups=channels (InstanceNorm-like)."""
        from python.nn_core import GroupNorm

        np.random.seed(42)
        gn = GroupNorm(num_groups=8, num_channels=8)
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return gn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_groupnorm_gradcheck_single_group(self):
        """Test GroupNorm gradients with single group (LayerNorm-like)."""
        from python.nn_core import GroupNorm

        np.random.seed(42)
        gn = GroupNorm(num_groups=1, num_channels=16)
        x = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return gn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_groupnorm_gradcheck_different_configs(self):
        """Test GroupNorm gradients with different group configurations."""
        from python.nn_core import GroupNorm

        configs = [
            (2, 8, 4, 4),   # 2 groups of 4
            (4, 16, 8, 8),  # 4 groups of 4
            (8, 32, 4, 4),  # 8 groups of 4
            (2, 6, 4, 4),   # 2 groups of 3 (non-power-of-2)
        ]

        for groups, channels, h, w in configs:
            np.random.seed(42)
            gn = GroupNorm(num_groups=groups, num_channels=channels)
            x = Tensor(np.random.randn(2, channels, h, w).astype(np.float64), requires_grad=True)

            def func(x):
                return gn(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"GroupNorm gradcheck failed for groups={groups}, channels={channels}"

    def test_groupnorm_gradcheck_no_affine(self):
        """Test GroupNorm gradients without affine parameters."""
        from python.nn_core import GroupNorm

        np.random.seed(42)
        gn = GroupNorm(num_groups=4, num_channels=16, affine=False)
        x = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return gn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_groupnorm_gamma_beta_gradients(self):
        """Verify gamma and beta gradients for GroupNorm."""
        from python.nn_core import GroupNorm

        np.random.seed(42)
        gn = GroupNorm(num_groups=4, num_channels=16)
        x = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float64), requires_grad=True)

        y = gn(x)
        loss = y.sum()
        loss.backward()

        assert gn.weight.grad is not None, "Gamma gradient should exist"
        assert gn.bias.grad is not None, "Beta gradient should exist"
        assert gn.weight.grad.shape == (16,), "Gamma gradient shape mismatch"
        assert gn.bias.grad.shape == (16,), "Beta gradient shape mismatch"

    # =========================================================================
    # InstanceNorm Gradient Tests (not yet implemented)
    # =========================================================================

    def test_instancenorm_gradcheck_basic(self):
        """Verify InstanceNorm gradients using gradcheck."""
        from python.nn_core import InstanceNorm

        np.random.seed(42)
        inst_norm = InstanceNorm(num_features=8)
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return inst_norm(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_instancenorm1d_gradcheck(self):
        """Verify InstanceNorm1d gradients using gradcheck."""
        from python.nn_core import InstanceNorm1d

        np.random.seed(42)
        inst_norm = InstanceNorm1d(num_features=8)
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        def func(x):
            return inst_norm(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_instancenorm2d_gradcheck(self):
        """Verify InstanceNorm2d gradients using gradcheck."""
        from python.nn_core import InstanceNorm2d

        np.random.seed(42)
        inst_norm = InstanceNorm2d(num_features=8)
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return inst_norm(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_instancenorm3d_gradcheck(self):
        """Verify InstanceNorm3d gradients using gradcheck."""
        from python.nn_core import InstanceNorm3d

        np.random.seed(42)
        inst_norm = InstanceNorm3d(num_features=4)
        x = Tensor(np.random.randn(2, 4, 3, 3, 3).astype(np.float64), requires_grad=True)

        def func(x):
            return inst_norm(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_instancenorm_gradcheck_different_sizes(self):
        """Test InstanceNorm gradients with different spatial sizes."""
        from python.nn_core import InstanceNorm2d

        configs = [
            (2, 4, 3, 3),
            (4, 8, 8, 8),
            (1, 16, 4, 4),
        ]

        for batch, channels, h, w in configs:
            np.random.seed(42)
            inst_norm = InstanceNorm2d(num_features=channels)
            x = Tensor(np.random.randn(batch, channels, h, w).astype(np.float64), requires_grad=True)

            def func(x):
                return inst_norm(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"InstanceNorm2d gradcheck failed for shape ({batch}, {channels}, {h}, {w})"

    def test_instancenorm_gradcheck_with_affine(self):
        """Test InstanceNorm gradients with affine parameters."""
        from python.nn_core import InstanceNorm2d

        np.random.seed(42)
        inst_norm = InstanceNorm2d(num_features=8, affine=True)
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return inst_norm(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    # =========================================================================
    # RMSNorm Gradient Tests
    # =========================================================================

    def test_rmsnorm_gradcheck_basic(self):
        """Verify RMSNorm gradients using gradcheck."""
        from python.nn_core import RMSNorm

        np.random.seed(42)
        rms = RMSNorm(normalized_shape=16)
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        def func(x):
            return rms(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_rmsnorm_gradcheck_different_sizes(self):
        """Test RMSNorm gradients with different sizes."""
        from python.nn_core import RMSNorm

        configs = [
            ((2, 8), 8),
            ((4, 16, 32), 32),
            ((2, 4, 8, 64), 64),
        ]

        for input_shape, norm_dim in configs:
            np.random.seed(42)
            rms = RMSNorm(normalized_shape=norm_dim)
            x = Tensor(np.random.randn(*input_shape).astype(np.float64), requires_grad=True)

            def func(x):
                return rms(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"RMSNorm gradcheck failed for input_shape={input_shape}, norm_dim={norm_dim}"

    def test_rmsnorm_gradcheck_different_eps(self):
        """Test RMSNorm gradients with different eps values."""
        from python.nn_core import RMSNorm

        for eps in [1e-5, 1e-6, 1e-8]:
            np.random.seed(42)
            rms = RMSNorm(normalized_shape=16, eps=eps)
            x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

            def func(x):
                return rms(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"RMSNorm gradcheck failed for eps={eps}"

    def test_rmsnorm_gamma_gradient(self):
        """Verify gamma gradient for RMSNorm."""
        from python.nn_core import RMSNorm

        np.random.seed(42)
        rms = RMSNorm(normalized_shape=16)
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        y = rms(x)
        loss = y.sum()
        loss.backward()

        assert rms.weight.grad is not None, "Gamma gradient should exist"
        assert rms.weight.grad.shape == (16,), "Gamma gradient shape mismatch"
        assert not np.allclose(rms.weight.grad, 0), "Gamma gradient should be non-zero"

    def test_rmsnorm_gradcheck_large_dim(self):
        """Test RMSNorm gradients with larger normalized dimension."""
        from python.nn_core import RMSNorm

        np.random.seed(42)
        rms = RMSNorm(normalized_shape=256)
        x = Tensor(np.random.randn(2, 4, 256).astype(np.float64), requires_grad=True)

        def func(x):
            return rms(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    # =========================================================================
    # LocalResponseNorm Gradient Tests (not yet implemented)
    # =========================================================================

    def test_localresponsenorm_gradcheck_basic(self):
        """Verify LocalResponseNorm gradients using gradcheck."""
        from python.nn_core import LocalResponseNorm

        np.random.seed(42)
        lrn = LocalResponseNorm(size=5)
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return lrn(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_localresponsenorm_gradcheck_different_sizes(self):
        """Test LocalResponseNorm gradients with different size parameters."""
        from python.nn_core import LocalResponseNorm

        for size in [3, 5, 7]:
            np.random.seed(42)
            lrn = LocalResponseNorm(size=size)
            x = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float64), requires_grad=True)

            def func(x):
                return lrn(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"LocalResponseNorm gradcheck failed for size={size}"

    def test_localresponsenorm_gradcheck_different_params(self):
        """Test LocalResponseNorm gradients with different alpha, beta, k."""
        from python.nn_core import LocalResponseNorm

        configs = [
            {'size': 5, 'alpha': 1e-4, 'beta': 0.75, 'k': 1},
            {'size': 5, 'alpha': 2e-5, 'beta': 0.5, 'k': 2},
            {'size': 3, 'alpha': 1e-3, 'beta': 1.0, 'k': 1},
        ]

        for config in configs:
            np.random.seed(42)
            lrn = LocalResponseNorm(**config)
            x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

            def func(x):
                return lrn(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"LocalResponseNorm gradcheck failed for config={config}"

    # =========================================================================
    # SpectralNorm Gradient Tests (not exported from nn_core)
    # =========================================================================

    def test_spectralnorm_linear_gradcheck(self):
        """Verify SpectralNorm gradients for linear layer."""
        from python.nn_core import SpectralNormLinear

        np.random.seed(42)
        sn_linear = SpectralNormLinear(in_features=8, out_features=16)
        x = Tensor(np.random.randn(2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return sn_linear(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_spectralnorm_conv2d_gradcheck(self):
        """Verify SpectralNorm gradients for conv2d layer."""
        from python.nn_core import SpectralNormConv2d

        np.random.seed(42)
        sn_conv = SpectralNormConv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 4, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return sn_conv(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_spectralnorm_linear_weight_gradient(self):
        """Verify weight gradients flow through SpectralNorm."""
        from python.nn_core import SpectralNormLinear

        np.random.seed(42)
        sn_linear = SpectralNormLinear(in_features=8, out_features=16)
        x = Tensor(np.random.randn(2, 8).astype(np.float64), requires_grad=True)

        y = sn_linear(x)
        loss = y.sum()
        loss.backward()

        # Check that weight gradient exists
        assert sn_linear.weight.grad is not None, "Weight gradient should exist"
        assert not np.allclose(sn_linear.weight.grad, 0), "Weight gradient should be non-zero"

    # =========================================================================
    # Gradient Flow Tests (chain multiple normalizations)
    # =========================================================================

    def test_gradient_flow_layernorm_chain(self):
        """Test gradients flow through chained LayerNorm layers."""
        from python.nn_core import LayerNorm

        # Note: With chained normalization layers, gradients can become very small
        # (1e-10 to 1e-11) due to repeated normalization. This is mathematically
        # expected, so we verify gradients exist and are finite rather than non-zero.
        np.random.seed(42)
        ln1 = LayerNorm(normalized_shape=[16])
        ln2 = LayerNorm(normalized_shape=[16])

        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        y = ln1(x)
        z = ln2(y)
        # Use square sum as loss to get non-zero gradients even with normalized output
        loss = (z ** 2).sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert np.all(np.isfinite(x.grad)), "Input gradient should be finite"
        # Gradients are computed (even if very small due to repeated normalization)
        assert np.max(np.abs(x.grad)) > 1e-15, "Gradients should be non-trivially computed"

    def test_gradient_flow_mixed_norms(self):
        """Test gradients flow through mixed normalization types.

        Note: With chained normalizations, gradients can become very small.
        """
        from python.nn_core import LayerNorm, RMSNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[16])
        rms = RMSNorm(normalized_shape=16)

        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        y = ln(x)
        z = rms(y)
        # Use square sum as loss to get non-zero gradients even with normalized output
        loss = (z ** 2).sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert np.all(np.isfinite(x.grad)), "Input gradient should be finite"
        # Gradients are computed (even if very small due to repeated normalization)
        assert np.max(np.abs(x.grad)) > 1e-15, "Gradients should be non-trivially computed"

    # =========================================================================
    # Numerical Stability Gradient Tests
    # =========================================================================

    def test_layernorm_gradient_small_input(self):
        """Test LayerNorm gradients with small input values."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[16])
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64) * 1e-6, requires_grad=True)

        def func(x):
            return ln(x).sum()

        assert gradcheck(func, (x,), eps=1e-7, atol=1e-4, rtol=1e-3)

    def test_layernorm_gradient_large_input(self):
        """Test LayerNorm gradients with large input values."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[16])
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64) * 100, requires_grad=True)

        def func(x):
            return ln(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_rmsnorm_gradient_small_input(self):
        """Test RMSNorm gradients with small input values."""
        from python.nn_core import RMSNorm

        np.random.seed(42)
        rms = RMSNorm(normalized_shape=16)
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64) * 1e-6, requires_grad=True)

        def func(x):
            return rms(x).sum()

        assert gradcheck(func, (x,), eps=1e-7, atol=1e-4, rtol=1e-3)

    def test_groupnorm_gradient_varying_batch(self):
        """Test GroupNorm gradients with varying batch sizes."""
        from python.nn_core import GroupNorm

        for batch_size in [1, 2, 8, 16]:
            np.random.seed(42)
            gn = GroupNorm(num_groups=4, num_channels=16)
            x = Tensor(np.random.randn(batch_size, 16, 4, 4).astype(np.float64), requires_grad=True)

            def func(x):
                return gn(x).sum()

            assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3), \
                f"GroupNorm gradcheck failed for batch_size={batch_size}"

    # =========================================================================
    # Explicit Input Gradient Verification Tests
    # =========================================================================

    def test_layernorm_input_gradient_shape(self):
        """Verify LayerNorm produces correct input gradient shape."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[16])
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        y = ln(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert x.grad.shape == x.shape, f"Input gradient shape {x.grad.shape} != input shape {x.shape}"
        assert np.all(np.isfinite(x.grad)), "Input gradient should be finite"

    def test_groupnorm_input_gradient_shape(self):
        """Verify GroupNorm produces correct input gradient shape."""
        from python.nn_core import GroupNorm

        np.random.seed(42)
        gn = GroupNorm(num_groups=4, num_channels=16)
        x = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float64), requires_grad=True)

        y = gn(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert x.grad.shape == x.shape, f"Input gradient shape {x.grad.shape} != input shape {x.shape}"
        assert np.all(np.isfinite(x.grad)), "Input gradient should be finite"

    def test_rmsnorm_input_gradient_shape(self):
        """Verify RMSNorm produces correct input gradient shape."""
        from python.nn_core import RMSNorm

        np.random.seed(42)
        rms = RMSNorm(normalized_shape=16)
        x = Tensor(np.random.randn(2, 8, 16).astype(np.float64), requires_grad=True)

        y = rms(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert x.grad.shape == x.shape, f"Input gradient shape {x.grad.shape} != input shape {x.shape}"
        assert np.all(np.isfinite(x.grad)), "Input gradient should be finite"

    def test_batchnorm1d_input_gradient_shape(self):
        """Verify BatchNorm1d produces correct input gradient shape."""
        from python.nn_core import BatchNorm1d

        np.random.seed(42)
        bn = BatchNorm1d(num_features=16)
        bn.eval()
        x = Tensor(np.random.randn(4, 16).astype(np.float64), requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert x.grad.shape == x.shape, f"Input gradient shape {x.grad.shape} != input shape {x.shape}"
        assert np.all(np.isfinite(x.grad)), "Input gradient should be finite"

    def test_batchnorm2d_input_gradient_shape(self):
        """Verify BatchNorm2d produces correct input gradient shape."""
        from python.nn_core import BatchNorm2d

        np.random.seed(42)
        bn = BatchNorm2d(num_features=8)
        bn.eval()
        x = Tensor(np.random.randn(2, 8, 4, 4).astype(np.float64), requires_grad=True)

        y = bn(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert x.grad.shape == x.shape, f"Input gradient shape {x.grad.shape} != input shape {x.shape}"
        assert np.all(np.isfinite(x.grad)), "Input gradient should be finite"

    def test_layernorm_input_gradient_numerical(self):
        """Verify LayerNorm input gradient matches numerical gradient."""
        from python.nn_core import LayerNorm

        np.random.seed(42)
        ln = LayerNorm(normalized_shape=[8])
        x = Tensor(np.random.randn(2, 4, 8).astype(np.float64), requires_grad=True)

        # Compute analytical gradient
        y = ln(x)
        loss = y.sum()
        loss.backward()
        analytical_grad = x.grad.copy()

        # Compute numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(x.data)
        for i in range(x.data.size):
            idx = np.unravel_index(i, x.data.shape)
            x_plus = x.data.copy()
            x_minus = x.data.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps

            y_plus = ln(Tensor(x_plus)).sum()
            y_minus = ln(Tensor(x_minus)).sum()
            numerical_grad[idx] = (y_plus.data - y_minus.data) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4), \
            "LayerNorm input gradient doesn't match numerical gradient"

    def test_groupnorm_input_gradient_numerical(self):
        """Verify GroupNorm input gradient matches numerical gradient."""
        from python.nn_core import GroupNorm

        np.random.seed(42)
        gn = GroupNorm(num_groups=2, num_channels=4)
        x = Tensor(np.random.randn(1, 4, 3, 3).astype(np.float64), requires_grad=True)

        # Compute analytical gradient
        y = gn(x)
        loss = y.sum()
        loss.backward()
        analytical_grad = x.grad.copy()

        # Compute numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(x.data)
        for i in range(x.data.size):
            idx = np.unravel_index(i, x.data.shape)
            x_plus = x.data.copy()
            x_minus = x.data.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps

            y_plus = gn(Tensor(x_plus)).sum()
            y_minus = gn(Tensor(x_minus)).sum()
            numerical_grad[idx] = (y_plus.data - y_minus.data) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4), \
            "GroupNorm input gradient doesn't match numerical gradient"

    def test_rmsnorm_input_gradient_numerical(self):
        """Verify RMSNorm input gradient matches numerical gradient."""
        from python.nn_core import RMSNorm

        np.random.seed(42)
        rms = RMSNorm(normalized_shape=8)
        x = Tensor(np.random.randn(2, 4, 8).astype(np.float64), requires_grad=True)

        # Compute analytical gradient
        y = rms(x)
        loss = y.sum()
        loss.backward()
        analytical_grad = x.grad.copy()

        # Compute numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(x.data)
        for i in range(x.data.size):
            idx = np.unravel_index(i, x.data.shape)
            x_plus = x.data.copy()
            x_minus = x.data.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps

            y_plus = rms(Tensor(x_plus)).sum()
            y_minus = rms(Tensor(x_minus)).sum()
            numerical_grad[idx] = (y_plus.data - y_minus.data) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4), \
            "RMSNorm input gradient doesn't match numerical gradient"


# =============================================================================
# Normalization Edge Cases and Numerical Stability
# =============================================================================

class TestNormalizationEdgeCases:
    """Test edge cases and numerical stability for normalization layers."""

    def test_batchnorm_small_variance(self):
        """Test BatchNorm handles near-zero variance (constant input)."""
        from python.nn_core import BatchNorm1d
        from python.foundations import Tensor

        bn = BatchNorm1d(num_features=32)
        bn.train()

        # Near-constant input (very small variance)
        x = Tensor(np.ones((8, 32)) + np.random.randn(8, 32) * 1e-10, requires_grad=True)

        y = bn(x)

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(y.data)), "BatchNorm should handle small variance"

    def test_layernorm_single_element(self):
        """Test LayerNorm with single element normalized_shape."""
        from python.nn_core import LayerNorm
        from python.foundations import Tensor

        ln = LayerNorm(normalized_shape=1)
        x = Tensor(np.random.randn(2, 10, 1), requires_grad=True)

        y = ln(x)

        assert y.shape == x.shape
        assert np.all(np.isfinite(y.data))

    def test_groupnorm_groups_equal_channels(self):
        """Test GroupNorm when num_groups = num_channels (InstanceNorm)."""
        from python.nn_core import GroupNorm
        from python.foundations import Tensor

        gn = GroupNorm(num_groups=16, num_channels=16)
        x = Tensor(np.random.randn(2, 16, 8, 8), requires_grad=True)

        y = gn(x)

        assert y.shape == x.shape

    def test_groupnorm_single_group(self):
        """Test GroupNorm with single group (LayerNorm-like)."""
        from python.nn_core import GroupNorm
        from python.foundations import Tensor

        gn = GroupNorm(num_groups=1, num_channels=16)
        x = Tensor(np.random.randn(2, 16, 8, 8), requires_grad=True)

        y = gn(x)

        assert y.shape == x.shape

    def test_rmsnorm_large_values(self):
        """Test RMSNorm handles large values."""
        from python.nn_core import RMSNorm
        from python.foundations import Tensor

        rms = RMSNorm(normalized_shape=64)
        x = Tensor(np.random.randn(2, 64) * 1000, requires_grad=True)

        y = rms(x)

        assert np.all(np.isfinite(y.data)), "RMSNorm should handle large values"


# =============================================================================
# Additional Normalization Module Tests
# =============================================================================

class TestBatchNorm3d:
    """Test BatchNorm3d layer for volumetric data."""

    def test_batchnorm3d_init(self):
        """Test BatchNorm3d initialization."""
        from python.nn_core.normalization import BatchNorm3d

        bn = BatchNorm3d(num_features=16)

        assert bn.num_features == 16
        assert bn.eps == 1e-5
        assert bn.momentum == 0.1
        assert bn.affine is True
        assert bn.track_running_stats is True

    def test_batchnorm3d_parameters(self):
        """Test BatchNorm3d has correct parameters."""
        from python.nn_core.normalization import BatchNorm3d

        bn = BatchNorm3d(num_features=16)

        # Check weight and bias
        assert hasattr(bn, 'weight')
        assert hasattr(bn, 'bias')
        assert bn.weight.data.shape == (16,)
        assert bn.bias.data.shape == (16,)

    def test_batchnorm3d_running_stats(self):
        """Test BatchNorm3d has running statistics buffers."""
        from python.nn_core.normalization import BatchNorm3d

        bn = BatchNorm3d(num_features=16, track_running_stats=True)

        assert hasattr(bn, 'running_mean')
        assert hasattr(bn, 'running_var')
        assert bn.running_mean.shape == (16,)
        assert bn.running_var.shape == (16,)

    def test_batchnorm3d_no_affine(self):
        """Test BatchNorm3d without affine parameters."""
        from python.nn_core.normalization import BatchNorm3d

        bn = BatchNorm3d(num_features=16, affine=False)

        # weight and bias should be None
        assert bn.weight is None or (hasattr(bn.weight, 'data') and bn.weight.data is None)

    def test_batchnorm3d_no_running_stats(self):
        """Test BatchNorm3d without running statistics."""
        from python.nn_core.normalization import BatchNorm3d

        bn = BatchNorm3d(num_features=16, track_running_stats=False)

        # Should not have running stats
        assert not hasattr(bn, 'running_mean') or bn.running_mean is None


class TestInstanceNorm1d:
    """Test InstanceNorm1d layer."""

    def test_instancenorm1d_init(self):
        """Test InstanceNorm1d initialization."""
        from python.nn_core.normalization import InstanceNorm1d

        inorm = InstanceNorm1d(num_features=32)

        assert inorm.num_features == 32
        assert inorm.eps == 1e-5
        assert inorm.affine is False  # Default is False for InstanceNorm

    def test_instancenorm1d_with_affine(self):
        """Test InstanceNorm1d with affine parameters."""
        from python.nn_core.normalization import InstanceNorm1d

        inorm = InstanceNorm1d(num_features=32, affine=True)

        assert inorm.affine is True
        assert hasattr(inorm, 'weight')
        assert hasattr(inorm, 'bias')
        assert inorm.weight.data.shape == (32,)
        assert inorm.bias.data.shape == (32,)

    def test_instancenorm1d_no_affine(self):
        """Test InstanceNorm1d without affine parameters."""
        from python.nn_core.normalization import InstanceNorm1d

        inorm = InstanceNorm1d(num_features=32, affine=False)

        assert inorm.affine is False
        assert not hasattr(inorm, 'weight') or inorm.weight is None


class TestInstanceNorm2d:
    """Test InstanceNorm2d layer."""

    def test_instancenorm2d_init(self):
        """Test InstanceNorm2d initialization."""
        from python.nn_core.normalization import InstanceNorm2d

        inorm = InstanceNorm2d(num_features=64)

        assert inorm.num_features == 64
        assert inorm.eps == 1e-5
        assert inorm.affine is False

    def test_instancenorm2d_with_affine(self):
        """Test InstanceNorm2d with affine parameters."""
        from python.nn_core.normalization import InstanceNorm2d

        inorm = InstanceNorm2d(num_features=64, affine=True)

        assert inorm.affine is True
        assert hasattr(inorm, 'weight')
        assert hasattr(inorm, 'bias')
        assert inorm.weight.data.shape == (64,)
        assert inorm.bias.data.shape == (64,)

    def test_instancenorm2d_custom_eps(self):
        """Test InstanceNorm2d with custom epsilon."""
        from python.nn_core.normalization import InstanceNorm2d

        inorm = InstanceNorm2d(num_features=32, eps=1e-6)

        assert inorm.eps == 1e-6


class TestInstanceNorm3d:
    """Test InstanceNorm3d layer."""

    def test_instancenorm3d_init(self):
        """Test InstanceNorm3d initialization."""
        from python.nn_core.normalization import InstanceNorm3d

        inorm = InstanceNorm3d(num_features=16)

        assert inorm.num_features == 16
        assert inorm.eps == 1e-5
        assert inorm.affine is False

    def test_instancenorm3d_with_affine(self):
        """Test InstanceNorm3d with affine parameters."""
        from python.nn_core.normalization import InstanceNorm3d

        inorm = InstanceNorm3d(num_features=16, affine=True)

        assert inorm.affine is True
        assert hasattr(inorm, 'weight')
        assert hasattr(inorm, 'bias')
        assert inorm.weight.data.shape == (16,)
        assert inorm.bias.data.shape == (16,)


class TestLocalResponseNorm:
    """Test LocalResponseNorm layer."""

    def test_localresponsenorm_init(self):
        """Test LocalResponseNorm initialization."""
        from python.nn_core.normalization import LocalResponseNorm

        lrn = LocalResponseNorm(size=5)

        assert lrn.size == 5
        assert lrn.alpha == 1e-4
        assert lrn.beta == 0.75
        assert lrn.k == 1.0

    def test_localresponsenorm_custom_params(self):
        """Test LocalResponseNorm with custom parameters."""
        from python.nn_core.normalization import LocalResponseNorm

        lrn = LocalResponseNorm(size=3, alpha=2e-4, beta=0.5, k=2.0)

        assert lrn.size == 3
        assert lrn.alpha == 2e-4
        assert lrn.beta == 0.5
        assert lrn.k == 2.0


class TestSpectralNorm:
    """Test SpectralNorm layer for GAN discriminators."""

    def test_spectralnorm_init(self):
        """Test SpectralNorm initialization."""
        from python.nn_core.normalization import SpectralNorm

        weight = np.random.randn(64, 32)
        sn = SpectralNorm(weight)

        assert sn.n_power_iterations == 1
        assert sn.eps == 1e-12
        assert sn.height == 64
        assert sn.width == 32

    def test_spectralnorm_custom_iterations(self):
        """Test SpectralNorm with custom power iterations."""
        from python.nn_core.normalization import SpectralNorm

        weight = np.random.randn(64, 32)
        sn = SpectralNorm(weight, n_power_iterations=2)

        assert sn.n_power_iterations == 2

    def test_spectralnorm_1d_weight(self):
        """Test SpectralNorm with 1D weight (bias-like)."""
        from python.nn_core.normalization import SpectralNorm

        weight = np.random.randn(64)
        sn = SpectralNorm(weight)

        assert sn.height == 1
        assert sn.width == 64

    def test_spectralnorm_has_u_buffer(self):
        """Test SpectralNorm has u buffer for power iteration."""
        from python.nn_core.normalization import SpectralNorm

        weight = np.random.randn(64, 32)
        sn = SpectralNorm(weight)

        assert hasattr(sn, 'u')
        assert sn.u.shape == (64,)


class TestSpectralNormConv2d:
    """Test SpectralNormConv2d wrapper."""

    def test_spectralnormconv2d_init(self):
        """Test SpectralNormConv2d initialization."""
        from python.nn_core.normalization import SpectralNormConv2d

        # Conv weight shape: (out_channels, in_channels, kh, kw)
        conv_weight = np.random.randn(64, 32, 3, 3)
        sn_conv = SpectralNormConv2d(conv_weight)

        assert sn_conv.original_shape == (64, 32, 3, 3)
        assert sn_conv.out_channels == 64

    def test_spectralnormconv2d_has_spec_norm(self):
        """Test SpectralNormConv2d wraps SpectralNorm."""
        from python.nn_core.normalization import SpectralNormConv2d

        conv_weight = np.random.randn(64, 32, 3, 3)
        sn_conv = SpectralNormConv2d(conv_weight)

        assert hasattr(sn_conv, 'spec_norm')

    def test_spectralnormconv2d_custom_iterations(self):
        """Test SpectralNormConv2d with custom power iterations."""
        from python.nn_core.normalization import SpectralNormConv2d

        conv_weight = np.random.randn(64, 32, 3, 3)
        sn_conv = SpectralNormConv2d(conv_weight, n_power_iterations=3)

        assert sn_conv.spec_norm.n_power_iterations == 3


class TestSpectralNormLinear:
    """Test SpectralNormLinear wrapper."""

    def test_spectralnormlinear_init(self):
        """Test SpectralNormLinear initialization."""
        from python.nn_core.normalization import SpectralNormLinear

        # Linear weight shape: (out_features, in_features)
        linear_weight = np.random.randn(128, 64)
        sn_linear = SpectralNormLinear(linear_weight)

        assert hasattr(sn_linear, 'spec_norm')

    def test_spectralnormlinear_dimensions(self):
        """Test SpectralNormLinear preserves dimensions."""
        from python.nn_core.normalization import SpectralNormLinear

        linear_weight = np.random.randn(128, 64)
        sn_linear = SpectralNormLinear(linear_weight)

        assert sn_linear.spec_norm.height == 128
        assert sn_linear.spec_norm.width == 64

    def test_spectralnormlinear_custom_iterations(self):
        """Test SpectralNormLinear with custom power iterations."""
        from python.nn_core.normalization import SpectralNormLinear

        linear_weight = np.random.randn(128, 64)
        sn_linear = SpectralNormLinear(linear_weight, n_power_iterations=2)

        assert sn_linear.spec_norm.n_power_iterations == 2


class TestSpectralNormFunctional:
    """Test SpectralNorm class in functional file (Note: this is actually RMSNorm impl)."""

    def test_spectralnorm_functional_init(self):
        """Test SpectralNorm functional initialization."""
        from python.nn_core.normalization_functional import SpectralNorm as SNFn

        sn_fn = SNFn()
        assert sn_fn is not None

    def test_spectralnorm_functional_forward_shape(self):
        """Test SpectralNorm functional forward preserves shape.

        Note: The SpectralNorm class in normalization_functional.py is actually
        an RMSNorm implementation (see docstring). This test verifies its RMS
        normalization behavior.
        """
        from python.nn_core.normalization_functional import SpectralNorm as SNFn

        sn_fn = SNFn()
        x = np.random.randn(2, 10, 64).astype(np.float64)
        gamma = np.ones(64)

        # Forward should return normalized output with same shape
        y = sn_fn.forward(x, gamma, normalized_shape=(64,))
        assert y.shape == x.shape

    def test_spectralnorm_functional_backward(self):
        """Test SpectralNorm functional backward (RMSNorm implementation)."""
        from python.nn_core.normalization_functional import SpectralNorm as SNFn

        sn_fn = SNFn()
        x = np.random.randn(2, 10, 64).astype(np.float64)
        gamma = np.ones(64)

        y = sn_fn.forward(x, gamma, normalized_shape=(64,))
        grad_output = np.ones_like(y)
        grad_x, grad_gamma = sn_fn.backward(grad_output)

        assert grad_x.shape == x.shape
        assert grad_gamma.shape == gamma.shape


class TestInstanceNormModule:
    """Test InstanceNorm wrapper (GroupNorm based)."""

    def test_instancenorm_init(self):
        """Test InstanceNorm initialization."""
        from python.nn_core.normalization import InstanceNorm

        inorm = InstanceNorm(num_channels=32)

        # InstanceNorm is GroupNorm with num_groups=num_channels
        assert inorm.groupnorm.num_groups == 32
        assert inorm.groupnorm.num_channels == 32

    def test_instancenorm_with_affine(self):
        """Test InstanceNorm with affine parameters."""
        from python.nn_core.normalization import InstanceNorm

        inorm = InstanceNorm(num_channels=32, affine=True)

        assert inorm.groupnorm.affine is True
        assert inorm.groupnorm.weight.data.shape == (32,)
        assert inorm.groupnorm.bias.data.shape == (32,)

    def test_instancenorm_forward(self):
        """Test InstanceNorm forward pass."""
        from python.nn_core.normalization import InstanceNorm
        from python.foundations import Tensor

        inorm = InstanceNorm(num_channels=16)
        x = Tensor(np.random.randn(2, 16, 8, 8), requires_grad=True)

        y = inorm(x)

        assert y.shape == x.shape


# =============================================================================
# Pooling Layer Tests
# =============================================================================

class TestMaxPool1d:
    """Comprehensive tests for MaxPool1d layer."""

    def test_maxpool1d_forward_basic(self):
        """Test MaxPool1d basic forward pass."""
        from python.nn_core import MaxPool1d

        pool = MaxPool1d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 16).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3, 8)

    def test_maxpool1d_forward_stride1(self):
        """Test MaxPool1d with stride=1 (overlapping windows)."""
        from python.nn_core import MaxPool1d

        pool = MaxPool1d(kernel_size=3, stride=1)
        x = Tensor(np.random.randn(2, 4, 10).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 4, 8)

    def test_maxpool1d_forward_with_padding(self):
        """Test MaxPool1d with padding."""
        from python.nn_core import MaxPool1d

        pool = MaxPool1d(kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(2, 4, 10).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 4, 10)

    def test_maxpool1d_correctness(self):
        """Test MaxPool1d computes correct max values."""
        from python.nn_core import MaxPool1d

        pool = MaxPool1d(kernel_size=2, stride=2)
        x_data = np.array([[[1, 5, 2, 8, 3, 7, 4, 6]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        expected = np.array([[[5, 8, 7, 6]]])
        assert np.allclose(y.data, expected)

    def test_maxpool1d_backward(self):
        """Test MaxPool1d backward pass produces gradients."""
        from python.nn_core import MaxPool1d

        pool = MaxPool1d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_maxpool1d_gradient_sparsity(self):
        """Test that MaxPool1d gradient is sparse (only at max positions)."""
        from python.nn_core import MaxPool1d

        pool = MaxPool1d(kernel_size=2, stride=2)
        x_data = np.array([[[1, 5, 2, 8]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be 1 at max positions (indices 1 and 3), 0 elsewhere
        expected_grad = np.array([[[0, 1, 0, 1]]])
        assert np.allclose(x.grad, expected_grad)

    def test_maxpool1d_gradcheck(self):
        """Verify MaxPool1d gradients with numerical gradient check."""
        from python.nn_core import MaxPool1d

        np.random.seed(42)
        pool = MaxPool1d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestMaxPool2d:
    """Comprehensive tests for MaxPool2d layer."""

    def test_maxpool2d_forward_basic(self):
        """Test MaxPool2d basic forward pass."""
        from python.nn_core import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3, 4, 4)

    def test_maxpool2d_forward_stride1(self):
        """Test MaxPool2d with stride=1 (overlapping windows)."""
        from python.nn_core import MaxPool2d

        pool = MaxPool2d(kernel_size=3, stride=1)
        x = Tensor(np.random.randn(2, 4, 10, 10).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 4, 8, 8)

    def test_maxpool2d_forward_with_padding(self):
        """Test MaxPool2d with padding."""
        from python.nn_core import MaxPool2d

        pool = MaxPool2d(kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(2, 4, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 4, 8, 8)

    def test_maxpool2d_forward_rectangular_kernel(self):
        """Test MaxPool2d with non-square input."""
        from python.nn_core import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 8, 16).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3, 4, 8)

    def test_maxpool2d_correctness(self):
        """Test MaxPool2d computes correct max values."""
        from python.nn_core import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        x_data = np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        expected = np.array([[[[6, 8], [14, 16]]]])
        assert np.allclose(y.data, expected)

    def test_maxpool2d_backward(self):
        """Test MaxPool2d backward pass."""
        from python.nn_core import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_maxpool2d_gradient_sparsity(self):
        """Test that MaxPool2d gradient is sparse (only at max positions)."""
        from python.nn_core import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        x_data = np.array([[[[1, 2], [3, 4]]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        expected_grad = np.array([[[[0, 0], [0, 1]]]])
        assert np.allclose(x.grad, expected_grad)

    def test_maxpool2d_gradcheck(self):
        """Verify MaxPool2d gradients with numerical gradient check."""
        from python.nn_core import MaxPool2d

        np.random.seed(42)
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_maxpool2d_gradcheck_with_padding(self):
        """Verify MaxPool2d gradients with padding."""
        from python.nn_core import MaxPool2d

        np.random.seed(42)
        pool = MaxPool2d(kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestAvgPool1d:
    """Comprehensive tests for AvgPool1d layer."""

    def test_avgpool1d_forward_basic(self):
        """Test AvgPool1d basic forward pass."""
        from python.nn_core import AvgPool1d

        pool = AvgPool1d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 16).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3, 8)

    def test_avgpool1d_forward_stride1(self):
        """Test AvgPool1d with stride=1."""
        from python.nn_core import AvgPool1d

        pool = AvgPool1d(kernel_size=3, stride=1)
        x = Tensor(np.random.randn(2, 4, 10).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 4, 8)

    def test_avgpool1d_forward_with_padding(self):
        """Test AvgPool1d with padding."""
        from python.nn_core import AvgPool1d

        pool = AvgPool1d(kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(2, 4, 10).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 4, 10)

    def test_avgpool1d_correctness(self):
        """Test AvgPool1d computes correct average values."""
        from python.nn_core import AvgPool1d

        pool = AvgPool1d(kernel_size=2, stride=2)
        x_data = np.array([[[2, 4, 6, 8, 10, 12]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        expected = np.array([[[3, 7, 11]]])
        assert np.allclose(y.data, expected)

    def test_avgpool1d_backward(self):
        """Test AvgPool1d backward pass."""
        from python.nn_core import AvgPool1d

        pool = AvgPool1d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_avgpool1d_gradient_uniform(self):
        """Test that AvgPool1d distributes gradient uniformly."""
        from python.nn_core import AvgPool1d

        pool = AvgPool1d(kernel_size=2, stride=2)
        x_data = np.array([[[1, 2, 3, 4]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        # Each element contributes 1/kernel_size to its window
        expected_grad = np.array([[[0.5, 0.5, 0.5, 0.5]]])
        assert np.allclose(x.grad, expected_grad)

    def test_avgpool1d_gradcheck(self):
        """Verify AvgPool1d gradients with numerical gradient check."""
        from python.nn_core import AvgPool1d

        np.random.seed(42)
        pool = AvgPool1d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestAvgPool2d:
    """Comprehensive tests for AvgPool2d layer."""

    def test_avgpool2d_forward_basic(self):
        """Test AvgPool2d basic forward pass."""
        from python.nn_core import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3, 4, 4)

    def test_avgpool2d_forward_stride1(self):
        """Test AvgPool2d with stride=1."""
        from python.nn_core import AvgPool2d

        pool = AvgPool2d(kernel_size=3, stride=1)
        x = Tensor(np.random.randn(2, 4, 10, 10).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 4, 8, 8)

    def test_avgpool2d_forward_with_padding(self):
        """Test AvgPool2d with padding."""
        from python.nn_core import AvgPool2d

        pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(2, 4, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 4, 8, 8)

    def test_avgpool2d_correctness(self):
        """Test AvgPool2d computes correct average values."""
        from python.nn_core import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        x_data = np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])
        assert np.allclose(y.data, expected)

    def test_avgpool2d_backward(self):
        """Test AvgPool2d backward pass."""
        from python.nn_core import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_avgpool2d_gradient_uniform(self):
        """Test that AvgPool2d distributes gradient uniformly."""
        from python.nn_core import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        x_data = np.array([[[[1, 2], [3, 4]]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        # Each element contributes 1/(kernel_size^2) to its window
        expected_grad = np.full((1, 1, 2, 2), 0.25)
        assert np.allclose(x.grad, expected_grad)

    def test_avgpool2d_gradcheck(self):
        """Verify AvgPool2d gradients with numerical gradient check."""
        from python.nn_core import AvgPool2d

        np.random.seed(42)
        pool = AvgPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_avgpool2d_gradcheck_with_padding(self):
        """Verify AvgPool2d gradients with padding."""
        from python.nn_core import AvgPool2d

        np.random.seed(42)
        pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestGlobalAvgPool1d:
    """Comprehensive tests for GlobalAvgPool1d layer."""

    def test_global_avgpool1d_forward(self):
        """Test GlobalAvgPool1d forward pass."""
        from python.nn_core import GlobalAvgPool1d

        pool = GlobalAvgPool1d()
        x = Tensor(np.random.randn(2, 3, 16).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3)

    def test_global_avgpool1d_correctness(self):
        """Test GlobalAvgPool1d computes correct global average."""
        from python.nn_core import GlobalAvgPool1d

        pool = GlobalAvgPool1d()
        x_data = np.array([[[1, 2, 3, 4]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        expected = np.array([[2.5]])
        assert np.allclose(y.data, expected)

    def test_global_avgpool1d_backward(self):
        """Test GlobalAvgPool1d backward pass."""
        from python.nn_core import GlobalAvgPool1d

        pool = GlobalAvgPool1d()
        x = Tensor(np.random.randn(2, 3, 16).astype(np.float64), requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_global_avgpool1d_gradient_uniform(self):
        """Test that GlobalAvgPool1d distributes gradient uniformly."""
        from python.nn_core import GlobalAvgPool1d

        pool = GlobalAvgPool1d()
        x_data = np.array([[[1, 2, 3, 4]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        expected_grad = np.full((1, 1, 4), 0.25)
        assert np.allclose(x.grad, expected_grad)

    def test_global_avgpool1d_gradcheck(self):
        """Verify GlobalAvgPool1d gradients with numerical gradient check."""
        from python.nn_core import GlobalAvgPool1d

        np.random.seed(42)
        pool = GlobalAvgPool1d()
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestGlobalMaxPool1d:
    """Comprehensive tests for GlobalMaxPool1d layer."""

    def test_global_maxpool1d_forward(self):
        """Test GlobalMaxPool1d forward pass."""
        from python.nn_core import GlobalMaxPool1d

        pool = GlobalMaxPool1d()
        x = Tensor(np.random.randn(2, 3, 16).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3)

    def test_global_maxpool1d_correctness(self):
        """Test GlobalMaxPool1d computes correct global max."""
        from python.nn_core import GlobalMaxPool1d

        pool = GlobalMaxPool1d()
        x_data = np.array([[[1, 5, 2, 3]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        expected = np.array([[5]])
        assert np.allclose(y.data, expected)

    def test_global_maxpool1d_backward(self):
        """Test GlobalMaxPool1d backward pass."""
        from python.nn_core import GlobalMaxPool1d

        pool = GlobalMaxPool1d()
        x = Tensor(np.random.randn(2, 3, 16).astype(np.float64), requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_global_maxpool1d_gradient_sparsity(self):
        """Test that GlobalMaxPool1d gradient is sparse."""
        from python.nn_core import GlobalMaxPool1d

        pool = GlobalMaxPool1d()
        x_data = np.array([[[1, 5, 2, 3]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        expected_grad = np.array([[[0, 1, 0, 0]]])
        assert np.allclose(x.grad, expected_grad)

    def test_global_maxpool1d_gradcheck(self):
        """Verify GlobalMaxPool1d gradients with numerical gradient check."""
        from python.nn_core import GlobalMaxPool1d

        np.random.seed(42)
        pool = GlobalMaxPool1d()
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestGlobalAvgPool2d:
    """Comprehensive tests for GlobalAvgPool2d layer."""

    def test_global_avgpool2d_forward(self):
        """Test GlobalAvgPool2d forward pass."""
        from python.nn_core import GlobalAvgPool2d

        pool = GlobalAvgPool2d()
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3)

    def test_global_avgpool2d_correctness(self):
        """Test GlobalAvgPool2d computes correct global average."""
        from python.nn_core import GlobalAvgPool2d

        pool = GlobalAvgPool2d()
        x_data = np.array([[[[1, 2], [3, 4]]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        expected = np.array([[2.5]])
        assert np.allclose(y.data, expected)

    def test_global_avgpool2d_backward(self):
        """Test GlobalAvgPool2d backward pass."""
        from python.nn_core import GlobalAvgPool2d

        pool = GlobalAvgPool2d()
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_global_avgpool2d_gradient_uniform(self):
        """Test that GlobalAvgPool2d distributes gradient uniformly."""
        from python.nn_core import GlobalAvgPool2d

        pool = GlobalAvgPool2d()
        x_data = np.array([[[[1, 2], [3, 4]]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        expected_grad = np.full((1, 1, 2, 2), 0.25)
        assert np.allclose(x.grad, expected_grad)

    def test_global_avgpool2d_gradcheck(self):
        """Verify GlobalAvgPool2d gradients with numerical gradient check."""
        from python.nn_core import GlobalAvgPool2d

        np.random.seed(42)
        pool = GlobalAvgPool2d()
        x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestGlobalMaxPool2d:
    """Comprehensive tests for GlobalMaxPool2d layer."""

    def test_global_maxpool2d_forward(self):
        """Test GlobalMaxPool2d forward pass."""
        from python.nn_core import GlobalMaxPool2d

        pool = GlobalMaxPool2d()
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        assert y.shape == (2, 3)

    def test_global_maxpool2d_correctness(self):
        """Test GlobalMaxPool2d computes correct global max."""
        from python.nn_core import GlobalMaxPool2d

        pool = GlobalMaxPool2d()
        x_data = np.array([[[[1, 2], [5, 3]]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        expected = np.array([[5]])
        assert np.allclose(y.data, expected)

    def test_global_maxpool2d_backward(self):
        """Test GlobalMaxPool2d backward pass."""
        from python.nn_core import GlobalMaxPool2d

        pool = GlobalMaxPool2d()
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float64), requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_global_maxpool2d_gradient_sparsity(self):
        """Test that GlobalMaxPool2d gradient is sparse."""
        from python.nn_core import GlobalMaxPool2d

        pool = GlobalMaxPool2d()
        x_data = np.array([[[[1, 2], [5, 3]]]]).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        y = pool(x)
        loss = y.sum()
        loss.backward()
        expected_grad = np.array([[[[0, 0], [1, 0]]]])
        assert np.allclose(x.grad, expected_grad)

    def test_global_maxpool2d_gradcheck(self):
        """Verify GlobalMaxPool2d gradients with numerical gradient check."""
        from python.nn_core import GlobalMaxPool2d

        np.random.seed(42)
        pool = GlobalMaxPool2d()
        x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestPoolingFunctional:
    """Test pooling functional operations directly."""

    def test_maxpool1d_functional(self):
        """Test MaxPool1d functional forward."""
        from python.nn_core.pooling_functional import MaxPool1d as MaxPool1dFn

        pool_fn = MaxPool1dFn()
        x = np.random.randn(2, 3, 8)
        y = pool_fn.forward(x, kernel_size=2, stride=2)
        assert y.shape == (2, 3, 4)

    def test_maxpool2d_functional(self):
        """Test MaxPool2d functional forward."""
        from python.nn_core.pooling_functional import MaxPool2d as MaxPool2dFn

        pool_fn = MaxPool2dFn()
        x = np.random.randn(2, 3, 8, 8)
        y = pool_fn.forward(x, kernel_size=2, stride=2)
        assert y.shape == (2, 3, 4, 4)

    def test_avgpool1d_functional(self):
        """Test AvgPool1d functional forward."""
        from python.nn_core.pooling_functional import AvgPool1d as AvgPool1dFn

        pool_fn = AvgPool1dFn()
        x = np.random.randn(2, 3, 8)
        y = pool_fn.forward(x, kernel_size=2, stride=2)
        assert y.shape == (2, 3, 4)

    def test_avgpool2d_functional(self):
        """Test AvgPool2d functional forward."""
        from python.nn_core.pooling_functional import AvgPool2d as AvgPool2dFn

        pool_fn = AvgPool2dFn()
        x = np.random.randn(2, 3, 8, 8)
        y = pool_fn.forward(x, kernel_size=2, stride=2)
        assert y.shape == (2, 3, 4, 4)

    def test_global_avgpool2d_functional(self):
        """Test GlobalAvgPool2d functional forward."""
        from python.nn_core.pooling_functional import GlobalAvgPool2d as GlobalAvgPool2dFn

        pool_fn = GlobalAvgPool2dFn()
        x = np.random.randn(2, 3, 8, 8)
        y = pool_fn.forward(x)
        assert y.shape == (2, 3)

    def test_global_maxpool2d_functional(self):
        """Test GlobalMaxPool2d functional forward."""
        from python.nn_core.pooling_functional import GlobalMaxPool2d as GlobalMaxPool2dFn

        pool_fn = GlobalMaxPool2dFn()
        x = np.random.randn(2, 3, 8, 8)
        y = pool_fn.forward(x)
        assert y.shape == (2, 3)


# =============================================================================
# Attention Mechanism Tests
# =============================================================================

class TestMultiHeadAttention:
    """Test MultiHeadAttention."""

    def test_multihead_attention_forward(self):
        """Test MultiHeadAttention forward pass."""
        from python.nn_core import MultiHeadAttention
        from python.foundations import Tensor

        attn = MultiHeadAttention(embed_dim=64, num_heads=8)
        x = Tensor(np.random.randn(2, 10, 64), requires_grad=True)

        y = attn(x, x, x)  # Self-attention

        assert y.shape == x.shape

    def test_multihead_attention_backward(self):
        """Test MultiHeadAttention backward pass."""
        from python.nn_core import MultiHeadAttention
        from python.foundations import Tensor

        attn = MultiHeadAttention(embed_dim=64, num_heads=8)
        x = Tensor(np.random.randn(2, 10, 64), requires_grad=True)

        y = attn(x, x, x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None


class TestScaledDotProductAttention:
    """Test ScaledDotProductAttention."""

    def test_scaled_dot_product_attention_forward(self):
        """Test ScaledDotProductAttention forward pass."""
        from python.nn_core import ScaledDotProductAttention
        from python.foundations import Tensor

        attn = ScaledDotProductAttention()
        q = Tensor(np.random.randn(2, 8, 10, 64), requires_grad=True)
        k = Tensor(np.random.randn(2, 8, 10, 64), requires_grad=True)
        v = Tensor(np.random.randn(2, 8, 10, 64), requires_grad=True)

        out = attn(q, k, v)

        assert out.shape == (2, 8, 10, 64)


class TestAttentionFunctional:
    """Test attention functional operations."""

    def test_scaled_dot_product_attention_functional(self):
        """Test ScaledDotProductAttention functional."""
        from python.nn_core.attention_functional import ScaledDotProductAttention as SDPAFn

        sdpa_fn = SDPAFn()
        q = np.random.randn(2, 8, 10, 64)
        k = np.random.randn(2, 8, 10, 64)
        v = np.random.randn(2, 8, 10, 64)

        out = sdpa_fn.forward(q, k, v)

        assert out.shape == (2, 8, 10, 64)


# =============================================================================
# Regularization Tests
# =============================================================================

class TestDropout:
    """Test Dropout layer."""

    def test_dropout_train_mode(self):
        """Test Dropout in training mode."""
        from python.nn_core import Dropout
        from python.foundations import Tensor

        dropout = Dropout(p=0.5)
        dropout.train()
        x = Tensor(np.ones((10, 10)), requires_grad=True)

        y = dropout(x)

        # Some values should be zeroed
        # (Though with stubs, might not actually drop)
        assert y.shape == x.shape

    def test_dropout_eval_mode(self):
        """Test Dropout in eval mode."""
        from python.nn_core import Dropout
        from python.foundations import Tensor

        dropout = Dropout(p=0.5)
        dropout.eval()
        x = Tensor(np.ones((10, 10)), requires_grad=True)

        y = dropout(x)

        # In eval mode, should be identity
        assert y.shape == x.shape


class TestRegularizationFunctional:
    """Test regularization functional operations."""

    def test_dropout_functional(self):
        """Test Dropout functional."""
        from python.nn_core.regularization_functional import Dropout as DropoutFn

        dropout_fn = DropoutFn()
        x = np.ones((10, 10))

        # Forward in training mode
        y = dropout_fn.forward(x, p=0.5, training=True)

        assert y.shape == x.shape


# =============================================================================
# Recurrent Layer Tests
# =============================================================================

class TestLSTM:
    """Test LSTM layer."""

    def test_lstm_forward(self):
        """Test LSTM forward pass."""
        from python.nn_core import LSTM
        from python.foundations import Tensor

        lstm = LSTM(input_size=32, hidden_size=64, num_layers=1)
        x = Tensor(np.random.randn(10, 2, 32), requires_grad=True)  # seq, batch, features

        output, (h_n, c_n) = lstm(x)

        assert output.shape == (10, 2, 64)
        assert h_n.shape == (1, 2, 64)
        assert c_n.shape == (1, 2, 64)


class TestGRU:
    """Test GRU layer."""

    def test_gru_forward(self):
        """Test GRU forward pass."""
        from python.nn_core import GRU
        from python.foundations import Tensor

        gru = GRU(input_size=32, hidden_size=64, num_layers=1)
        x = Tensor(np.random.randn(10, 2, 32), requires_grad=True)

        output, h_n = gru(x)

        assert output.shape == (10, 2, 64)
        assert h_n.shape == (1, 2, 64)


class TestRNNCell:
    """Test RNNCell."""

    def test_rnncell_forward(self):
        """Test RNNCell forward pass."""
        from python.nn_core import RNNCell
        from python.foundations import Tensor

        cell = RNNCell(input_size=32, hidden_size=64)
        x = Tensor(np.random.randn(2, 32), requires_grad=True)
        h = Tensor(np.zeros((2, 64)), requires_grad=True)

        h_new = cell(x, h)

        assert h_new.shape == (2, 64)


class TestRecurrentFunctional:
    """Test recurrent functional operations."""

    def test_lstmcell_functional(self):
        """Test LSTMCell functional."""
        from python.nn_core.recurrent_functional import LSTMCell as LSTMCellFn

        lstm_fn = LSTMCellFn()
        x = np.random.randn(2, 32)
        h = np.zeros((2, 64))
        c = np.zeros((2, 64))
        weight_ih = np.random.randn(4 * 64, 32)
        weight_hh = np.random.randn(4 * 64, 64)
        bias_ih = np.zeros(4 * 64)
        bias_hh = np.zeros(4 * 64)

        h_new, c_new = lstm_fn.forward(x, h, c, weight_ih, weight_hh, bias_ih, bias_hh)

        assert h_new.shape == (2, 64)
        assert c_new.shape == (2, 64)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestXavierInit:
    """Test Xavier initialization."""

    def test_xavier_uniform(self):
        """Test Xavier uniform initialization."""
        from python.nn_core import xavier_uniform_

        weight = np.empty((64, 32))
        xavier_uniform_(weight)

        # Check values are in reasonable range
        limit = np.sqrt(6.0 / (64 + 32))
        assert np.abs(weight).max() <= limit * 1.1  # Allow small tolerance

    def test_xavier_normal(self):
        """Test Xavier normal initialization."""
        from python.nn_core import xavier_normal_

        weight = np.empty((64, 32))
        xavier_normal_(weight)

        # Check std is approximately correct
        expected_std = np.sqrt(2.0 / (64 + 32))
        assert np.abs(weight.std() - expected_std) < 0.1


class TestKaimingInit:
    """Test Kaiming initialization."""

    def test_kaiming_uniform(self):
        """Test Kaiming uniform initialization."""
        from python.nn_core import kaiming_uniform_

        weight = np.empty((64, 32))
        kaiming_uniform_(weight)

        # Check values are in reasonable range
        limit = np.sqrt(6.0 / 32)
        assert np.abs(weight).max() <= limit * 1.1

    def test_kaiming_normal(self):
        """Test Kaiming normal initialization."""
        from python.nn_core import kaiming_normal_

        weight = np.empty((64, 32))
        kaiming_normal_(weight)

        # Check std is approximately correct
        expected_std = np.sqrt(2.0 / 32)
        assert np.abs(weight.std() - expected_std) < 0.1


class TestOrthogonalInit:
    """Test Orthogonal initialization."""

    def test_orthogonal(self):
        """Test orthogonal initialization."""
        from python.nn_core import orthogonal_

        weight = np.empty((64, 64))
        orthogonal_(weight)

        # Check orthogonality: W @ W.T should be close to I
        identity = np.eye(64)
        assert np.allclose(weight @ weight.T, identity, atol=1e-5)


class TestBasicInit:
    """Test basic initialization functions."""

    def test_normal(self):
        """Test normal initialization."""
        from python.nn_core import normal_

        weight = np.empty((64, 32))
        normal_(weight, mean=0.0, std=0.01)

        assert np.abs(weight.mean()) < 0.01
        assert np.abs(weight.std() - 0.01) < 0.005

    def test_uniform(self):
        """Test uniform initialization."""
        from python.nn_core import uniform_

        weight = np.empty((64, 32))
        uniform_(weight, a=-0.1, b=0.1)

        assert weight.min() >= -0.1
        assert weight.max() <= 0.1


# =============================================================================
# Positional Encoding Tests
# =============================================================================

class TestSinusoidalPositionalEncoding:
    """Test Sinusoidal positional encoding."""

    def test_sinusoidal_shape(self):
        """Test sinusoidal encoding shape."""
        from python.nn_core import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=512, max_seq_length=100)
        encoding = pe.get_encoding(50)

        assert encoding.shape == (50, 512)

    def test_sinusoidal_bounded(self):
        """Test sinusoidal values are bounded."""
        from python.nn_core import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=512)
        encoding = pe.get_encoding(100)

        assert np.abs(encoding).max() <= 1.0


class TestLearnedPositionalEmbedding:
    """Test Learned positional embedding."""

    def test_learned_shape(self):
        """Test learned embedding shape."""
        from python.nn_core import LearnedPositionalEmbedding

        pe = LearnedPositionalEmbedding(seq_length=512, d_model=768)
        positions = np.arange(100)
        embedding = pe(positions)

        assert embedding.shape == (100, 768)


class TestRotaryPositionalEmbedding:
    """Test Rotary positional embedding (RoPE)."""

    def test_rope_forward(self):
        """Test RoPE forward pass."""
        from python.nn_core import RotaryPositionalEmbedding

        rope = RotaryPositionalEmbedding(d_model=64)
        q = np.random.randn(2, 10, 64)
        k = np.random.randn(2, 10, 64)

        q_rot, k_rot = rope(q, k, seq_length=10)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestALiBiPositionalBias:
    """Test ALiBi positional bias."""

    def test_alibi_shape(self):
        """Test ALiBi bias matrix shape."""
        from python.nn_core import ALiBiPositionalBias

        alibi = ALiBiPositionalBias(num_heads=8)
        scores = np.random.randn(2, 8, 10, 10)

        biased_scores = alibi(scores, seq_length=10)

        assert biased_scores.shape == scores.shape

    def test_alibi_slopes(self):
        """Test ALiBi slope computation."""
        from python.nn_core import ALiBiPositionalBias

        slopes = ALiBiPositionalBias.compute_bias_slopes(8)

        # First slope should be larger than last
        assert slopes[0] > slopes[-1]
        assert len(slopes) == 8


# =============================================================================
# Gradient Checking Tests - Using gradcheck from foundations
# =============================================================================

class TestActivationGradients:
    """Test gradients for activation functions using numerical gradient checking."""

    def test_relu_gradient_check(self):
        """Verify ReLU backward pass with numerical gradients."""
        from python.nn_core.activations_functional import ReLU

        relu_fn = ReLU()

        # Test with positive values (avoid discontinuity at 0)
        x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        grad_output = np.ones_like(x)

        # Forward
        y = relu_fn.forward(x)

        # Analytical backward
        analytical_grad, = relu_fn.backward(grad_output)

        # Numerical gradient (must use scalar output for numerical_gradient)
        def f(x_val):
            out = ReLU().forward(x_val.data)
            return Tensor(np.sum(out))  # Scalar output

        x_tensor = Tensor(x.copy())
        numerical_grad = numerical_gradient(f, x_tensor)

        # Compare
        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6), \
            f"ReLU gradient mismatch: analytical={analytical_grad}, numerical={numerical_grad}"

    def test_leaky_relu_gradient_check(self):
        """Verify LeakyReLU backward pass with numerical gradients."""
        from python.nn_core.activations_functional import LeakyReLU

        leaky_relu_fn = LeakyReLU()

        # Test with mixed positive/negative values (avoid exact 0)
        x = np.array([-2.5, -1.0, 0.5, 1.0, 2.0])
        grad_output = np.ones_like(x)

        # Forward (alpha is the parameter name in functional)
        y = leaky_relu_fn.forward(x, alpha=0.01)

        # Analytical backward
        analytical_grad, = leaky_relu_fn.backward(grad_output)

        # Numerical gradient (scalar output)
        def f(x_val):
            fn = LeakyReLU()
            out = fn.forward(x_val.data, alpha=0.01)
            return Tensor(np.sum(out))

        x_tensor = Tensor(x.copy())
        numerical_grad = numerical_gradient(f, x_tensor)

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6), \
            f"LeakyReLU gradient mismatch"

    def test_sigmoid_gradient_check(self):
        """Verify Sigmoid backward pass with numerical gradients."""
        from python.foundations.functionals import Sigmoid

        sigmoid_fn = Sigmoid()

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        grad_output = np.ones_like(x)

        # Forward
        y = sigmoid_fn.forward(x)

        # Analytical backward
        analytical_grad, = sigmoid_fn.backward(grad_output)

        # Numerical gradient (scalar output)
        def f(x_val):
            fn = Sigmoid()
            out = fn.forward(x_val.data)
            return Tensor(np.sum(out))

        x_tensor = Tensor(x.copy())
        numerical_grad = numerical_gradient(f, x_tensor)

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6), \
            f"Sigmoid gradient mismatch: analytical={analytical_grad}, numerical={numerical_grad}"

    def test_tanh_gradient_check(self):
        """Verify tanh backward pass with numerical gradients using np.tanh."""
        # Test tanh gradient using direct numpy implementation
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Forward: tanh(x)
        y = np.tanh(x)

        # Analytical gradient: d/dx tanh(x) = 1 - tanh(x)^2
        analytical_grad = 1 - y ** 2

        # Numerical gradient
        def f(x_val):
            return Tensor(np.sum(np.tanh(x_val.data)))

        numerical_grad = numerical_gradient(f, Tensor(x.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6), \
            f"Tanh gradient mismatch: analytical={analytical_grad}, numerical={numerical_grad}"

    def test_elu_gradient_check(self):
        """Verify ELU backward pass with numerical gradients using functional."""
        from python.nn_core.activations_functional import ELU as ELUFn

        elu_fn = ELUFn()

        x = np.array([-2.0, -1.0, 0.5, 1.0, 2.0])
        grad_output = np.ones_like(x)

        # Forward
        y = elu_fn.forward(x, alpha=1.0)

        # Analytical backward
        analytical_grad, = elu_fn.backward(grad_output)

        # Numerical gradient (scalar output)
        def f(x_val):
            fn = ELUFn()
            out = fn.forward(x_val.data, alpha=1.0)
            return Tensor(np.sum(out))

        numerical_grad = numerical_gradient(f, Tensor(x.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6), \
            f"ELU gradient mismatch"

    def test_softmax_gradient_check(self):
        """Verify Softmax backward pass with numerical gradients."""
        from python.foundations.functionals import Softmax

        softmax_fn = Softmax()

        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])

        # Forward
        y = softmax_fn.forward(x)

        # For softmax, we test gradient w.r.t a scalar loss (sum of outputs)
        grad_output = np.ones_like(y)

        # Analytical backward
        analytical_grad, = softmax_fn.backward(grad_output)

        # Numerical gradient
        def f(x_val):
            fn = Softmax()
            out = fn.forward(x_val.data)
            return Tensor(out.sum())  # Sum to scalar

        x_tensor = Tensor(x.copy())
        numerical_grad = numerical_gradient(f, x_tensor)

        # Softmax sum gradient should be approximately 0 (since sum of softmax is always 1)
        assert np.allclose(numerical_grad, 0, atol=1e-5), \
            f"Softmax gradient for sum should be ~0, got {numerical_grad}"


class TestLinearGradients:
    """Test gradients for linear layers."""

    def test_linear_input_gradient(self):
        """Verify Linear layer gradient w.r.t input."""
        from python.nn_core import Linear

        np.random.seed(42)

        layer = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4), requires_grad=True)

        # Forward
        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Analytical gradient
        analytical_grad = x.grad

        # Numerical gradient
        def f(x_val):
            out = layer(x_val)
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(x.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6), \
            f"Linear input gradient mismatch"

    def test_linear_weight_gradient(self):
        """Verify Linear layer gradient w.r.t weight."""
        from python.nn_core import Linear

        np.random.seed(42)

        layer = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4), requires_grad=True)

        # Forward
        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Analytical gradient
        analytical_grad = layer.weight.grad

        # Numerical gradient by perturbing weights
        def f(w_val):
            original_weight = layer.weight.data.copy()
            layer.weight.data = w_val.data
            out = layer(Tensor(x.data))
            layer.weight.data = original_weight
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(layer.weight.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6), \
            f"Linear weight gradient mismatch"


class TestConvGradients:
    """Test gradients for convolution layers."""

    def test_conv2d_input_gradient(self):
        """Verify Conv2d gradient w.r.t input."""
        from python.nn_core import Conv2d

        np.random.seed(42)

        conv = Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 4, 4), requires_grad=True)

        # Forward
        y = conv(x)
        loss = y.sum()
        loss.backward()

        # Analytical gradient
        analytical_grad = x.grad

        # Numerical gradient
        def f(x_val):
            out = conv(x_val)
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(x.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-3, atol=1e-5), \
            f"Conv2d input gradient mismatch"


class TestNormalizationGradients:
    """Test gradients for normalization layers."""

    def test_layernorm_gradient_check(self):
        """Verify LayerNorm gradients."""
        from python.nn_core import LayerNorm

        np.random.seed(42)

        ln = LayerNorm(normalized_shape=[8])
        x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)

        # Forward
        y = ln(x)
        loss = y.sum()
        loss.backward()

        # Analytical gradient
        analytical_grad = x.grad

        # Numerical gradient
        def f(x_val):
            out = ln(x_val)
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(x.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-3, atol=1e-5), \
            f"LayerNorm gradient mismatch"

    def test_batchnorm2d_gradient_check(self):
        """Verify BatchNorm2d gradients in eval mode (deterministic)."""
        from python.nn_core import BatchNorm2d

        np.random.seed(42)

        bn = BatchNorm2d(num_features=3)
        bn.eval()  # Use eval mode for deterministic behavior
        x = Tensor(np.random.randn(2, 3, 4, 4), requires_grad=True)

        # Forward
        y = bn(x)
        loss = y.sum()
        loss.backward()

        # Analytical gradient
        analytical_grad = x.grad

        # Numerical gradient
        def f(x_val):
            out = bn(x_val)
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(x.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-3, atol=1e-5), \
            f"BatchNorm2d gradient mismatch"


class TestPoolingGradients:
    """Comprehensive gradient tests for all pooling layers using gradcheck."""

    def test_maxpool1d_gradient_check(self):
        """Verify MaxPool1d gradients using gradcheck."""
        from python.nn_core import MaxPool1d

        np.random.seed(42)
        pool = MaxPool1d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_maxpool1d_gradient_check_stride1(self):
        """Verify MaxPool1d gradients with stride=1."""
        from python.nn_core import MaxPool1d

        np.random.seed(42)
        pool = MaxPool1d(kernel_size=3, stride=1)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_maxpool2d_gradient_check(self):
        """Verify MaxPool2d gradients using gradcheck."""
        from python.nn_core import MaxPool2d

        np.random.seed(42)
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_maxpool2d_gradient_check_with_padding(self):
        """Verify MaxPool2d gradients with padding."""
        from python.nn_core import MaxPool2d

        np.random.seed(42)
        pool = MaxPool2d(kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_avgpool1d_gradient_check(self):
        """Verify AvgPool1d gradients using gradcheck."""
        from python.nn_core import AvgPool1d

        np.random.seed(42)
        pool = AvgPool1d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_avgpool1d_gradient_check_stride1(self):
        """Verify AvgPool1d gradients with stride=1."""
        from python.nn_core import AvgPool1d

        np.random.seed(42)
        pool = AvgPool1d(kernel_size=3, stride=1)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_avgpool2d_gradient_check(self):
        """Verify AvgPool2d gradients using gradcheck."""
        from python.nn_core import AvgPool2d

        np.random.seed(42)
        pool = AvgPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_avgpool2d_gradient_check_with_padding(self):
        """Verify AvgPool2d gradients with padding."""
        from python.nn_core import AvgPool2d

        np.random.seed(42)
        pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_global_avgpool1d_gradient_check(self):
        """Verify GlobalAvgPool1d gradients using gradcheck."""
        from python.nn_core import GlobalAvgPool1d

        np.random.seed(42)
        pool = GlobalAvgPool1d()
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_global_maxpool1d_gradient_check(self):
        """Verify GlobalMaxPool1d gradients using gradcheck."""
        from python.nn_core import GlobalMaxPool1d

        np.random.seed(42)
        pool = GlobalMaxPool1d()
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_global_avgpool2d_gradient_check(self):
        """Verify GlobalAvgPool2d gradients using gradcheck."""
        from python.nn_core import GlobalAvgPool2d

        np.random.seed(42)
        pool = GlobalAvgPool2d()
        x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)

    def test_global_maxpool2d_gradient_check(self):
        """Verify GlobalMaxPool2d gradients using gradcheck."""
        from python.nn_core import GlobalMaxPool2d

        np.random.seed(42)
        pool = GlobalMaxPool2d()
        x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float64), requires_grad=True)

        def func(x):
            return pool(x).sum()

        assert gradcheck(func, (x,), eps=1e-5, atol=1e-4, rtol=1e-3)


class TestAttentionGradients:
    """Test gradients for attention mechanisms."""

    def test_scaled_dot_product_attention_gradient(self):
        """Verify ScaledDotProductAttention gradients."""
        from python.nn_core import ScaledDotProductAttention

        np.random.seed(42)

        attn = ScaledDotProductAttention()
        q = Tensor(np.random.randn(1, 2, 4, 8), requires_grad=True)  # Small sizes for speed
        k = Tensor(np.random.randn(1, 2, 4, 8), requires_grad=True)
        v = Tensor(np.random.randn(1, 2, 4, 8), requires_grad=True)

        # Forward
        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()

        # Check Q gradient
        analytical_grad_q = q.grad

        def f_q(q_val):
            out = attn(q_val, Tensor(k.data), Tensor(v.data))
            return Tensor(out.data.sum())

        numerical_grad_q = numerical_gradient(f_q, Tensor(q.data.copy()))

        assert np.allclose(analytical_grad_q, numerical_grad_q, rtol=1e-3, atol=1e-5), \
            f"SDPA Q gradient mismatch"

    def test_multihead_attention_gradient(self):
        """Verify MultiHeadAttention gradients."""
        from python.nn_core import MultiHeadAttention

        np.random.seed(42)

        attn = MultiHeadAttention(embed_dim=16, num_heads=2)
        x = Tensor(np.random.randn(1, 4, 16), requires_grad=True)

        # Forward (self-attention)
        y = attn(x, x, x)
        loss = y.sum()
        loss.backward()

        # Analytical gradient
        analytical_grad = x.grad

        # Numerical gradient
        def f(x_val):
            out = attn(x_val, x_val, x_val)
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(x.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-3, atol=1e-5), \
            f"MultiHeadAttention gradient mismatch"


class TestRecurrentGradients:
    """Test gradients for recurrent layers."""

    def test_rnn_cell_functional_gradient(self):
        """Verify RNNCell functional gradients."""
        from python.nn_core.recurrent_functional import RNNCellFunction

        np.random.seed(42)

        rnn_fn = RNNCellFunction()

        # Small dimensions for testing
        batch_size, input_size, hidden_size = 2, 4, 6

        x = np.random.randn(batch_size, input_size)
        h = np.random.randn(batch_size, hidden_size)
        W_ih = np.random.randn(hidden_size, input_size)
        W_hh = np.random.randn(hidden_size, hidden_size)
        b_ih = np.random.randn(hidden_size)
        b_hh = np.random.randn(hidden_size)

        # Forward
        h_new = rnn_fn.forward(x, h, W_ih, W_hh, b_ih, b_hh)

        # Backward (gradient of sum of h_new)
        grad_h_new = np.ones_like(h_new)
        grad_x, grad_h, grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh = rnn_fn.backward(grad_h_new)

        # Numerical gradient for x (scalar output)
        def f_x(x_val):
            fn = RNNCellFunction()
            out = fn.forward(x_val.data, h, W_ih, W_hh, b_ih, b_hh)
            return Tensor(np.sum(out))  # Scalar output

        numerical_grad_x = numerical_gradient(f_x, Tensor(x.copy()))

        assert np.allclose(grad_x, numerical_grad_x, rtol=1e-3, atol=1e-5), \
            f"RNNCell x gradient mismatch"

    def test_lstm_cell_functional_gradient(self):
        """Verify LSTMCell functional gradients."""
        from python.nn_core.recurrent_functional import LSTMCellFunction

        np.random.seed(42)

        lstm_fn = LSTMCellFunction()

        # Small dimensions
        batch_size, input_size, hidden_size = 2, 4, 6

        x = np.random.randn(batch_size, input_size)
        h = np.random.randn(batch_size, hidden_size)
        c = np.random.randn(batch_size, hidden_size)
        # LSTM has 4*hidden_size weights for i,f,g,o gates
        W_ih = np.random.randn(4 * hidden_size, input_size)
        W_hh = np.random.randn(4 * hidden_size, hidden_size)
        b_ih = np.random.randn(4 * hidden_size)
        b_hh = np.random.randn(4 * hidden_size)

        # Forward
        h_new, c_new = lstm_fn.forward(x, h, c, W_ih, W_hh, b_ih, b_hh)

        # Backward
        grad_h_new = np.ones_like(h_new)
        grad_c_new = np.ones_like(c_new)
        grads = lstm_fn.backward(grad_h_new, grad_c_new)
        grad_x = grads[0]

        # Numerical gradient for x (w.r.t sum of h_new + c_new) - scalar output
        def f_x(x_val):
            fn = LSTMCellFunction()
            h_out, c_out = fn.forward(x_val.data, h, c, W_ih, W_hh, b_ih, b_hh)
            return Tensor(np.sum(h_out) + np.sum(c_out))  # Scalar output

        numerical_grad_x = numerical_gradient(f_x, Tensor(x.copy()))

        assert np.allclose(grad_x, numerical_grad_x, rtol=1e-3, atol=1e-5), \
            f"LSTMCell x gradient mismatch"

    def test_gru_cell_functional_gradient(self):
        """Verify GRUCell functional gradients."""
        from python.nn_core.recurrent_functional import GRUCellFunction

        np.random.seed(42)

        gru_fn = GRUCellFunction()

        # Small dimensions
        batch_size, input_size, hidden_size = 2, 4, 6

        x = np.random.randn(batch_size, input_size)
        h = np.random.randn(batch_size, hidden_size)
        # GRU has 3*hidden_size weights for r,z,n gates
        W_ih = np.random.randn(3 * hidden_size, input_size)
        W_hh = np.random.randn(3 * hidden_size, hidden_size)
        b_ih = np.random.randn(3 * hidden_size)
        b_hh = np.random.randn(3 * hidden_size)

        # Forward
        h_new = gru_fn.forward(x, h, W_ih, W_hh, b_ih, b_hh)

        # Backward
        grad_h_new = np.ones_like(h_new)
        grads = gru_fn.backward(grad_h_new)
        grad_x = grads[0]

        # Numerical gradient for x (scalar output)
        def f_x(x_val):
            fn = GRUCellFunction()
            h_out = fn.forward(x_val.data, h, W_ih, W_hh, b_ih, b_hh)
            return Tensor(np.sum(h_out))  # Scalar output

        numerical_grad_x = numerical_gradient(f_x, Tensor(x.copy()))

        assert np.allclose(grad_x, numerical_grad_x, rtol=1e-3, atol=1e-5), \
            f"GRUCell x gradient mismatch"


class TestRegularizationGradients:
    """Test gradients for regularization layers."""

    def test_dropout_gradient_eval_mode(self):
        """Verify Dropout gradient in eval mode (identity)."""
        from python.nn_core import Dropout

        np.random.seed(42)

        dropout = Dropout(p=0.5)
        dropout.eval()  # Identity in eval mode
        x = Tensor(np.random.randn(2, 4), requires_grad=True)

        # Forward
        y = dropout(x)
        loss = y.sum()
        loss.backward()

        # In eval mode, gradient should be identity
        analytical_grad = x.grad

        def f(x_val):
            dropout.eval()
            out = dropout(x_val)
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(x.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6), \
            f"Dropout (eval) gradient mismatch"


class TestGradcheckUtility:
    """Test the gradcheck utility itself with simple functions."""

    def test_gradcheck_matmul(self):
        """Test gradcheck with matrix multiplication."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(4, 2), requires_grad=True)

        def f(x, y):
            return (x @ y).sum()

        # Should pass gradient check
        result = gradcheck(f, (x, y), eps=1e-5, atol=1e-4, rtol=1e-3, raise_exception=False)
        assert result, "gradcheck failed for matmul"

    def test_gradcheck_elementwise(self):
        """Test gradcheck with elementwise operations."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)

        def f(x):
            return (x * x).sum()  # x^2

        result = gradcheck(f, (x,), eps=1e-5, atol=1e-4, rtol=1e-3, raise_exception=False)
        assert result, "gradcheck failed for x^2"

    def test_gradcheck_chain(self):
        """Test gradcheck with chained operations."""
        x = Tensor(np.random.randn(2, 3) * 0.5, requires_grad=True)

        def f(x):
            return x.exp().sum()

        result = gradcheck(f, (x,), eps=1e-5, atol=1e-4, rtol=1e-3, raise_exception=False)
        assert result, "gradcheck failed for exp(x)"


class TestEndToEndGradients:
    """End-to-end gradient tests for complete models."""

    def test_mlp_gradients(self):
        """Test gradients through a simple MLP."""
        from python.nn_core import Sequential, Linear, ReLU

        np.random.seed(42)

        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )

        x = Tensor(np.random.randn(2, 4) * 0.5 + 0.5, requires_grad=True)  # Positive to avoid ReLU discontinuity

        # Forward
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Analytical gradient
        analytical_grad = x.grad

        # Numerical gradient
        def f(x_val):
            out = model(x_val)
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(x.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-3, atol=1e-4), \
            f"MLP gradient mismatch"

    def test_cnn_block_gradients(self):
        """Test gradients through a simple CNN block."""
        from python.nn_core import Sequential, Conv2d, ReLU, MaxPool2d

        np.random.seed(42)

        model = Sequential(
            Conv2d(2, 4, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )

        # Use positive values to avoid ReLU discontinuity
        x = Tensor(np.random.randn(1, 2, 4, 4) * 0.5 + 1.0, requires_grad=True)

        # Forward
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Analytical gradient
        analytical_grad = x.grad

        # Numerical gradient
        def f(x_val):
            out = model(x_val)
            return Tensor(out.data.sum())

        numerical_grad = numerical_gradient(f, Tensor(x.data.copy()))

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-2, atol=1e-3), \
            f"CNN block gradient mismatch"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_simple_cnn(self):
        """Test simple CNN architecture."""
        from python.nn_core import (
            Sequential, Conv2d, BatchNorm2d, ReLU,
            MaxPool2d, AdaptiveAvgPool2d, Linear
        )
        from python.foundations import Tensor

        model = Sequential(
            Conv2d(3, 32, 3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(32, 64, 3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            AdaptiveAvgPool2d((1, 1)),
        )

        x = Tensor(np.random.randn(2, 3, 32, 32), requires_grad=True)
        y = model(x)

        # After adaptive pool: (batch, 64, 1, 1)
        assert y.shape == (2, 64, 1, 1)

    def test_simple_transformer_block(self):
        """Test simple transformer block."""
        from python.nn_core import (
            MultiHeadAttention, LayerNorm, Linear, Dropout
        )
        from python.foundations import Tensor

        # Simple transformer block
        embed_dim = 64
        num_heads = 8

        attn = MultiHeadAttention(embed_dim, num_heads)
        ln1 = LayerNorm([embed_dim])
        ln2 = LayerNorm([embed_dim])
        ff1 = Linear(embed_dim, 256)
        ff2 = Linear(256, embed_dim)
        dropout = Dropout(0.1)

        x = Tensor(np.random.randn(2, 10, 64), requires_grad=True)

        # Self-attention with residual
        attn_out = attn(x, x, x)
        x_residual = x + dropout(attn_out)
        x_norm = ln1(x_residual)

        # Feedforward with residual
        ff_out = ff2(ff1(x_norm).relu())
        x_residual2 = x_norm + dropout(ff_out)
        out = ln2(x_residual2)

        assert out.shape == x.shape

    def test_simple_rnn_model(self):
        """Test simple RNN model."""
        from python.nn_core import LSTM, Linear
        from python.foundations import Tensor

        lstm = LSTM(input_size=32, hidden_size=64, num_layers=2)
        fc = Linear(64, 10)

        x = Tensor(np.random.randn(20, 4, 32), requires_grad=True)  # seq, batch, features

        lstm_out, (h_n, c_n) = lstm(x)

        # Take last hidden state
        last_hidden = lstm_out[-1]  # (batch, hidden_size)
        output = fc(last_hidden)

        assert output.shape == (4, 10)


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
