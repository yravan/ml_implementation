"""
Comprehensive Tests for Foundations Module
==========================================

This test suite covers all components of the foundations module:
- Tensor class and basic operations
- Function classes (forward and backward)
- Automatic differentiation
- Gradient checking utilities

Each test verifies both forward computation and backward gradients.
"""

import numpy as np
import pytest
from typing import Callable, Tuple


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
def tensor_2d(seed):
    """Create a 2D tensor for testing."""
    from python.foundations import Tensor
    data = np.random.randn(4, 5).astype(np.float32)
    return Tensor(data, requires_grad=True)


@pytest.fixture
def tensor_3d(seed):
    """Create a 3D tensor for testing."""
    from python.foundations import Tensor
    data = np.random.randn(2, 3, 4).astype(np.float32)
    return Tensor(data, requires_grad=True)


# =============================================================================
# Tensor Class Tests
# =============================================================================

class TestTensor:
    """Test the Tensor class."""

    def test_tensor_creation_from_ndarray(self):
        """Test creating Tensor from NumPy array."""
        from python.foundations import Tensor
        data = np.array([1.0, 2.0, 3.0])
        t = Tensor(data)

        assert t.shape == (3,)
        assert np.allclose(t.data, data)
        assert t.requires_grad == False

    def test_tensor_creation_with_requires_grad(self):
        """Test creating Tensor with gradient tracking."""
        from python.foundations import Tensor
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = Tensor(data, requires_grad=True)

        assert t.requires_grad == True
        assert t.grad is None  # No gradient until backward

    def test_tensor_from_scalar(self):
        """Test creating Tensor from scalar."""
        from python.foundations import Tensor
        t = Tensor(5.0)

        assert t.shape == ()
        assert t.data == 5.0

    def test_tensor_from_list(self):
        """Test creating Tensor from Python list."""
        from python.foundations import Tensor
        t = Tensor([1.0, 2.0, 3.0])

        assert t.shape == (3,)
        assert np.allclose(t.data, [1.0, 2.0, 3.0])

    def test_tensor_shape_property(self):
        """Test shape property."""
        from python.foundations import Tensor
        t = Tensor(np.zeros((2, 3, 4)))

        assert t.shape == (2, 3, 4)

    def test_tensor_dtype(self):
        """Test data type handling."""
        from python.foundations import Tensor
        t = Tensor(np.array([1, 2, 3], dtype=np.float32))

        assert t.data.dtype == np.float32


# =============================================================================
# Basic Arithmetic Operations Tests
# =============================================================================

class TestAdd:
    """Test element-wise addition."""

    def test_add_forward(self):
        """Test Add forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)

        z = x + y

        assert np.allclose(z.data, [5.0, 7.0, 9.0])

    def test_add_backward(self):
        """Test Add backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)

        z = x + y
        loss = z.sum()
        loss.backward()

        # ∂L/∂x = ∂L/∂z * ∂z/∂x = 1 * 1 = 1
        assert np.allclose(x.grad, [1.0, 1.0, 1.0])
        assert np.allclose(y.grad, [1.0, 1.0, 1.0])

    def test_add_broadcast(self):
        """Test Add with broadcasting."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = Tensor(np.array([10.0, 20.0]), requires_grad=True)

        z = x + y
        loss = z.sum()
        loss.backward()

        expected = np.array([[11.0, 22.0], [13.0, 24.0]])
        assert np.allclose(z.data, expected)

        # Gradient for y should be summed over broadcast dimension
        assert np.allclose(y.grad, [2.0, 2.0])

    def test_add_scalar(self):
        """Test adding scalar to tensor."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        z = x + 5.0
        loss = z.sum()
        loss.backward()

        assert np.allclose(z.data, [6.0, 7.0, 8.0])
        assert np.allclose(x.grad, [1.0, 1.0, 1.0])


class TestMul:
    """Test element-wise multiplication."""

    def test_mul_forward(self):
        """Test Mul forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)

        z = x * y

        assert np.allclose(z.data, [2.0, 6.0, 12.0])

    def test_mul_backward(self):
        """Test Mul backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)

        z = x * y
        loss = z.sum()
        loss.backward()

        # ∂L/∂x = ∂L/∂z * y = 1 * y
        assert np.allclose(x.grad, [2.0, 3.0, 4.0])
        assert np.allclose(y.grad, [1.0, 2.0, 3.0])

    def test_mul_broadcast(self):
        """Test Mul with broadcasting."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = Tensor(np.array([2.0, 3.0]), requires_grad=True)

        z = x * y
        loss = z.sum()
        loss.backward()

        expected = np.array([[2.0, 6.0], [6.0, 12.0]])
        assert np.allclose(z.data, expected)

        # Gradient for y: sum over broadcast dimension
        assert np.allclose(y.grad, [1.0 + 3.0, 2.0 + 4.0])


class TestMatMul:
    """Test matrix multiplication."""

    def test_matmul_forward(self):
        """Test MatMul forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)

        z = x @ y

        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(z.data, expected)

    def test_matmul_backward(self):
        """Test MatMul backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)

        z = x @ y
        loss = z.sum()
        loss.backward()

        # ∂L/∂X = ∂L/∂Z @ Y^T
        # ∂L/∂Y = X^T @ ∂L/∂Z
        expected_dx = np.ones((2, 2)) @ y.data.T
        expected_dy = x.data.T @ np.ones((2, 2))

        assert np.allclose(x.grad, expected_dx)
        assert np.allclose(y.grad, expected_dy)

    def test_matmul_vector(self):
        """Test matrix-vector multiplication."""
        from python.foundations import Tensor
        A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)

        y = A @ x
        loss = y.sum()
        loss.backward()

        expected = np.array([5.0, 11.0])
        assert np.allclose(y.data, expected)


class TestPow:
    """Test power operation."""

    def test_pow_forward(self):
        """Test Pow forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        z = x ** 2

        assert np.allclose(z.data, [1.0, 4.0, 9.0])

    def test_pow_backward(self):
        """Test Pow backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        z = x ** 2
        loss = z.sum()
        loss.backward()

        # ∂L/∂x = 2 * x
        assert np.allclose(x.grad, [2.0, 4.0, 6.0])

    def test_pow_with_different_powers(self):
        """Test Pow with various power values."""
        from python.foundations import Tensor
        x = Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)

        z = x ** 3
        loss = z.sum()
        loss.backward()

        # ∂L/∂x = 3 * x^2
        expected_grad = 3 * np.array([2.0, 3.0, 4.0]) ** 2
        assert np.allclose(x.grad, expected_grad)


class TestSum:
    """Test sum reduction."""

    def test_sum_all(self):
        """Test Sum over all dimensions."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)

        z = x.sum()
        z.backward()

        assert z.data == 10.0
        assert np.allclose(x.grad, np.ones((2, 2)))

    def test_sum_axis(self):
        """Test Sum along specific axis."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)

        z = x.sum(axis=1)
        loss = z.sum()
        loss.backward()

        expected = np.array([6.0, 15.0])
        assert np.allclose(z.data, expected)
        assert np.allclose(x.grad, np.ones((2, 3)))

    def test_sum_keepdims(self):
        """Test Sum with keepdims=True."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)

        z = x.sum(axis=1, keepdims=True)
        loss = z.sum()
        loss.backward()

        assert z.shape == (2, 1)
        assert np.allclose(x.grad, np.ones((2, 2)))


class TestExp:
    """Test exponential operation."""

    def test_exp_forward(self):
        """Test Exp forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([0.0, 1.0, 2.0]), requires_grad=True)

        z = x.exp()

        expected = np.exp([0.0, 1.0, 2.0])
        assert np.allclose(z.data, expected)

    def test_exp_backward(self):
        """Test Exp backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([0.0, 1.0, 2.0]), requires_grad=True)

        z = x.exp()
        loss = z.sum()
        loss.backward()

        # ∂(exp(x))/∂x = exp(x)
        expected_grad = np.exp([0.0, 1.0, 2.0])
        assert np.allclose(x.grad, expected_grad)


class TestLog:
    """Test logarithm operation."""

    def test_log_forward(self):
        """Test Log forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, np.e]), requires_grad=True)

        z = x.log()

        expected = np.log([1.0, 2.0, np.e])
        assert np.allclose(z.data, expected)

    def test_log_backward(self):
        """Test Log backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 4.0]), requires_grad=True)

        z = x.log()
        loss = z.sum()
        loss.backward()

        # ∂(log(x))/∂x = 1/x
        expected_grad = 1.0 / np.array([1.0, 2.0, 4.0])
        assert np.allclose(x.grad, expected_grad)


# =============================================================================
# Shape Operations Tests
# =============================================================================

class TestReshape:
    """Test reshape operation."""

    def test_reshape_forward(self):
        """Test Reshape forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.arange(12).reshape(3, 4).astype(float), requires_grad=True)

        z = x.reshape(4, 3)

        assert z.shape == (4, 3)

    def test_reshape_backward(self):
        """Test Reshape backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.arange(12).reshape(3, 4).astype(float), requires_grad=True)

        z = x.reshape(4, 3)
        loss = z.sum()
        loss.backward()

        assert x.grad.shape == (3, 4)
        assert np.allclose(x.grad, np.ones((3, 4)))


class TestTranspose:
    """Test transpose operation."""

    def test_transpose_forward(self):
        """Test Transpose forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)

        z = x.T

        assert z.shape == (3, 2)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        assert np.allclose(z.data, expected)

    def test_transpose_backward(self):
        """Test Transpose backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)

        z = x.T
        loss = z.sum()
        loss.backward()

        assert x.grad.shape == (2, 3)
        assert np.allclose(x.grad, np.ones((2, 3)))


class TestMax:
    """Test max operation."""

    def test_max_all(self):
        """Test Max over all dimensions."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 5.0], [3.0, 2.0]]), requires_grad=True)

        z = x.max()
        z.backward()

        assert z.data == 5.0
        # Gradient flows only to max element
        expected_grad = np.array([[0.0, 1.0], [0.0, 0.0]])
        assert np.allclose(x.grad, expected_grad)

    def test_max_axis(self):
        """Test Max along specific axis."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]), requires_grad=True)

        z = x.max(axis=1)
        loss = z.sum()
        loss.backward()

        expected = np.array([5.0, 6.0])
        assert np.allclose(z.data, expected)


# =============================================================================
# Activation Functions Tests
# =============================================================================

class TestSigmoid:
    """Test Sigmoid activation."""

    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([0.0, 1.0, -1.0]), requires_grad=True)

        z = x.sigmoid()

        expected = 1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
        assert np.allclose(z.data, expected)

    def test_sigmoid_backward(self):
        """Test Sigmoid backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([0.0, 1.0, -1.0]), requires_grad=True)

        z = x.sigmoid()
        loss = z.sum()
        loss.backward()

        # ∂σ/∂x = σ(x) * (1 - σ(x))
        s = 1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
        expected_grad = s * (1 - s)
        assert np.allclose(x.grad, expected_grad)


class TestSoftmax:
    """Test Softmax activation."""

    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

        z = x.softmax(axis=-1)

        # Softmax should sum to 1
        assert np.allclose(z.data.sum(), 1.0)
        # Values should be in (0, 1)
        assert np.all(z.data > 0) and np.all(z.data < 1)

    def test_softmax_backward(self):
        """Test Softmax backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

        z = x.softmax(axis=-1)
        # Use a specific element for loss to get non-trivial gradient
        loss = z[0, 1]  # Second element
        loss.backward()

        # Gradient should not be all zeros
        assert not np.allclose(x.grad, 0)


# =============================================================================
# Mean and Variance Tests
# =============================================================================

class TestMean:
    """Test mean operation."""

    def test_mean_all(self):
        """Test Mean over all dimensions."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)

        z = x.mean()
        z.backward()

        assert z.data == 2.5
        # ∂mean/∂x = 1/n for each element
        assert np.allclose(x.grad, 0.25 * np.ones((2, 2)))

    def test_mean_axis(self):
        """Test Mean along specific axis."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)

        z = x.mean(axis=1)
        loss = z.sum()
        loss.backward()

        expected = np.array([2.0, 5.0])
        assert np.allclose(z.data, expected)
        assert np.allclose(x.grad, (1.0/3.0) * np.ones((2, 3)))


# =============================================================================
# Chain Rule Tests (Multiple Operations)
# =============================================================================

class TestChainRule:
    """Test gradient flow through multiple operations."""

    def test_simple_chain(self):
        """Test gradient flow through y = x^2 + x."""
        from python.foundations import Tensor
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)

        y = x ** 2 + x
        loss = y.sum()
        loss.backward()

        # ∂y/∂x = 2x + 1
        expected_grad = 2 * np.array([2.0, 3.0]) + 1
        assert np.allclose(x.grad, expected_grad)

    def test_longer_chain(self):
        """Test gradient flow through y = (x^2 + 1)^2."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)

        y = (x ** 2 + 1) ** 2
        loss = y.sum()
        loss.backward()

        # Let u = x^2 + 1
        # y = u^2
        # ∂y/∂x = ∂y/∂u * ∂u/∂x = 2u * 2x = 4x(x^2 + 1)
        x_np = np.array([1.0, 2.0])
        expected_grad = 4 * x_np * (x_np ** 2 + 1)
        assert np.allclose(x.grad, expected_grad)

    def test_tensor_used_multiple_times(self):
        """Test gradient accumulation when tensor is used twice."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        y = x * x  # x used twice
        loss = y.sum()
        loss.backward()

        # ∂(x*x)/∂x = 2x
        expected_grad = 2 * np.array([1.0, 2.0, 3.0])
        assert np.allclose(x.grad, expected_grad)

    def test_neural_network_like_computation(self):
        """Test gradient flow in a simple neural network."""
        from python.foundations import Tensor

        # Input and weight
        x = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        W = Tensor(np.array([[0.5, 0.3], [0.2, 0.4]]), requires_grad=True)

        # Forward: y = sigmoid(x @ W)
        y = (x @ W).sigmoid()
        loss = y.sum()
        loss.backward()

        # Gradients should exist and not be zero
        assert x.grad is not None
        assert W.grad is not None
        assert not np.allclose(W.grad, 0)


# =============================================================================
# Autograd Utility Tests
# =============================================================================

class TestGrad:
    """Test the grad function."""

    def test_grad_simple(self):
        """Test grad function with simple function."""
        from python.foundations import grad, Tensor

        def f(x):
            return (x ** 2).sum()

        grad_f = grad(f)
        x = Tensor(np.array([1.0, 2.0, 3.0]))

        dx = grad_f(x)

        expected = 2 * np.array([1.0, 2.0, 3.0])
        assert np.allclose(dx, expected)


class TestValueAndGrad:
    """Test the value_and_grad function."""

    def test_value_and_grad_simple(self):
        """Test value_and_grad with simple function."""
        from python.foundations import value_and_grad, Tensor

        def f(x):
            return (x ** 2).sum()

        val_grad_f = value_and_grad(f)
        x = Tensor(np.array([1.0, 2.0, 3.0]))

        val, dx = val_grad_f(x)

        assert np.isclose(val.data, 14.0)  # 1 + 4 + 9
        expected_grad = 2 * np.array([1.0, 2.0, 3.0])
        assert np.allclose(dx, expected_grad)


# =============================================================================
# Gradient Checking Tests
# =============================================================================

class TestGradientCheck:
    """Test gradient checking utilities."""

    def test_numerical_gradient_matches_analytical(self):
        """Test that numerical gradient matches analytical gradient."""
        from python.foundations import Tensor, numerical_gradient

        def f(x):
            return (x ** 2).sum()

        x = np.array([1.0, 2.0, 3.0])
        x = Tensor(x, requires_grad=True)

        # Numerical gradient
        num_grad = numerical_gradient(f, x)

        # Analytical gradient: 2x
        analytical_grad = 2 * x

        assert np.allclose(num_grad, analytical_grad.data, atol=1e-5)

    def test_gradcheck_passes(self):
        """Test that gradcheck passes for correct implementation."""
        from python.foundations import Tensor, gradcheck

        def f(x):
            y = (x ** 2).sum()
            return y

        x = np.array([1.0, 2.0, 3.0])
        x = Tensor(x, requires_grad=True)

        # This should not raise an assertion error
        result = gradcheck(f, (x,), eps=1e-5, atol=1e-4)
        assert result == True


# =============================================================================
# No Grad Context Tests
# =============================================================================

class TestNoGrad:
    """Test no_grad context manager."""

    def test_no_grad_disables_tracking(self):
        """Test that no_grad disables gradient tracking."""
        from python.foundations import Tensor, no_grad

        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        with no_grad():
            y = x ** 2
            # y should not track gradients
            assert y.requires_grad == False

    def test_no_grad_restores_state(self):
        """Test that no_grad restores state after exiting."""
        from python.foundations import Tensor, no_grad

        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        # Outside no_grad
        y1 = x * 2
        assert y1.requires_grad == True

        with no_grad():
            y2 = x * 2
            assert y2.requires_grad == False

        # After exiting no_grad
        y3 = x * 2
        assert y3.requires_grad == True


# =============================================================================
# Stack and Concat Tests
# =============================================================================

class TestStack:
    """Test stack operation."""

    def test_stack_forward(self):
        """Test Stack forward pass."""
        from python.foundations import Tensor, stack

        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)

        z = stack(x, y, axis=0)

        assert z.shape == (2, 3)
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.allclose(z.data, expected)


class TestConcat:
    """Test concatenate operation."""

    def test_concat_forward(self):
        """Test Concat forward pass."""
        from python.foundations import Tensor, concat

        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)

        z = concat(x, y, axis=0)

        assert z.shape == (4, 2)


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_tensor(self):
        """Test operations on zero tensor."""
        from python.foundations import Tensor
        x = Tensor(np.zeros((3, 3)), requires_grad=True)

        y = x ** 2
        loss = y.sum()
        loss.backward()

        assert np.allclose(y.data, 0)
        assert np.allclose(x.grad, 0)

    def test_negative_values(self):
        """Test operations on negative values."""
        from python.foundations import Tensor
        x = Tensor(np.array([-1.0, -2.0, -3.0]), requires_grad=True)

        y = x ** 2
        loss = y.sum()
        loss.backward()

        assert np.allclose(y.data, [1.0, 4.0, 9.0])
        assert np.allclose(x.grad, [-2.0, -4.0, -6.0])

    def test_large_tensor(self):
        """Test operations on larger tensor."""
        from python.foundations import Tensor
        x = Tensor(np.random.randn(100, 100), requires_grad=True)

        y = x * 2
        loss = y.sum()
        loss.backward()

        assert np.allclose(x.grad, 2 * np.ones((100, 100)))

    def test_scalar_output(self):
        """Test that backward works correctly for scalar output."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        y = (x ** 2).sum()  # Scalar output
        y.backward()

        assert y.shape == ()
        assert x.grad is not None


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of operations."""

    def test_exp_large_values(self):
        """Test exp with large values doesn't overflow."""
        from python.foundations import Tensor
        x = Tensor(np.array([100.0]), requires_grad=True)

        # This might overflow in naive implementation
        # A numerically stable softmax shifts by max
        y = x.exp()

        # Just check it doesn't crash
        assert np.isfinite(y.data).all() or np.isinf(y.data).all()

    def test_log_small_values(self):
        """Test log with small positive values."""
        from python.foundations import Tensor
        x = Tensor(np.array([1e-10, 1e-5, 1e-1]), requires_grad=True)

        y = x.log()
        loss = y.sum()
        loss.backward()

        assert np.isfinite(y.data).all()
        assert np.isfinite(x.grad).all()

    def test_softmax_stability(self):
        """Test softmax numerical stability with large values."""
        from python.foundations import Tensor
        # Large values that could cause overflow in naive exp
        x = Tensor(np.array([[100.0, 200.0, 300.0]]), requires_grad=True)

        y = x.softmax(axis=-1)

        # Should still be valid probabilities
        assert np.isfinite(y.data).all()
        assert np.allclose(y.data.sum(), 1.0)


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
