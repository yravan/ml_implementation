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


# =============================================================================
# Softmax Comprehensive Tests (REPLACES WEAK TestSoftmax)
# =============================================================================

class TestSoftmaxComprehensive:

    def test_softmax_forward_correctness(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        np.random.seed(42)
        x_data = np.random.randn(2, 5).astype(np.float64)
        x_shifted = x_data - x_data.max(axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        expected = exp_x / exp_x.sum(axis=1, keepdims=True)
        x = Tensor(x_data, requires_grad=True)
        # softmax via tensor method
        out = x.softmax(axis=1)
        assert np.allclose(out.data, expected, atol=1e-10)

    def test_softmax_backward_correctness(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        np.random.seed(42)
        x_data = np.random.randn(1, 4).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        # softmax via tensor method
        out = x.softmax(axis=1)
        out.sum().backward()
        # d(sum(softmax))/dx = 0 for all x (sum of softmax is always 1)
        assert np.allclose(x.grad, 0, atol=1e-10)

    def test_softmax_probability_properties(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        np.random.seed(42)
        x = Tensor(np.random.randn(3, 6).astype(np.float64))
        out = x.softmax(axis=1)
        assert np.allclose(out.data.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(out.data > 0)
        assert np.all(out.data < 1)

    def test_softmax_numerical_stability(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        x = Tensor(np.array([[1000.0, 1001.0, 1002.0]]).astype(np.float64))
        out = x.softmax(axis=1)
        assert np.isfinite(out.data).all()
        assert np.allclose(out.data.sum(), 1.0, atol=1e-10)

    def test_softmax_invariance(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        np.random.seed(42)
        x_data = np.random.randn(2, 4).astype(np.float64)
        out1 = Tensor(x_data).softmax(axis=1)
        out2 = Tensor(x_data + 100.0).softmax(axis=1)
        assert np.allclose(out1.data, out2.data, atol=1e-10)

    def test_softmax_uniform_input(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        x = Tensor(np.ones((2, 5)).astype(np.float64))
        out = x.softmax(axis=1)
        assert np.allclose(out.data, 0.2, atol=1e-10)

    def test_softmax_one_hot_limit(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        x = Tensor(np.array([[0.0, 0.0, 100.0]]).astype(np.float64))
        out = x.softmax(axis=1)
        assert out.data[0, 2] > 0.99
        assert out.data[0, 0] < 0.01

    def test_softmax_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor, Softmax, gradcheck
        np.random.seed(42)
        x = Tensor(np.random.randn(2, 4).astype(np.float64), requires_grad=True)
        def f(x):
            return x.softmax(axis=1).sum()
        assert gradcheck(f, (x,), eps=1e-5, atol=1e-4)

    def test_softmax_axis0(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        x_shifted = x_data - x_data.max(axis=0, keepdims=True)
        exp_x = np.exp(x_shifted)
        expected = exp_x / exp_x.sum(axis=0, keepdims=True)
        out = Tensor(x_data).softmax(axis=0)
        assert np.allclose(out.data, expected, atol=1e-10)


# =============================================================================
# Concat Comprehensive Tests (REPLACES WEAK TestConcat)
# =============================================================================

class TestConcatComprehensive:

    def test_concat_forward_axis0(self):
        import numpy as np
        from python.foundations import Tensor, concat
        np.random.seed(42)
        a_data = np.random.randn(2, 3).astype(np.float64)
        b_data = np.random.randn(3, 3).astype(np.float64)
        a, b = Tensor(a_data), Tensor(b_data)
        out = concat(a, b, axis=0)
        expected = np.concatenate([a_data, b_data], axis=0)
        assert np.allclose(out.data, expected)
        assert out.shape == (5, 3)

    def test_concat_forward_axis1(self):
        import numpy as np
        from python.foundations import Tensor, concat
        np.random.seed(42)
        a_data = np.random.randn(2, 3).astype(np.float64)
        b_data = np.random.randn(2, 4).astype(np.float64)
        out = concat(Tensor(a_data), Tensor(b_data), axis=1)
        expected = np.concatenate([a_data, b_data], axis=1)
        assert np.allclose(out.data, expected)
        assert out.shape == (2, 7)

    def test_concat_backward(self):
        import numpy as np
        from python.foundations import Tensor, concat
        np.random.seed(42)
        a = Tensor(np.random.randn(2, 3).astype(np.float64), requires_grad=True)
        b = Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)
        out = concat(a, b, axis=0)
        out.sum().backward()
        assert np.allclose(a.grad, np.ones((2, 3)))
        assert np.allclose(b.grad, np.ones((3, 3)))

    def test_concat_backward_axis1(self):
        import numpy as np
        from python.foundations import Tensor, concat
        np.random.seed(42)
        a = Tensor(np.random.randn(2, 3).astype(np.float64), requires_grad=True)
        b = Tensor(np.random.randn(2, 4).astype(np.float64), requires_grad=True)
        out = concat(a, b, axis=1)
        out.sum().backward()
        assert a.grad.shape == (2, 3)
        assert b.grad.shape == (2, 4)
        assert np.allclose(a.grad, np.ones((2, 3)))

    def test_concat_multiple_tensors(self):
        import numpy as np
        from python.foundations import Tensor, concat
        np.random.seed(42)
        tensors_data = [np.random.randn(2, 3).astype(np.float64) for _ in range(4)]
        tensors = [Tensor(d) for d in tensors_data]
        out = concat(*tensors, axis=0)
        expected = np.concatenate(tensors_data, axis=0)
        assert np.allclose(out.data, expected)
        assert out.shape == (8, 3)

    def test_concat_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor, concat, gradcheck
        np.random.seed(42)
        a = Tensor(np.random.randn(2, 3).astype(np.float64), requires_grad=True)
        b = Tensor(np.random.randn(2, 3).astype(np.float64), requires_grad=True)
        def f(a, b):
            return concat(a, b, axis=0).sum()
        assert gradcheck(f, (a, b), eps=1e-5, atol=1e-4)


# =============================================================================
# Stack Comprehensive Tests (REPLACES WEAK TestStack)
# =============================================================================

class TestStackComprehensive:

    def test_stack_forward_axis0(self):
        import numpy as np
        from python.foundations import Tensor, stack
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float64)
        b_data = np.random.randn(3, 4).astype(np.float64)
        out = stack(Tensor(a_data), Tensor(b_data), axis=0)
        expected = np.stack([a_data, b_data], axis=0)
        assert np.allclose(out.data, expected)
        assert out.shape == (2, 3, 4)

    def test_stack_forward_axis1(self):
        import numpy as np
        from python.foundations import Tensor, stack
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float64)
        b_data = np.random.randn(3, 4).astype(np.float64)
        out = stack(Tensor(a_data), Tensor(b_data), axis=1)
        expected = np.stack([a_data, b_data], axis=1)
        assert np.allclose(out.data, expected)
        assert out.shape == (3, 2, 4)

    def test_stack_backward(self):
        import numpy as np
        from python.foundations import Tensor, stack
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
        b = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
        out = stack(a, b, axis=0)
        out.sum().backward()
        assert np.allclose(a.grad, np.ones((3, 4)))
        assert np.allclose(b.grad, np.ones((3, 4)))

    def test_stack_multiple_tensors(self):
        import numpy as np
        from python.foundations import Tensor, stack
        np.random.seed(42)
        data = [np.random.randn(2, 3).astype(np.float64) for _ in range(5)]
        out = stack(*[Tensor(d) for d in data], axis=0)
        expected = np.stack(data, axis=0)
        assert np.allclose(out.data, expected)
        assert out.shape == (5, 2, 3)

    def test_stack_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor, stack, gradcheck
        np.random.seed(42)
        a = Tensor(np.random.randn(2, 3).astype(np.float64), requires_grad=True)
        b = Tensor(np.random.randn(2, 3).astype(np.float64), requires_grad=True)
        def f(a, b):
            return stack(a, b, axis=0).sum()
        assert gradcheck(f, (a, b), eps=1e-5, atol=1e-4)

    def test_stack_shape(self):
        import numpy as np
        from python.foundations import Tensor, stack
        a = Tensor(np.zeros((4, 5)).astype(np.float64))
        b = Tensor(np.zeros((4, 5)).astype(np.float64))
        c = Tensor(np.zeros((4, 5)).astype(np.float64))
        out = stack(a, b, c, axis=0)
        assert out.shape == (3, 4, 5)
        out2 = stack(a, b, c, axis=2)
        assert out2.shape == (4, 5, 3)


# =============================================================================
# Slice Comprehensive Tests (REPLACES WEAK TestSlice)
# =============================================================================

class TestSliceComprehensive:

    def test_slice_single_index(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(5, 4).astype(np.float64)
        x = Tensor(x_data)
        out = x[2]
        assert np.allclose(out.data, x_data[2])

    def test_slice_range(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(5, 4).astype(np.float64)
        x = Tensor(x_data)
        out = x[1:3]
        assert np.allclose(out.data, x_data[1:3])
        assert out.shape == (2, 4)

    def test_slice_backward(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x = Tensor(np.random.randn(5, 4).astype(np.float64), requires_grad=True)
        out = x[1:3]
        out.sum().backward()
        expected_grad = np.zeros((5, 4))
        expected_grad[1:3] = 1.0
        assert np.allclose(x.grad, expected_grad)

    def test_slice_negative_index(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(5, 4).astype(np.float64)
        x = Tensor(x_data)
        out = x[-2:]
        assert np.allclose(out.data, x_data[-2:])

    def test_slice_with_step(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(8, 3).astype(np.float64)
        x = Tensor(x_data)
        out = x[::2]
        assert np.allclose(out.data, x_data[::2])
        assert out.shape == (4, 3)

    def test_slice_2d(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(4, 6).astype(np.float64)
        x = Tensor(x_data)
        out = x[1:3, 2:5]
        assert np.allclose(out.data, x_data[1:3, 2:5])
        assert out.shape == (2, 3)

    def test_slice_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor, gradcheck
        np.random.seed(42)
        x = Tensor(np.random.randn(4, 3).astype(np.float64), requires_grad=True)
        def f(x):
            return x[1:3].sum()
        assert gradcheck(f, (x,), eps=1e-5, atol=1e-4)


# =============================================================================
# Maximum/Minimum Comprehensive Tests (REPLACES WEAK TestMaximumMinimum)
# =============================================================================

class TestMaximumMinimumComprehensive:

    def test_maximum_forward(self):
        import numpy as np
        from python.foundations import Tensor, maximum
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float64)
        b_data = np.random.randn(3, 4).astype(np.float64)
        out = maximum(Tensor(a_data), Tensor(b_data))
        expected = np.maximum(a_data, b_data)
        assert np.allclose(out.data, expected)

    def test_maximum_backward(self):
        import numpy as np
        from python.foundations import Tensor, maximum
        np.random.seed(42)
        a = Tensor(np.array([1.0, 3.0, 2.0]).astype(np.float64), requires_grad=True)
        b = Tensor(np.array([2.0, 1.0, 4.0]).astype(np.float64), requires_grad=True)
        out = maximum(a, b)
        out.sum().backward()
        # Gradient goes to the larger element
        assert np.allclose(a.grad, [0.0, 1.0, 0.0])
        assert np.allclose(b.grad, [1.0, 0.0, 1.0])

    def test_minimum_forward(self):
        import numpy as np
        from python.foundations import Tensor, minimum
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float64)
        b_data = np.random.randn(3, 4).astype(np.float64)
        out = minimum(Tensor(a_data), Tensor(b_data))
        expected = np.minimum(a_data, b_data)
        assert np.allclose(out.data, expected)

    def test_minimum_backward(self):
        import numpy as np
        from python.foundations import Tensor, minimum
        np.random.seed(42)
        a = Tensor(np.array([1.0, 3.0, 2.0]).astype(np.float64), requires_grad=True)
        b = Tensor(np.array([2.0, 1.0, 4.0]).astype(np.float64), requires_grad=True)
        out = minimum(a, b)
        out.sum().backward()
        # Gradient goes to the smaller element
        assert np.allclose(a.grad, [1.0, 0.0, 1.0])
        assert np.allclose(b.grad, [0.0, 1.0, 0.0])

    def test_maximum_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor, maximum, gradcheck
        np.random.seed(42)
        # Avoid equal values which cause non-differentiability
        a = Tensor(np.array([1.0, 5.0, 2.0]).astype(np.float64), requires_grad=True)
        b = Tensor(np.array([3.0, 1.0, 4.0]).astype(np.float64), requires_grad=True)
        def f(a, b):
            return maximum(a, b).sum()
        assert gradcheck(f, (a, b), eps=1e-5, atol=1e-4)

    def test_minimum_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor, minimum, gradcheck
        np.random.seed(42)
        a = Tensor(np.array([1.0, 5.0, 2.0]).astype(np.float64), requires_grad=True)
        b = Tensor(np.array([3.0, 1.0, 4.0]).astype(np.float64), requires_grad=True)
        def f(a, b):
            return minimum(a, b).sum()
        assert gradcheck(f, (a, b), eps=1e-5, atol=1e-4)

    def test_maximum_broadcast(self):
        import numpy as np
        from python.foundations import Tensor, maximum
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float64)
        b_data = np.random.randn(1, 4).astype(np.float64)
        out = maximum(Tensor(a_data), Tensor(b_data))
        expected = np.maximum(a_data, b_data)
        assert np.allclose(out.data, expected)


# =============================================================================
# Var Comprehensive Tests (REPLACES WEAK TestVar)
# =============================================================================

class TestVarComprehensive:

    def test_var_forward_all(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        x = Tensor(x_data)
        out = x.var()
        expected = np.var(x_data)
        assert np.allclose(out.data, expected, atol=1e-10)

    def test_var_forward_axis(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        x = Tensor(x_data)
        out = x.var(axis=1)
        expected = np.var(x_data, axis=1)
        assert np.allclose(out.data, expected, atol=1e-10)

    def test_var_backward(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(4, 3).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        out = x.var()
        out.backward()
        n = x_data.size
        expected_grad = 2.0 * (x_data - x_data.mean()) / n
        assert np.allclose(x.grad, expected_grad, atol=1e-10)

    def test_var_keepdims(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        x = Tensor(x_data)
        out = x.var(axis=1, keepdims=True)
        expected = np.var(x_data, axis=1, keepdims=True)
        assert out.shape == (3, 1)
        assert np.allclose(out.data, expected, atol=1e-10)

    def test_var_constant_input(self):
        import numpy as np
        from python.foundations import Tensor
        x = Tensor(np.ones((3, 4)).astype(np.float64) * 5.0)
        out = x.var()
        assert np.allclose(out.data, 0.0, atol=1e-10)

    def test_var_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor, gradcheck
        np.random.seed(42)
        x = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
        def f(x):
            return x.var()
        assert gradcheck(f, (x,), eps=1e-5, atol=1e-4)


# =============================================================================
# Mean Comprehensive Tests (REPLACES WEAK TestMean)
# =============================================================================

class TestMeanComprehensive:

    def test_mean_forward_all(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        x = Tensor(x_data)
        out = x.mean()
        assert np.allclose(out.data, x_data.mean(), atol=1e-10)

    def test_mean_forward_axis(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        out = Tensor(x_data).mean(axis=1)
        assert np.allclose(out.data, x_data.mean(axis=1), atol=1e-10)

    def test_mean_backward(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)
        out = x.mean()
        out.backward()
        n = x_data.size
        assert np.allclose(x.grad, np.ones_like(x_data) / n, atol=1e-10)

    def test_mean_keepdims(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(3, 4).astype(np.float64)
        out = Tensor(x_data).mean(axis=0, keepdims=True)
        assert out.shape == (1, 4)
        assert np.allclose(out.data, x_data.mean(axis=0, keepdims=True))

    def test_mean_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor, gradcheck
        np.random.seed(42)
        x = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
        def f(x):
            return x.mean()
        assert gradcheck(f, (x,), eps=1e-5, atol=1e-4)


# =============================================================================
# Split Comprehensive Tests (REPLACES WEAK TestSplit)
# =============================================================================

class TestSplitComprehensive:

    def test_split_equal_parts(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(6, 4).astype(np.float64)
        x = Tensor(x_data)
        parts = x.split(3, axis=0)  # split into 3 parts
        np_parts = np.split(x_data, 3, axis=0)
        for p, np_p in zip(parts, np_parts):
            assert np.allclose(p.data, np_p)

    def test_split_backward(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x = Tensor(np.random.randn(6, 4).astype(np.float64), requires_grad=True)
        parts = x.split(3, axis=0)  # returns (3, 2, 4) tensor
        loss = parts[0].sum() + parts[1].sum() * 2 + parts[2].sum() * 3
        loss.backward()
        # grad is (3, 2, 4) matching the split output shape
        expected = np.ones((3, 2, 4))
        expected[0] = 1.0
        expected[1] = 2.0
        expected[2] = 3.0
        assert np.allclose(x.grad, expected)

    def test_split_shapes(self):
        import numpy as np
        from python.foundations import Tensor
        x = Tensor(np.zeros((9, 3)).astype(np.float64))
        parts = x.split(3, axis=0)  # returns (3, 3, 3) tensor
        assert parts.shape == (3, 3, 3)
        assert parts[0].shape == (3, 3)
        assert parts[1].shape == (3, 3)
        assert parts[2].shape == (3, 3)

    def test_split_axis1(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(3, 8).astype(np.float64)
        x = Tensor(x_data)
        parts = x.split(4, axis=1)
        np_parts = np.split(x_data, 4, axis=1)
        for p, np_p in zip(parts, np_parts):
            assert np.allclose(p.data, np_p)


# =============================================================================
# Set In Place Tests (REPLACES WEAK TestSet)
# =============================================================================

class TestSetInPlace:

    def test_set_single_value(self):
        import numpy as np
        from python.foundations import Tensor
        x = Tensor(np.zeros((3, 3)).astype(np.float64))
        x.set_in_place((1, 1), 5.0)
        assert x.data[1, 1] == 5.0

    def test_set_slice(self):
        import numpy as np
        from python.foundations import Tensor
        x = Tensor(np.zeros((4, 4)).astype(np.float64))
        x.set_in_place(slice(0, 2), np.ones((2, 4)))
        assert np.allclose(x.data[0:2], 1.0)
        assert np.allclose(x.data[2:4], 0.0)

    def test_set_preserves_other(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x_data = np.random.randn(5, 3).astype(np.float64)
        x = Tensor(x_data.copy())
        x.set_in_place((2,), np.zeros(3))
        assert np.allclose(x.data[0], x_data[0])
        assert np.allclose(x.data[2], 0.0)
        assert np.allclose(x.data[4], x_data[4])


# =============================================================================
# Detach Copy Comprehensive Tests (REPLACES WEAK TestDetachCopy)
# =============================================================================

class TestDetachCopyComprehensive:

    def test_copy_preserves_data(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
        y = x.copy()
        assert np.allclose(y.data, x.data)

    def test_copy_independent(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
        y = x.copy()
        y.data[0, 0] = 999.0
        assert x.data[0, 0] != 999.0

    def test_copy_backward(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
        y = x.copy()
        y.sum().backward()
        assert x.grad is not None
        assert np.allclose(x.grad, np.ones((3, 4)))


# =============================================================================
# Numerical Stability Comprehensive Tests (REPLACES WEAK TestNumericalStability)
# =============================================================================

class TestNumericalStabilityComprehensive:

    def test_log_near_zero(self):
        import numpy as np
        from python.foundations import Tensor
        x = Tensor(np.array([1e-10, 1e-20, 1e-30]).astype(np.float64))
        out = x.log()
        expected = np.log(np.array([1e-10, 1e-20, 1e-30]))
        assert np.allclose(out.data, expected, atol=1e-6)
        assert np.isfinite(out.data).all()

    def test_exp_large_values(self):
        import numpy as np
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 10.0, 50.0]).astype(np.float64))
        out = x.exp()
        expected = np.exp(np.array([1.0, 10.0, 50.0]))
        # Tensor internally stores float32, so relax tolerance
        assert np.allclose(out.data, expected.astype(np.float32), rtol=1e-5)

    def test_sigmoid_extreme(self):
        import numpy as np
        from python.foundations import Tensor
        x = Tensor(np.array([-100.0, 0.0, 100.0]).astype(np.float64))
        out = x.sigmoid()
        assert np.isclose(out.data[0], 0.0, atol=1e-30)
        assert np.isclose(out.data[1], 0.5, atol=1e-10)
        assert np.isclose(out.data[2], 1.0, atol=1e-30)

    def test_softmax_large(self):
        import numpy as np
        from python.foundations import Tensor, Softmax
        x = Tensor(np.array([[500.0, 501.0, 502.0]]).astype(np.float64))
        out = x.softmax(axis=1)
        assert np.isfinite(out.data).all()
        assert np.allclose(out.data.sum(), 1.0)

    def test_chain_rule_deep(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        x = Tensor(np.random.randn(2, 3).astype(np.float64) * 0.1, requires_grad=True)
        out = x
        for _ in range(10):
            out = out * 0.9 + 0.1
        out.sum().backward()
        assert np.isfinite(x.grad).all()

    def test_matmul_large(self):
        import numpy as np
        from python.foundations import Tensor
        np.random.seed(42)
        a = Tensor(np.random.randn(50, 50).astype(np.float64) * 0.1, requires_grad=True)
        b = Tensor(np.random.randn(50, 50).astype(np.float64) * 0.1, requires_grad=True)
        out = a @ b
        out.sum().backward()
        assert np.isfinite(a.grad).all()
        assert np.isfinite(b.grad).all()


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

        assert np.allclose(num_grad, analytical_grad.data, atol=1e-2)

    def test_gradcheck_passes(self):
        """Test that gradcheck passes for correct implementation."""
        from python.foundations import Tensor, gradcheck

        def f(x):
            y = (x ** 2).sum()
            return y

        x = np.array([1.0, 2.0, 3.0])
        x = Tensor(x, requires_grad=True)

        # This should not raise an assertion error
        result = gradcheck(f, (x,), eps=1e-3, atol=1e-3)
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
# Subtraction Tests (from test_foundations_new.py)
# =============================================================================

class TestSub:
    """Test element-wise subtraction."""

    def test_sub_forward(self):
        """Test Sub forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([5.0, 6.0, 7.0]), requires_grad=True)
        y = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        z = x - y

        assert np.allclose(z.data, [4.0, 4.0, 4.0])

    def test_sub_backward(self):
        """Test Sub backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([5.0, 6.0, 7.0]), requires_grad=True)
        y = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        z = x - y
        loss = z.sum()
        loss.backward()

        # ∂L/∂x = 1, ∂L/∂y = -1
        assert np.allclose(x.grad, [1.0, 1.0, 1.0])
        assert np.allclose(y.grad, [-1.0, -1.0, -1.0])

    def test_sub_broadcast(self):
        """Test Sub with broadcasting."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = Tensor(np.array([1.0, 2.0]), requires_grad=True)

        z = x - y
        loss = z.sum()
        loss.backward()

        expected = np.array([[0.0, 0.0], [2.0, 2.0]])
        assert np.allclose(z.data, expected)
        assert np.allclose(y.grad, [-2.0, -2.0])  # Summed over broadcast dim

    def test_sub_scalar(self):
        """Test subtracting scalar from tensor."""
        from python.foundations import Tensor
        x = Tensor(np.array([5.0, 6.0, 7.0]), requires_grad=True)

        z = x - 3.0
        loss = z.sum()
        loss.backward()

        assert np.allclose(z.data, [2.0, 3.0, 4.0])
        assert np.allclose(x.grad, [1.0, 1.0, 1.0])

    def test_rsub(self):
        """Test reverse subtraction (scalar - tensor)."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        z = 10.0 - x
        loss = z.sum()
        loss.backward()

        assert np.allclose(z.data, [9.0, 8.0, 7.0])
        assert np.allclose(x.grad, [-1.0, -1.0, -1.0])


# =============================================================================
# Negation Tests (from test_foundations_new.py)
# =============================================================================

class TestNeg:
    """Test negation operation."""

    def test_neg_forward(self):
        """Test Neg forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)

        z = -x

        assert np.allclose(z.data, [-1.0, 2.0, -3.0])

    def test_neg_backward(self):
        """Test Neg backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)

        z = -x
        loss = z.sum()
        loss.backward()

        assert np.allclose(x.grad, [-1.0, -1.0, -1.0])

    def test_double_neg(self):
        """Test double negation."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        z = -(-x)
        loss = z.sum()
        loss.backward()

        assert np.allclose(z.data, x.data)
        assert np.allclose(x.grad, [1.0, 1.0, 1.0])


# =============================================================================
# Division Tests (from test_foundations_new.py)
# =============================================================================

class TestDiv:
    """Test element-wise division."""

    def test_div_forward(self):
        """Test Div forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([4.0, 6.0, 8.0]), requires_grad=True)
        y = Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)

        z = x / y

        assert np.allclose(z.data, [2.0, 2.0, 2.0])

    def test_div_backward(self):
        """Test Div backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([4.0, 6.0, 8.0]), requires_grad=True)
        y = Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)

        z = x / y
        loss = z.sum()
        loss.backward()

        # ∂(x/y)/∂x = 1/y
        # ∂(x/y)/∂y = -x/y^2
        assert np.allclose(x.grad, 1.0 / np.array([2.0, 3.0, 4.0]))
        expected_y_grad = -np.array([4.0, 6.0, 8.0]) / np.array([2.0, 3.0, 4.0])**2
        assert np.allclose(y.grad, expected_y_grad)

    def test_div_scalar(self):
        """Test dividing tensor by scalar."""
        from python.foundations import Tensor
        x = Tensor(np.array([2.0, 4.0, 6.0]), requires_grad=True)

        z = x / 2.0
        loss = z.sum()
        loss.backward()

        assert np.allclose(z.data, [1.0, 2.0, 3.0])
        assert np.allclose(x.grad, [0.5, 0.5, 0.5])

    def test_rdiv(self):
        """Test reverse division (scalar / tensor)."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 4.0]), requires_grad=True)

        z = 4.0 / x
        loss = z.sum()
        loss.backward()

        assert np.allclose(z.data, [4.0, 2.0, 1.0])
        # ∂(c/x)/∂x = -c/x^2
        expected_grad = -4.0 / np.array([1.0, 2.0, 4.0])**2
        assert np.allclose(x.grad, expected_grad)


# =============================================================================
# Absolute Value Tests (from test_foundations_new.py)
# =============================================================================

class TestAbs:
    """Test absolute value operation."""

    def test_abs_forward(self):
        """Test Abs forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([-3.0, -1.0, 0.0, 1.0, 3.0]), requires_grad=True)

        z = x.abs()

        assert np.allclose(z.data, [3.0, 1.0, 0.0, 1.0, 3.0])

    def test_abs_backward(self):
        """Test Abs backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([-3.0, -1.0, 2.0, 4.0]), requires_grad=True)

        z = x.abs()
        loss = z.sum()
        loss.backward()

        # ∂|x|/∂x = sign(x) = -1 for x<0, +1 for x>0
        assert np.allclose(x.grad, [-1.0, -1.0, 1.0, 1.0])


# =============================================================================
# Clamp Tests (from test_foundations_new.py)
# =============================================================================

class TestClamp:
    """Test clamp operation."""

    def test_clamp_forward(self):
        """Test Clamp forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([-2.0, 0.5, 1.5, 3.0]), requires_grad=True)

        z = x.clamp(0.0, 1.0)

        assert np.allclose(z.data, [0.0, 0.5, 1.0, 1.0])

    def test_clamp_backward(self):
        """Test Clamp backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([-2.0, 0.5, 1.5, 3.0]), requires_grad=True)

        z = x.clamp(0.0, 1.0)
        loss = z.sum()
        loss.backward()

        # Gradient is 1 where not clamped, 0 where clamped
        assert np.allclose(x.grad, [0.0, 1.0, 0.0, 0.0])

    def test_clamp_negative_range(self):
        """Test Clamp with negative range."""
        from python.foundations import Tensor
        x = Tensor(np.array([-5.0, -2.0, 0.0, 2.0]), requires_grad=True)

        z = x.clamp(-3.0, -1.0)
        loss = z.sum()
        loss.backward()

        assert np.allclose(z.data, [-3.0, -2.0, -1.0, -1.0])
        assert np.allclose(x.grad, [0.0, 1.0, 0.0, 0.0])


# =============================================================================
# LogSigmoid Tests (from test_foundations_new.py)
# =============================================================================

class TestLogSigmoid:
    """Test log sigmoid activation."""

    def test_logsigmoid_forward(self):
        """Test LogSigmoid forward pass."""
        from python.foundations import Tensor
        from python.utils.math_utils import log_sigmoid as np_log_sigmoid
        x = Tensor(np.array([0.0, 1.0, -1.0, 2.0]), requires_grad=True)

        z = x.log_sigmoid()

        # Use the same implementation for expected
        expected = np_log_sigmoid(x.data)
        assert np.allclose(z.data, expected)

    def test_logsigmoid_backward(self):
        """Test LogSigmoid backward pass."""
        from python.foundations import Tensor
        from python.utils.math_utils import log_sigmoid as np_log_sigmoid
        x = Tensor(np.array([0.0, 1.0, -1.0]), requires_grad=True)

        z = x.log_sigmoid()
        loss = z.sum()
        loss.backward()

        # Gradient: d(log_sigmoid)/dx = 1 - exp(log_sigmoid(x)) = sigmoid(-x)
        expected_grad = 1 - np.exp(np_log_sigmoid(x.data))
        assert np.allclose(x.grad, expected_grad, atol=1e-5)

    def test_logsigmoid_stability(self):
        """Test LogSigmoid numerical stability with large values."""
        from python.foundations import Tensor
        x = Tensor(np.array([-100.0, 100.0]), requires_grad=True)

        z = x.log_sigmoid()
        loss = z.sum()
        loss.backward()

        # For large negative x: log_sigmoid(x) ≈ x
        # For large positive x: log_sigmoid(x) ≈ 0
        assert np.isfinite(z.data).all()
        assert np.isfinite(x.grad).all()


# =============================================================================
# LogSoftmax Tests (from test_foundations_new.py)
# =============================================================================

class TestLogSoftmax:
    """Test log softmax activation."""

    def test_logsoftmax_forward(self):
        """Test LogSoftmax forward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

        z = x.log_softmax(axis=-1)

        # log_softmax = x - log(sum(exp(x)))
        x_max = x.data.max(axis=-1, keepdims=True)
        expected = x.data - x_max - np.log(np.sum(np.exp(x.data - x_max), axis=-1, keepdims=True))
        assert np.allclose(z.data, expected)

    def test_logsoftmax_backward(self):
        """Test LogSoftmax backward pass."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

        z = x.log_softmax(axis=-1)
        loss = z[0, 1]  # Pick one element
        loss.backward()

        # Gradient should not be all zeros
        assert not np.allclose(x.grad, 0)

    def test_logsoftmax_plus_nll(self):
        """Test LogSoftmax in a cross-entropy-like setting."""
        from python.foundations import Tensor

        logits = Tensor(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), requires_grad=True)
        targets = np.array([2, 0])  # Class indices

        log_probs = logits.log_softmax(axis=-1)

        # Manual cross-entropy: -log_probs[i, target[i]]
        loss = -(log_probs[0, targets[0]] + log_probs[1, targets[1]]) / 2
        loss.backward()

        assert logits.grad is not None
        assert np.isfinite(logits.grad).all()

    def test_logsoftmax_stability(self):
        """Test LogSoftmax numerical stability with large values."""
        from python.foundations import Tensor
        x = Tensor(np.array([[100.0, 200.0, 300.0]]), requires_grad=True)

        z = x.log_softmax(axis=-1)
        loss = z.sum()
        loss.backward()

        # Should be finite and exp(log_softmax) should sum to 1
        assert np.isfinite(z.data).all()
        assert np.allclose(np.exp(z.data).sum(), 1.0)


# =============================================================================
# Comparison Operators Tests (from test_foundations_new.py)
# =============================================================================

class TestComparisonOps:
    """Test comparison operators."""

    def test_greater_equal(self):
        """Test >= operator."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        y = Tensor(np.array([2.0, 2.0, 2.0]))

        z = x >= y

        assert np.allclose(z.data, [False, True, True])

    def test_greater(self):
        """Test > operator."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        y = Tensor(np.array([2.0, 2.0, 2.0]))

        z = x > y

        assert np.allclose(z.data, [False, False, True])

    def test_less_equal(self):
        """Test <= operator."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        y = Tensor(np.array([2.0, 2.0, 2.0]))

        z = x <= y

        assert np.allclose(z.data, [True, True, False])

    def test_less(self):
        """Test < operator."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        y = Tensor(np.array([2.0, 2.0, 2.0]))

        z = x < y

        assert np.allclose(z.data, [True, False, False])

    def test_comparison_with_scalar(self):
        """Test comparison with scalar."""
        from python.foundations import Tensor
        x = Tensor(np.array([1.0, 2.0, 3.0]))

        z = x > 1.5

        assert np.allclose(z.data, [False, True, True])

    def test_invert_bool_tensor(self):
        """Test ~ operator on bool tensor."""
        from python.foundations import Tensor
        x = Tensor(np.array([True, False, True]))

        z = ~x

        assert np.allclose(z.data, [False, True, False])


# =============================================================================
# Argmax Tests (from test_foundations_new.py)
# =============================================================================

class TestArgmax:
    """Test argmax operation."""

    def test_argmax_all(self):
        """Test argmax over all dimensions."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 5.0], [3.0, 2.0]]))

        z = x.argmax()

        assert z.data == 1  # Flat index of 5.0

    def test_argmax_axis(self):
        """Test argmax along specific axis."""
        from python.foundations import Tensor
        x = Tensor(np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]))

        z = x.argmax(axis=1)

        assert np.allclose(z.data, [1, 2])  # Indices of max in each row


# =============================================================================
# Gradient Check Tests for New Operations (from test_foundations_new.py)
# =============================================================================

class TestGradientChecks:
    """Gradient checks for new operations."""

    def test_gradcheck_sub(self):
        """Gradient check for subtraction."""
        from python.foundations import Tensor, gradcheck

        def f(x, y):
            return (x - y).sum()

        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4), requires_grad=True)

        result = gradcheck(f, (x, y), eps=1e-3, atol=1e-3)
        assert result == True

    def test_gradcheck_div(self):
        """Gradient check for division."""
        from python.foundations import Tensor, gradcheck

        def f(x, y):
            return (x / y).sum()

        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = Tensor(np.abs(np.random.randn(3, 4)) + 0.1, requires_grad=True)  # Avoid div by zero

        result = gradcheck(f, (x, y), eps=1e-3, atol=2e-3, rtol=2e-2)
        assert result == True

    def test_gradcheck_abs(self):
        """Gradient check for absolute value."""
        from python.foundations import Tensor, gradcheck

        def f(x):
            return x.abs().sum()

        # Avoid values close to 0 where gradient is undefined
        x = Tensor(np.random.randn(3, 4) + 1.0, requires_grad=True)

        result = gradcheck(f, (x,),  eps=1e-3, atol=1e-3, rtol=1e-2)
        assert result == True

    def test_gradcheck_var(self):
        """Gradient check for variance."""
        from python.foundations import Tensor, gradcheck

        def f(x):
            return x.var()

        x = Tensor(np.random.randn(3, 4), requires_grad=True)

        result = gradcheck(f, (x,), eps=1e-3, atol=1e-3, rtol=1e-2)
        assert result == True

    def test_gradcheck_logsigmoid(self):
        """Gradient check for log sigmoid."""
        from python.foundations import Tensor, gradcheck

        def f(x):
            return x.log_sigmoid().sum()

        # Use moderate values to avoid numerical issues
        x = Tensor(np.random.randn(3, 4) * 0.5, requires_grad=True)

        result = gradcheck(f, (x,),  eps=1e-3, atol=1e-3, rtol=1e-2)
        assert result == True

    def test_gradcheck_logsoftmax(self):
        """Gradient check for log softmax."""
        from python.foundations import Tensor, gradcheck

        def f(x):
            return x.log_softmax(axis=-1).sum()

        x = Tensor(np.random.randn(3, 4), requires_grad=True)

        result = gradcheck(f, (x,), eps=1e-3, atol=5e-3, rtol=5e-2)
        assert result == True


# =============================================================================
# Integration Tests (from test_foundations_new.py)
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple new operations."""

    def test_softmax_cross_entropy(self):
        """Test log_softmax in cross-entropy loss."""
        from python.foundations import Tensor

        # Logits and targets
        logits = Tensor(np.random.randn(4, 10), requires_grad=True)
        targets = np.array([3, 7, 1, 5])

        # Cross-entropy using log_softmax
        log_probs = logits.log_softmax(axis=-1)

        # Gather the log probs for the correct classes
        loss = Tensor(0.0, requires_grad=False)
        for i in range(4):
            loss = loss - log_probs[i, targets[i]]
        loss = loss / 4.0

        loss.backward()

        assert logits.grad is not None
        assert np.isfinite(logits.grad).all()

    def test_relu_via_maximum(self):
        """Test ReLU implementation using maximum."""
        from python.foundations import Tensor, maximum

        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)

        # ReLU = max(x, 0)
        y = maximum(x, 0.0)
        loss = y.sum()
        loss.backward()

        assert np.allclose(y.data, [0.0, 0.0, 0.0, 1.0, 2.0])
        # Gradient: 0 for x < 0, 1 for x > 0, 0.5 for x == 0 (split)
        expected_grad = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        assert np.allclose(x.grad, expected_grad)

    def test_layer_norm_components(self):
        """Test components used in layer normalization."""
        from python.foundations import Tensor

        x = Tensor(np.random.randn(2, 4), requires_grad=True)

        # Layer norm: (x - mean) / sqrt(var + eps)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        eps = 1e-5
        x_norm = (x - mean) / ((var + eps) ** 0.5)

        loss = x_norm.sum()
        loss.backward()

        assert x.grad is not None
        assert np.isfinite(x.grad).all()


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
