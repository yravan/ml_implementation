import numpy as np
import pytest
from nn_primitives.layers import (
    linear_forward, linear_backward,
    relu_forward, relu_backward,
    sigmoid_forward, sigmoid_backward,
    softmax_forward,
    mlp_forward, mlp_backward,
)


def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient of scalar f w.r.t. array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f()
        x[idx] = old - eps
        fm = f()
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx] = old
        it.iternext()
    return grad


class TestLinear:
    """Tests for fully-connected layer."""

    def test_forward_shape(self):
        X = np.random.randn(4, 3)
        W = np.random.randn(3, 5)
        b = np.random.randn(5)
        Y, _ = linear_forward(X, W, b)
        assert Y.shape == (4, 5)

    def test_forward_values(self):
        """Y = X @ W + b."""
        X = np.array([[1.0, 2.0]])
        W = np.array([[3.0], [4.0]])
        b = np.array([1.0])
        Y, _ = linear_forward(X, W, b)
        # 1*3 + 2*4 + 1 = 12
        np.testing.assert_almost_equal(Y[0, 0], 12.0)

    def test_backward_dX_shape(self):
        X = np.random.randn(4, 3)
        W = np.random.randn(3, 5)
        b = np.random.randn(5)
        _, cache = linear_forward(X, W, b)
        d_out = np.random.randn(4, 5)
        d_X, d_W, d_b = linear_backward(d_out, cache)
        assert d_X.shape == X.shape
        assert d_W.shape == W.shape
        assert d_b.shape == b.shape

    def test_backward_numerical(self):
        """Analytical gradients should match numerical gradients."""
        np.random.seed(42)
        X = np.random.randn(3, 4)
        W = np.random.randn(4, 2)
        b = np.random.randn(2)
        d_out = np.random.randn(3, 2)

        _, cache = linear_forward(X, W, b)
        d_X, d_W, d_b = linear_backward(d_out, cache)

        def loss_X():
            Y, _ = linear_forward(X, W, b)
            return np.sum(Y * d_out)

        def loss_W():
            Y, _ = linear_forward(X, W, b)
            return np.sum(Y * d_out)

        dX_num = numerical_gradient(loss_X, X)
        dW_num = numerical_gradient(loss_W, W)

        np.testing.assert_array_almost_equal(d_X, dX_num, decimal=5)
        np.testing.assert_array_almost_equal(d_W, dW_num, decimal=5)


class TestReLU:
    """Tests for ReLU activation."""

    def test_positive_passthrough(self):
        X = np.array([[1.0, 2.0, 3.0]])
        Y, _ = relu_forward(X)
        np.testing.assert_array_equal(Y, X)

    def test_negative_zeroed(self):
        X = np.array([[-1.0, -2.0, -3.0]])
        Y, _ = relu_forward(X)
        np.testing.assert_array_equal(Y, np.zeros_like(X))

    def test_mixed(self):
        X = np.array([[-1.0, 2.0, -3.0, 4.0]])
        Y, _ = relu_forward(X)
        np.testing.assert_array_equal(Y, [[0, 2, 0, 4]])

    def test_backward_positive(self):
        X = np.array([[1.0, 2.0]])
        _, cache = relu_forward(X)
        d_out = np.array([[5.0, 6.0]])
        d_X = relu_backward(d_out, cache)
        np.testing.assert_array_equal(d_X, d_out)

    def test_backward_negative(self):
        X = np.array([[-1.0, -2.0]])
        _, cache = relu_forward(X)
        d_out = np.array([[5.0, 6.0]])
        d_X = relu_backward(d_out, cache)
        np.testing.assert_array_equal(d_X, np.zeros_like(d_out))


class TestSigmoid:
    """Tests for sigmoid activation."""

    def test_zero(self):
        """sigmoid(0) = 0.5."""
        Y, _ = sigmoid_forward(np.array([[0.0]]))
        np.testing.assert_almost_equal(Y[0, 0], 0.5)

    def test_large_positive(self):
        """sigmoid(large) ≈ 1."""
        Y, _ = sigmoid_forward(np.array([[100.0]]))
        assert Y[0, 0] > 0.99

    def test_large_negative(self):
        """sigmoid(very negative) ≈ 0."""
        Y, _ = sigmoid_forward(np.array([[-100.0]]))
        assert Y[0, 0] < 0.01

    def test_output_range(self):
        """All outputs should be in (0, 1)."""
        X = np.random.randn(10, 5)
        Y, _ = sigmoid_forward(X)
        assert np.all(Y > 0) and np.all(Y < 1)

    def test_backward_numerical(self):
        np.random.seed(42)
        X = np.random.randn(3, 4)
        d_out = np.random.randn(3, 4)
        _, cache = sigmoid_forward(X)
        d_X = sigmoid_backward(d_out, cache)

        def loss():
            Y, _ = sigmoid_forward(X)
            return np.sum(Y * d_out)

        d_X_num = numerical_gradient(loss, X)
        np.testing.assert_array_almost_equal(d_X, d_X_num, decimal=5)


class TestSoftmax:
    """Tests for softmax."""

    def test_sums_to_one(self):
        X = np.random.randn(3, 5)
        probs = softmax_forward(X)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(3))

    def test_non_negative(self):
        X = np.random.randn(4, 3)
        probs = softmax_forward(X)
        assert np.all(probs >= 0)

    def test_shape(self):
        X = np.random.randn(2, 7)
        assert softmax_forward(X).shape == (2, 7)

    def test_numerical_stability(self):
        """Should not overflow with large inputs."""
        X = np.array([[1000.0, 1001.0, 1002.0]])
        probs = softmax_forward(X)
        assert np.all(np.isfinite(probs))
        np.testing.assert_almost_equal(probs.sum(), 1.0)

    def test_uniform(self):
        """Equal logits -> uniform distribution."""
        X = np.array([[5.0, 5.0, 5.0, 5.0]])
        probs = softmax_forward(X)
        np.testing.assert_array_almost_equal(probs, [[0.25, 0.25, 0.25, 0.25]])


class TestMLP:
    """Tests for multi-layer perceptron forward/backward."""

    def test_forward_shape(self):
        X = np.random.randn(4, 3)
        params = [
            (np.random.randn(3, 5), np.random.randn(5)),
            (np.random.randn(5, 2), np.random.randn(2)),
        ]
        output, caches = mlp_forward(X, params)
        assert output.shape == (4, 2)

    def test_single_layer(self):
        """Single-layer MLP = linear layer (no ReLU after last layer)."""
        X = np.array([[1.0, 2.0]])
        W = np.array([[1.0], [1.0]])
        b = np.array([0.0])
        output, _ = mlp_forward(X, [(W, b)])
        # 1*1 + 2*1 + 0 = 3
        np.testing.assert_almost_equal(output[0, 0], 3.0)

    def test_backward_shapes(self):
        """Backward should return gradients matching parameter shapes."""
        X = np.random.randn(4, 3)
        params = [
            (np.random.randn(3, 5), np.random.randn(5)),
            (np.random.randn(5, 2), np.random.randn(2)),
        ]
        output, caches = mlp_forward(X, params)
        d_out = np.random.randn(4, 2)
        d_input, grads = mlp_backward(d_out, caches)
        assert d_input.shape == X.shape
        assert len(grads) == len(params)
        for (dW, db), (W, b) in zip(grads, params):
            assert dW.shape == W.shape
            assert db.shape == b.shape

    def test_backward_numerical(self):
        """Numerical gradient check for 2-layer MLP."""
        np.random.seed(42)
        X = np.random.randn(2, 3)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(4)
        W2 = np.random.randn(4, 2)
        b2 = np.random.randn(2)
        params = [(W1, b1), (W2, b2)]
        d_out = np.random.randn(2, 2)

        output, caches = mlp_forward(X, params)
        _, grads = mlp_backward(d_out, caches)

        def loss_fn():
            out, _ = mlp_forward(X, params)
            return np.sum(out * d_out)

        dW1_num = numerical_gradient(loss_fn, W1)
        np.testing.assert_array_almost_equal(grads[0][0], dW1_num, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
