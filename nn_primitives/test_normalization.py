import numpy as np
import pytest
from nn_primitives.normalization import (
    batch_norm_forward, batch_norm_backward,
    layer_norm_forward, layer_norm_backward,
)


class TestBatchNormForward:
    """Tests for batch normalization forward pass."""

    def test_output_shape(self):
        X = np.random.randn(8, 4)
        gamma = np.ones(4)
        beta = np.zeros(4)
        rm = np.zeros(4)
        rv = np.ones(4)
        Y, _ = batch_norm_forward(X, gamma, beta, rm, rv)
        assert Y.shape == (8, 4)

    def test_zero_mean_unit_var(self):
        """With gamma=1, beta=0, output should have ~zero mean, ~unit var per feature."""
        np.random.seed(42)
        X = np.random.randn(100, 5) * 3 + 7  # Mean ~7, std ~3
        gamma = np.ones(5)
        beta = np.zeros(5)
        rm = np.zeros(5)
        rv = np.ones(5)
        Y, _ = batch_norm_forward(X, gamma, beta, rm, rv, training=True)
        np.testing.assert_array_almost_equal(Y.mean(axis=0), np.zeros(5), decimal=5)
        np.testing.assert_array_almost_equal(Y.var(axis=0), np.ones(5), decimal=1)

    def test_gamma_beta_effect(self):
        """gamma scales, beta shifts."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        gamma = np.array([2.0, 3.0, 0.5])
        beta = np.array([1.0, -1.0, 0.0])
        rm = np.zeros(3)
        rv = np.ones(3)
        Y, _ = batch_norm_forward(X, gamma, beta, rm, rv, training=True)
        # Mean should be close to beta, std close to gamma
        np.testing.assert_array_almost_equal(Y.mean(axis=0), beta, decimal=1)

    def test_inference_mode(self):
        """In inference mode, should use running stats, not batch stats."""
        X = np.random.randn(4, 3)
        gamma = np.ones(3)
        beta = np.zeros(3)
        rm = np.array([5.0, 5.0, 5.0])
        rv = np.array([4.0, 4.0, 4.0])
        Y, _ = batch_norm_forward(X, gamma, beta, rm.copy(), rv.copy(), training=False)
        # Should normalize using running stats: (X - 5) / sqrt(4 + eps) = (X - 5) / 2
        expected = (X - 5.0) / np.sqrt(4.0 + 1e-5)
        np.testing.assert_array_almost_equal(Y, expected, decimal=5)

    def test_running_stats_updated(self):
        """Running mean/var should be updated during training."""
        X = np.random.randn(10, 3) + 5
        gamma = np.ones(3)
        beta = np.zeros(3)
        rm = np.zeros(3)
        rv = np.ones(3)
        batch_norm_forward(X, gamma, beta, rm, rv, training=True, momentum=0.1)
        # Running mean should have moved toward batch mean
        assert np.all(rm > 0)  # Batch mean is ~5, so rm should increase


class TestBatchNormBackward:
    """Tests for batch normalization backward pass."""

    def test_gradient_shapes(self):
        np.random.seed(42)
        X = np.random.randn(8, 4)
        gamma = np.ones(4)
        beta = np.zeros(4)
        rm = np.zeros(4)
        rv = np.ones(4)
        Y, cache = batch_norm_forward(X, gamma, beta, rm, rv)
        d_out = np.random.randn(8, 4)
        d_X, d_gamma, d_beta = batch_norm_backward(d_out, cache)
        assert d_X.shape == X.shape
        assert d_gamma.shape == gamma.shape
        assert d_beta.shape == beta.shape

    def test_d_beta(self):
        """d_beta = sum(d_out, axis=0)."""
        np.random.seed(42)
        X = np.random.randn(4, 3)
        gamma = np.ones(3)
        beta = np.zeros(3)
        rm = np.zeros(3)
        rv = np.ones(3)
        _, cache = batch_norm_forward(X, gamma, beta, rm, rv)
        d_out = np.ones((4, 3))
        _, _, d_beta = batch_norm_backward(d_out, cache)
        np.testing.assert_array_almost_equal(d_beta, [4, 4, 4])

    def test_numerical_gradient(self):
        """Numerical gradient check for d_X."""
        np.random.seed(42)
        X = np.random.randn(5, 3)
        gamma = np.random.randn(3) * 0.5 + 1
        beta = np.random.randn(3) * 0.1
        d_out = np.random.randn(5, 3)

        rm = np.zeros(3)
        rv = np.ones(3)
        _, cache = batch_norm_forward(X, gamma, beta, rm.copy(), rv.copy())
        d_X, _, _ = batch_norm_backward(d_out, cache)

        eps = 1e-5
        d_X_num = np.zeros_like(X)
        it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old = X[idx]
            X[idx] = old + eps
            Yp, _ = batch_norm_forward(X, gamma, beta, rm.copy(), rv.copy())
            X[idx] = old - eps
            Ym, _ = batch_norm_forward(X, gamma, beta, rm.copy(), rv.copy())
            d_X_num[idx] = np.sum((Yp - Ym) * d_out) / (2 * eps)
            X[idx] = old
            it.iternext()

        np.testing.assert_array_almost_equal(d_X, d_X_num, decimal=4)


class TestLayerNormForward:
    """Tests for layer normalization forward pass."""

    def test_output_shape(self):
        X = np.random.randn(4, 8)
        gamma = np.ones(8)
        beta = np.zeros(8)
        Y, _ = layer_norm_forward(X, gamma, beta)
        assert Y.shape == (4, 8)

    def test_per_sample_normalization(self):
        """Each sample should be independently normalized."""
        X = np.random.randn(3, 10) * 5 + 3
        gamma = np.ones(10)
        beta = np.zeros(10)
        Y, _ = layer_norm_forward(X, gamma, beta)
        # Each row should have ~zero mean, ~unit var
        for i in range(3):
            np.testing.assert_almost_equal(Y[i].mean(), 0, decimal=5)
            np.testing.assert_almost_equal(Y[i].var(), 1, decimal=1)

    def test_no_running_stats(self):
        """LayerNorm should not depend on batch size."""
        X1 = np.array([[1.0, 2.0, 3.0]])
        X2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        gamma = np.ones(3)
        beta = np.zeros(3)
        Y1, _ = layer_norm_forward(X1, gamma, beta)
        Y2, _ = layer_norm_forward(X2, gamma, beta)
        np.testing.assert_array_almost_equal(Y1[0], Y2[0])


class TestLayerNormBackward:
    """Tests for layer normalization backward pass."""

    def test_gradient_shapes(self):
        X = np.random.randn(4, 6)
        gamma = np.ones(6)
        beta = np.zeros(6)
        _, cache = layer_norm_forward(X, gamma, beta)
        d_out = np.random.randn(4, 6)
        d_X, d_gamma, d_beta = layer_norm_backward(d_out, cache)
        assert d_X.shape == X.shape
        assert d_gamma.shape == gamma.shape
        assert d_beta.shape == beta.shape

    def test_numerical_gradient(self):
        np.random.seed(42)
        X = np.random.randn(3, 4)
        gamma = np.ones(4) * 1.5
        beta = np.random.randn(4) * 0.1
        d_out = np.random.randn(3, 4)

        _, cache = layer_norm_forward(X, gamma, beta)
        d_X, _, _ = layer_norm_backward(d_out, cache)

        eps = 1e-5
        d_X_num = np.zeros_like(X)
        it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old = X[idx]
            X[idx] = old + eps
            Yp, _ = layer_norm_forward(X, gamma, beta)
            X[idx] = old - eps
            Ym, _ = layer_norm_forward(X, gamma, beta)
            d_X_num[idx] = np.sum((Yp - Ym) * d_out) / (2 * eps)
            X[idx] = old
            it.iternext()

        np.testing.assert_array_almost_equal(d_X, d_X_num, decimal=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
