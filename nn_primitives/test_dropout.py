import numpy as np
import pytest
from nn_primitives.dropout import dropout_forward, dropout_backward


class TestDropoutForward:
    """Tests for dropout forward pass."""

    def test_inference_identity(self):
        """In inference mode, output should equal input."""
        X = np.random.randn(5, 3)
        Y, _ = dropout_forward(X, p=0.5, training=False)
        np.testing.assert_array_equal(Y, X)

    def test_output_shape(self):
        X = np.random.randn(4, 6)
        Y, _ = dropout_forward(X, p=0.3, training=True)
        assert Y.shape == X.shape

    def test_some_zeros(self):
        """With p=0.5 and large input, some outputs should be zero."""
        np.random.seed(42)
        X = np.ones((100, 100))
        Y, _ = dropout_forward(X, p=0.5, training=True)
        zero_frac = np.mean(Y == 0)
        assert 0.3 < zero_frac < 0.7  # Roughly 50% dropped

    def test_inverted_scaling(self):
        """Surviving values should be scaled by 1/(1-p)."""
        np.random.seed(42)
        X = np.ones((1000, 100))
        p = 0.3
        Y, _ = dropout_forward(X, p=p, training=True)
        # Non-zero values should be 1 / (1-p) ≈ 1.4286
        nonzero = Y[Y != 0]
        np.testing.assert_almost_equal(nonzero.mean(), 1.0 / (1 - p), decimal=1)

    def test_expected_value_preserved(self):
        """E[Y] ≈ X due to inverted dropout scaling."""
        np.random.seed(42)
        X = np.ones((10000, 10)) * 5.0
        Y, _ = dropout_forward(X, p=0.4, training=True)
        np.testing.assert_almost_equal(Y.mean(), 5.0, decimal=1)

    def test_custom_mask(self):
        """Custom mask should be used when provided."""
        X = np.array([[1.0, 2.0, 3.0]])
        mask = np.array([[1, 0, 1]])
        Y, _ = dropout_forward(X, p=0.5, training=True, mask=mask)
        assert Y[0, 1] == 0.0  # Dropped
        assert Y[0, 0] != 0.0  # Kept


class TestDropoutBackward:
    """Tests for dropout backward pass."""

    def test_gradient_shape(self):
        X = np.random.randn(4, 3)
        _, cache = dropout_forward(X, p=0.5, training=True)
        d_out = np.random.randn(4, 3)
        d_X = dropout_backward(d_out, cache)
        assert d_X.shape == X.shape

    def test_zero_gradient_where_dropped(self):
        """Gradient should be zero where units were dropped."""
        X = np.ones((1, 5))
        mask = np.array([[1, 0, 1, 0, 1]])
        _, cache = dropout_forward(X, p=0.5, training=True, mask=mask)
        d_out = np.ones((1, 5))
        d_X = dropout_backward(d_out, cache)
        assert d_X[0, 1] == 0.0
        assert d_X[0, 3] == 0.0

    def test_gradient_scaled_where_kept(self):
        """Gradient should be scaled by 1/(1-p) where kept."""
        X = np.ones((1, 3))
        mask = np.array([[1, 1, 1]])
        p = 0.4
        _, cache = dropout_forward(X, p=p, training=True, mask=mask)
        d_out = np.ones((1, 3))
        d_X = dropout_backward(d_out, cache)
        np.testing.assert_array_almost_equal(d_X, np.full((1, 3), 1.0 / (1 - p)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
