import numpy as np
import pytest
from nn_primitives.conv import (
    conv2d_forward, conv2d_backward,
    max_pool2d_forward, max_pool2d_backward,
)


class TestConv2DForward:
    """Tests for Conv2D forward pass."""

    def test_output_shape_no_pad(self):
        """Output dims: (H-kH)/stride + 1."""
        X = np.random.randn(1, 1, 5, 5)
        W = np.random.randn(1, 1, 3, 3)
        b = np.zeros(1)
        Y, _ = conv2d_forward(X, W, b, stride=1, pad=0)
        assert Y.shape == (1, 1, 3, 3)

    def test_output_shape_with_pad(self):
        """With pad=1, same-size output for 3x3 kernel."""
        X = np.random.randn(2, 3, 8, 8)
        W = np.random.randn(16, 3, 3, 3)
        b = np.zeros(16)
        Y, _ = conv2d_forward(X, W, b, stride=1, pad=1)
        assert Y.shape == (2, 16, 8, 8)

    def test_output_shape_stride2(self):
        X = np.random.randn(1, 1, 6, 6)
        W = np.random.randn(1, 1, 3, 3)
        b = np.zeros(1)
        Y, _ = conv2d_forward(X, W, b, stride=2, pad=0)
        assert Y.shape == (1, 1, 2, 2)

    def test_identity_filter(self):
        """A 1x1 identity filter should copy the input."""
        X = np.random.randn(1, 1, 4, 4)
        W = np.array([[[[1.0]]]]) # (1, 1, 1, 1)
        b = np.zeros(1)
        Y, _ = conv2d_forward(X, W, b, stride=1, pad=0)
        np.testing.assert_array_almost_equal(Y, X)

    def test_bias_added(self):
        """Output should include bias."""
        X = np.zeros((1, 1, 3, 3))
        W = np.zeros((1, 1, 1, 1))
        b = np.array([5.0])
        Y, _ = conv2d_forward(X, W, b, stride=1, pad=0)
        np.testing.assert_array_almost_equal(Y, np.full((1, 1, 3, 3), 5.0))

    def test_known_convolution(self):
        """Hand-computed 2x2 convolution on 3x3 input."""
        X = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(float)
        W = np.array([[[[1, 0],
                        [0, 1]]]]).astype(float)
        b = np.zeros(1)
        Y, _ = conv2d_forward(X, W, b, stride=1, pad=0)
        # Top-left: 1*1 + 2*0 + 4*0 + 5*1 = 6
        # Top-right: 2*1 + 3*0 + 5*0 + 6*1 = 8
        expected = np.array([[[[6, 8], [12, 14]]]]).astype(float)
        np.testing.assert_array_almost_equal(Y, expected)


class TestConv2DBackward:
    """Tests for Conv2D backward pass."""

    def test_gradient_shapes(self):
        X = np.random.randn(2, 3, 5, 5)
        W = np.random.randn(4, 3, 3, 3)
        b = np.zeros(4)
        Y, cache = conv2d_forward(X, W, b, stride=1, pad=1)
        d_out = np.random.randn(*Y.shape)
        d_X, d_W, d_b = conv2d_backward(d_out, cache)
        assert d_X.shape == X.shape
        assert d_W.shape == W.shape
        assert d_b.shape == b.shape

    def test_bias_gradient(self):
        """d_b should sum over batch, height, width."""
        X = np.random.randn(2, 1, 4, 4)
        W = np.random.randn(3, 1, 3, 3)
        b = np.zeros(3)
        Y, cache = conv2d_forward(X, W, b, stride=1, pad=0)
        d_out = np.ones_like(Y)
        _, _, d_b = conv2d_backward(d_out, cache)
        # d_b[c] = sum of all d_out for channel c = 2 * 2 * 2 = 8
        np.testing.assert_array_almost_equal(d_b, np.full(3, 8.0))

    def test_numerical_gradient_W(self):
        """Numerical gradient check for filter weights."""
        np.random.seed(42)
        X = np.random.randn(1, 1, 4, 4)
        W = np.random.randn(1, 1, 2, 2)
        b = np.zeros(1)
        d_out_fixed = np.random.randn(1, 1, 3, 3)

        _, cache = conv2d_forward(X, W, b)
        _, d_W, _ = conv2d_backward(d_out_fixed, cache)

        eps = 1e-5
        d_W_num = np.zeros_like(W)
        it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old = W[idx]
            W[idx] = old + eps
            Yp, _ = conv2d_forward(X, W, b)
            W[idx] = old - eps
            Ym, _ = conv2d_forward(X, W, b)
            d_W_num[idx] = np.sum((Yp - Ym) * d_out_fixed) / (2 * eps)
            W[idx] = old
            it.iternext()

        np.testing.assert_array_almost_equal(d_W, d_W_num, decimal=4)


class TestMaxPool2DForward:
    """Tests for max pooling forward."""

    def test_output_shape(self):
        X = np.random.randn(2, 3, 4, 4)
        Y, _ = max_pool2d_forward(X, pool_size=2, stride=2)
        assert Y.shape == (2, 3, 2, 2)

    def test_picks_max(self):
        """Should select the maximum in each window."""
        X = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]]).astype(float)
        Y, _ = max_pool2d_forward(X, pool_size=2, stride=2)
        expected = np.array([[[[6, 8], [14, 16]]]]).astype(float)
        np.testing.assert_array_almost_equal(Y, expected)

    def test_preserves_channels(self):
        X = np.random.randn(1, 5, 4, 4)
        Y, _ = max_pool2d_forward(X, pool_size=2, stride=2)
        assert Y.shape[1] == 5


class TestMaxPool2DBackward:
    """Tests for max pooling backward."""

    def test_gradient_shape(self):
        X = np.random.randn(2, 3, 4, 4)
        Y, cache = max_pool2d_forward(X, pool_size=2, stride=2)
        d_out = np.random.randn(*Y.shape)
        d_X = max_pool2d_backward(d_out, cache)
        assert d_X.shape == X.shape

    def test_gradient_routing(self):
        """Gradient should only go to the max position in each window."""
        X = np.array([[[[1, 2],
                        [3, 4]]]]).astype(float)
        Y, cache = max_pool2d_forward(X, pool_size=2, stride=2)
        d_out = np.array([[[[1.0]]]])
        d_X = max_pool2d_backward(d_out, cache)
        # Max is at position (1,1) = 4
        expected = np.array([[[[0, 0], [0, 1]]]]).astype(float)
        np.testing.assert_array_almost_equal(d_X, expected)

    def test_non_max_get_zero(self):
        """Non-max positions should have zero gradient."""
        X = np.array([[[[10, 1], [1, 1]]]]).astype(float)
        _, cache = max_pool2d_forward(X, pool_size=2, stride=2)
        d_out = np.array([[[[5.0]]]])
        d_X = max_pool2d_backward(d_out, cache)
        assert d_X[0, 0, 0, 0] == 5.0  # Max position
        assert d_X[0, 0, 0, 1] == 0.0
        assert d_X[0, 0, 1, 0] == 0.0
        assert d_X[0, 0, 1, 1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
