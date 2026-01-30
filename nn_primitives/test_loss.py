import numpy as np
import pytest
from nn_primitives.loss import mse_loss, cross_entropy_loss, binary_cross_entropy_loss


class TestMSELoss:
    """Tests for mean squared error loss."""

    def test_perfect_prediction(self):
        y_pred = np.array([[1.0, 2.0]])
        loss, grad = mse_loss(y_pred, y_pred)
        np.testing.assert_almost_equal(loss, 0.0)
        np.testing.assert_array_almost_equal(grad, np.zeros_like(y_pred))

    def test_known_value(self):
        y_pred = np.array([[0.0, 0.0]])
        y_true = np.array([[3.0, 4.0]])
        loss, _ = mse_loss(y_pred, y_true)
        # MSE = mean(9 + 16) = 12.5
        np.testing.assert_almost_equal(loss, 12.5)

    def test_gradient_direction(self):
        """Gradient should push prediction toward target."""
        y_pred = np.array([[0.0]])
        y_true = np.array([[5.0]])
        _, grad = mse_loss(y_pred, y_true)
        assert grad[0, 0] < 0  # Should push pred upward

    def test_gradient_shape(self):
        y_pred = np.random.randn(8, 3)
        y_true = np.random.randn(8, 3)
        _, grad = mse_loss(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_symmetric(self):
        """MSE(a, b) = MSE(b, a)."""
        a = np.random.randn(3, 2)
        b = np.random.randn(3, 2)
        loss1, _ = mse_loss(a, b)
        loss2, _ = mse_loss(b, a)
        np.testing.assert_almost_equal(loss1, loss2)


class TestCrossEntropyLoss:
    """Tests for cross-entropy loss from logits."""

    def test_perfect_prediction(self):
        """High logit for correct class should give low loss."""
        logits = np.array([[10.0, -10.0, -10.0]])
        y_true = np.array([0])
        loss, _ = cross_entropy_loss(logits, y_true)
        assert loss < 0.01

    def test_wrong_prediction(self):
        """High logit for wrong class should give high loss."""
        logits = np.array([[-10.0, 10.0, -10.0]])
        y_true = np.array([0])
        loss, _ = cross_entropy_loss(logits, y_true)
        assert loss > 10

    def test_gradient_shape(self):
        logits = np.random.randn(4, 5)
        y_true = np.array([0, 1, 2, 3])
        _, grad = cross_entropy_loss(logits, y_true)
        assert grad.shape == logits.shape

    def test_gradient_formula(self):
        """Gradient should be (softmax - one_hot) / N."""
        logits = np.array([[1.0, 2.0, 3.0]])
        y_true = np.array([1])
        _, grad = cross_entropy_loss(logits, y_true)
        # softmax of [1, 2, 3]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        expected = probs.copy()
        expected[0, 1] -= 1.0
        expected /= 1  # N=1
        np.testing.assert_array_almost_equal(grad, expected)

    def test_uniform_logits(self):
        """Uniform logits should give loss = log(n_classes)."""
        n_classes = 5
        logits = np.zeros((1, n_classes))
        y_true = np.array([0])
        loss, _ = cross_entropy_loss(logits, y_true)
        np.testing.assert_almost_equal(loss, np.log(n_classes), decimal=5)

    def test_non_negative(self):
        np.random.seed(42)
        logits = np.random.randn(10, 4)
        y_true = np.random.randint(0, 4, size=10)
        loss, _ = cross_entropy_loss(logits, y_true)
        assert loss >= 0


class TestBinaryCrossEntropyLoss:
    """Tests for binary cross-entropy loss."""

    def test_perfect_prediction(self):
        y_pred = np.array([0.99, 0.01])
        y_true = np.array([1.0, 0.0])
        loss, _ = binary_cross_entropy_loss(y_pred, y_true)
        assert loss < 0.1

    def test_worst_prediction(self):
        y_pred = np.array([0.01, 0.99])
        y_true = np.array([1.0, 0.0])
        loss, _ = binary_cross_entropy_loss(y_pred, y_true)
        assert loss > 2

    def test_gradient_shape(self):
        y_pred = np.random.uniform(0.1, 0.9, size=5)
        y_true = np.array([1, 0, 1, 0, 1]).astype(float)
        _, grad = binary_cross_entropy_loss(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_non_negative(self):
        y_pred = np.random.uniform(0.1, 0.9, size=10)
        y_true = np.random.randint(0, 2, size=10).astype(float)
        loss, _ = binary_cross_entropy_loss(y_pred, y_true)
        assert loss >= 0

    def test_midpoint(self):
        """BCE(0.5, 1) = BCE(0.5, 0) = log(2)."""
        y_pred = np.array([0.5])
        loss1, _ = binary_cross_entropy_loss(y_pred, np.array([1.0]))
        loss2, _ = binary_cross_entropy_loss(y_pred, np.array([0.0]))
        np.testing.assert_almost_equal(loss1, np.log(2), decimal=5)
        np.testing.assert_almost_equal(loss2, np.log(2), decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
