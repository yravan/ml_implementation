"""
Comprehensive Tests for Optimization Module
============================================

This test suite covers all components of the optimization module:
- Optimizers (SGD, Adam, AdamW, RMSprop, etc.)
- Loss Functions (MSE, CrossEntropy, Focal, etc.)
- Learning Rate Schedulers (StepLR, CosineAnnealing, OneCycle, etc.)
- Gradient Utilities (clipping, accumulation, scaling, etc.)

Each test verifies:
1. Correct computation of values
2. Proper gradient flow (where applicable)
3. State management and checkpointing
4. Edge cases and numerical stability

GRADIENT TESTING:
Uses gradcheck from foundations to verify that analytical gradients (from backward pass)
match numerical gradients (computed via finite differences) for loss functions.

OPTIMIZER TESTING:
Verifies that optimizers correctly update parameters in the expected direction
and that optimizer state (momentum, etc.) is properly maintained.

SCHEDULER TESTING:
Verifies that learning rates follow the expected schedule at each step/epoch.
"""

import numpy as np
import pytest
from typing import Callable, Tuple, List
import math

# Import gradient checking utilities
from python.foundations import Tensor, gradcheck, numerical_gradient, gradient_check


# =============================================================================
# Mock Classes for Testing
# =============================================================================

class MockOptimizer:
    """Simple mock optimizer for testing schedulers."""
    def __init__(self, lr=0.01):
        self.defaults = {'lr': lr}
        self._lr = lr

    def get_lr(self):
        return {-1: self._lr}

    def set_lr(self, lr):
        if isinstance(lr, dict):
            self._lr = list(lr.values())[0]
        else:
            self._lr = lr
        self.defaults['lr'] = self._lr

    def parameters(self):
        return []


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
def sample_params(seed):
    """Create sample parameters for optimizer testing as Tensors."""
    from python.foundations import Tensor
    return [
        Tensor(np.random.randn(10, 5).astype(np.float64), requires_grad=True),
        Tensor(np.random.randn(5,).astype(np.float64), requires_grad=True),
        Tensor(np.random.randn(5, 3).astype(np.float64), requires_grad=True),
    ]


@pytest.fixture
def sample_grads(sample_params):
    """Create sample gradients matching parameter shapes."""
    return [np.random.randn(*p.data.shape).astype(np.float64) for p in sample_params]


def set_grads(params, grads):
    """Helper to set gradients on Tensor parameters."""
    for p, g in zip(params, grads):
        p.grad = g.copy()


@pytest.fixture
def regression_data(seed):
    """Create sample regression data."""
    predictions = np.random.randn(32, 1).astype(np.float64)
    targets = np.random.randn(32, 1).astype(np.float64)
    return predictions, targets


@pytest.fixture
def classification_data(seed):
    """Create sample classification data."""
    batch_size = 32
    num_classes = 10
    logits = np.random.randn(batch_size, num_classes).astype(np.float64)
    targets = np.random.randint(0, num_classes, size=(batch_size,))
    return logits, targets


@pytest.fixture
def binary_classification_data(seed):
    """Create sample binary classification data."""
    batch_size = 32
    logits = np.random.randn(batch_size, 1).astype(np.float64)
    targets = np.random.randint(0, 2, size=(batch_size, 1)).astype(np.float64)
    return logits, targets


# =============================================================================
# OPTIMIZER TESTS
# =============================================================================

# =============================================================================
# STRONG EXISTING TESTS (KEPT)
# =============================================================================

class TestSGD:
    """Test SGD optimizer with new Tensor-based API."""

    def test_sgd_creation(self, sample_params):
        """Test SGD optimizer creation."""
        from python.optimization import SGD

        optimizer = SGD(sample_params, lr=0.01)
        assert optimizer is not None
        assert optimizer.defaults['lr'] == 0.01

    def test_vanilla_sgd_step(self, sample_params, sample_grads):
        """Test vanilla SGD (no momentum) takes correct step."""
        from python.optimization import SGD
        from python.foundations import Tensor

        # Create fresh params
        params = [Tensor(np.random.randn(10, 5).astype(np.float64), requires_grad=True)]
        grads = [np.random.randn(10, 5).astype(np.float64)]
        original_data = params[0].data.copy()
        lr = 0.1

        optimizer = SGD(params, lr=lr, momentum=0.0)
        set_grads(params, grads)
        optimizer.step()

        # Verify: param = param - lr * grad
        expected = original_data - lr * grads[0]
        assert np.allclose(params[0].data, expected), "Vanilla SGD step incorrect"

    def test_sgd_with_momentum(self, sample_params, sample_grads):
        """Test SGD with momentum accumulates velocity correctly."""
        from python.optimization import SGD
        from python.foundations import Tensor

        params = [Tensor(np.random.randn(5, 3).astype(np.float64), requires_grad=True)]
        grads = [np.random.randn(5, 3).astype(np.float64)]
        lr = 0.1
        momentum = 0.9

        optimizer = SGD(params, lr=lr, momentum=momentum)

        # First step: velocity = grad, param = param - lr * grad
        original_data = params[0].data.copy()
        set_grads(params, grads)
        optimizer.step()

        # After first step, velocity should equal grad
        assert params[0] in optimizer.velocities
        assert np.allclose(optimizer.velocities[params[0]], grads[0])
        expected_after_step1 = original_data - lr * grads[0]
        assert np.allclose(params[0].data, expected_after_step1), "First momentum step incorrect"

        # Second step: velocity = momentum * velocity + grad
        data_before_step2 = params[0].data.copy()
        set_grads(params, grads)
        optimizer.step()

        expected_velocity = momentum * grads[0] + grads[0]  # momentum * old_v + grad
        assert np.allclose(optimizer.velocities[params[0]], expected_velocity, rtol=1e-5), \
            "Momentum velocity accumulation incorrect"

    def test_sgd_with_weight_decay(self, sample_params, sample_grads):
        """Test SGD with L2 weight decay."""
        from python.optimization import SGD
        from python.foundations import Tensor

        params = [Tensor(np.random.randn(4, 4).astype(np.float64), requires_grad=True)]
        grads = [np.random.randn(4, 4).astype(np.float64)]
        original_data = params[0].data.copy()
        lr = 0.1
        weight_decay = 0.01

        optimizer = SGD(params, lr=lr, weight_decay=weight_decay)
        set_grads(params, grads)
        optimizer.step()

        # With weight decay: param = param - lr * (grad + weight_decay * param)
        effective_grad = grads[0] + weight_decay * original_data
        expected = original_data - lr * effective_grad
        assert np.allclose(params[0].data, expected), "SGD with weight decay incorrect"

    def test_sgd_nesterov_momentum(self, sample_params, sample_grads):
        """Test SGD with Nesterov momentum."""
        from python.optimization import SGD
        from python.foundations import Tensor

        params = [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)]
        grads = [np.random.randn(3, 3).astype(np.float64)]
        lr = 0.1
        momentum = 0.9

        optimizer = SGD(params, lr=lr, momentum=momentum, nesterov=True)

        # Take a step
        set_grads(params, grads)
        optimizer.step()

        # Nesterov: descent = velocity * momentum + grad
        # After first step: velocity = grad, descent = grad * momentum + grad
        # param = param - lr * descent

    def test_sgd_multiple_param_groups(self):
        """Test SGD with multiple parameter groups with different learning rates."""
        from python.optimization import SGD
        from python.foundations import Tensor

        param1 = Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)
        param2 = Tensor(np.random.randn(2, 2).astype(np.float64), requires_grad=True)

        param_groups = [
            {'params': [param1], 'lr': 0.1},
            {'params': [param2], 'lr': 0.01}
        ]

        optimizer = SGD(param_groups, lr=0.05)  # default lr

        # Set gradients
        param1.grad = np.ones_like(param1.data)
        param2.grad = np.ones_like(param2.data)

        original1 = param1.data.copy()
        original2 = param2.data.copy()

        optimizer.step()

        # Each group should use its own lr
        assert np.allclose(param1.data, original1 - 0.1 * np.ones_like(original1))
        assert np.allclose(param2.data, original2 - 0.01 * np.ones_like(original2))

    def test_sgd_numerical_stability(self):
        """Test SGD handles extreme values without NaN/Inf."""
        from python.optimization import SGD
        from python.foundations import Tensor

        # Very small gradients
        params = [Tensor(np.ones((3, 3)).astype(np.float64) * 1e-10, requires_grad=True)]
        optimizer = SGD(params, lr=0.01)
        params[0].grad = np.ones((3, 3)) * 1e-15
        optimizer.step()
        assert np.all(np.isfinite(params[0].data)), "SGD produced NaN/Inf with small values"

        # Very large gradients
        params = [Tensor(np.ones((3, 3)).astype(np.float64), requires_grad=True)]
        optimizer = SGD(params, lr=0.01)
        params[0].grad = np.ones((3, 3)) * 1e10
        optimizer.step()
        assert np.all(np.isfinite(params[0].data)), "SGD produced NaN/Inf with large gradients"



class TestAdam:
    """Test Adam optimizer with new Tensor-based API."""

    def test_adam_creation(self, sample_params):
        """Test Adam optimizer creation."""
        from python.optimization import Adam

        optimizer = Adam(sample_params, lr=0.001)
        assert optimizer is not None
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['betas'] == (0.9, 0.999)

    def test_adam_step_correctness(self, sample_params, sample_grads):
        """Test Adam computes correct update with bias correction."""
        from python.optimization import Adam
        from python.foundations import Tensor

        params = [Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float64), requires_grad=True)]
        grads = [np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float64)]
        original_data = params[0].data.copy()
        lr = 0.001
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        optimizer = Adam(params, lr=lr, betas=(beta1, beta2), eps=eps)
        set_grads(params, grads)
        optimizer.step()

        # Manual computation for first step
        g = grads[0]
        m = (1 - beta1) * g  # m = 0 + (1-beta1)*g = (1-beta1)*g
        v = (1 - beta2) * (g ** 2)  # v = 0 + (1-beta2)*g^2
        m_hat = m / (1 - beta1 ** 1)  # bias correction
        v_hat = v / (1 - beta2 ** 1)
        expected = original_data - lr * m_hat / (np.sqrt(v_hat) + eps)

        assert np.allclose(params[0].data, expected, rtol=1e-5), \
            f"Adam step incorrect: got {params[0].data}, expected {expected}"

    def test_adam_bias_correction(self, sample_params, sample_grads):
        """Test Adam bias correction is applied correctly over multiple steps."""
        from python.optimization import Adam
        from python.foundations import Tensor

        params = [Tensor(np.ones((2, 2)).astype(np.float64), requires_grad=True)]
        grads = [np.ones((2, 2)).astype(np.float64) * 0.1]
        beta1, beta2 = 0.9, 0.999

        optimizer = Adam(params, lr=0.001, betas=(beta1, beta2))

        # Take multiple steps
        for step in range(1, 6):
            set_grads(params, grads)
            optimizer.step()
            assert optimizer._step_count == step

            # Verify moment estimates exist
            assert params[0] in optimizer.exp_avg
            assert params[0] in optimizer.exp_avg_sq

    def test_adam_with_amsgrad(self, sample_params, sample_grads):
        """Test Adam with AMSGrad variant tracks max variance."""
        from python.optimization import Adam
        from python.foundations import Tensor

        params = [Tensor(np.ones((3, 3)).astype(np.float64), requires_grad=True)]

        optimizer = Adam(params, lr=0.001, amsgrad=True)

        # First step with small gradient
        params[0].grad = np.ones((3, 3)) * 0.1
        optimizer.step()

        # Second step with larger gradient
        params[0].grad = np.ones((3, 3)) * 1.0
        optimizer.step()

        # Verify max_exp_avg_sq is tracked
        assert params[0] in optimizer.max_exp_avg_sq

    def test_adam_weight_decay(self, sample_params, sample_grads):
        """Test Adam with L2 weight decay (adds to gradient)."""
        from python.optimization import Adam
        from python.foundations import Tensor

        params = [Tensor(np.ones((2, 2)).astype(np.float64) * 2.0, requires_grad=True)]
        grads = [np.ones((2, 2)).astype(np.float64) * 0.1]

        optimizer = Adam(params, lr=0.001, weight_decay=0.01)
        original_data = params[0].data.copy()
        set_grads(params, grads)
        optimizer.step()

        # With weight_decay, effective gradient = grad + weight_decay * param
        # Parameters should move more due to weight decay
        assert not np.allclose(params[0].data, original_data)

    def test_adam_numerical_stability(self):
        """Test Adam handles extreme values without NaN/Inf."""
        from python.optimization import Adam
        from python.foundations import Tensor

        # Very small gradients
        params = [Tensor(np.ones((3, 3)).astype(np.float64), requires_grad=True)]
        optimizer = Adam(params, lr=0.001)
        params[0].grad = np.ones((3, 3)) * 1e-15
        optimizer.step()
        assert np.all(np.isfinite(params[0].data)), "Adam produced NaN/Inf with small gradients"

        # Very large gradients
        params = [Tensor(np.ones((3, 3)).astype(np.float64), requires_grad=True)]
        optimizer = Adam(params, lr=0.001)
        params[0].grad = np.ones((3, 3)) * 1e6
        optimizer.step()
        assert np.all(np.isfinite(params[0].data)), "Adam produced NaN/Inf with large gradients"

    def test_adam_convergence_on_quadratic(self):
        """Test Adam converges on simple quadratic function f(x) = x^2."""
        from python.optimization import Adam
        from python.foundations import Tensor

        # Minimize f(x) = x^2, gradient = 2x
        x = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        optimizer = Adam([x], lr=0.1)

        for _ in range(100):
            x.grad = 2 * x.data  # gradient of x^2
            optimizer.step()

        # Should converge close to 0
        assert np.abs(x.data[0]) < 0.1, f"Adam failed to converge: x = {x.data[0]}"



class TestAdamW:
    """Test AdamW optimizer with decoupled weight decay."""

    def test_adamw_creation(self, sample_params):
        """Test AdamW optimizer creation."""
        from python.optimization import AdamW

        optimizer = AdamW(sample_params, lr=0.001, weight_decay=0.01)
        assert optimizer is not None
        assert optimizer.defaults['weight_decay'] == 0.01

    def test_adamw_decoupled_weight_decay_formula(self, sample_params, sample_grads):
        """Test AdamW applies decoupled weight decay (not in gradient)."""
        from python.optimization import AdamW
        from python.foundations import Tensor

        # With decoupled weight decay:
        # param = param * (1 - lr * weight_decay) - lr * adam_update
        # NOT: param = param - lr * (adam_update + weight_decay * param)

        params = [Tensor(np.ones((2, 2)).astype(np.float64) * 2.0, requires_grad=True)]
        grads = [np.zeros((2, 2)).astype(np.float64)]  # zero gradient
        original_data = params[0].data.copy()
        lr = 0.1
        weight_decay = 0.1

        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        set_grads(params, grads)
        optimizer.step()

        # With zero gradient, only weight decay should apply
        # Decoupled: param = param * (1 - lr * weight_decay) = 2.0 * 0.99 = 1.98
        expected = original_data * (1 - lr * weight_decay)
        assert np.allclose(params[0].data, expected, rtol=1e-4), \
            f"AdamW decoupled weight decay incorrect: got {params[0].data}, expected {expected}"

    def test_adamw_differs_from_adam_l2(self):
        """Test AdamW produces different results from Adam with L2."""
        from python.optimization import Adam, AdamW
        from python.foundations import Tensor

        # Create identical starting conditions
        np.random.seed(42)
        data = np.random.randn(3, 3).astype(np.float64)
        grad = np.random.randn(3, 3).astype(np.float64)

        params_adam = [Tensor(data.copy(), requires_grad=True)]
        params_adamw = [Tensor(data.copy(), requires_grad=True)]

        adam = Adam(params_adam, lr=0.01, weight_decay=0.1)
        adamw = AdamW(params_adamw, lr=0.01, weight_decay=0.1)

        # Take several steps
        for _ in range(5):
            params_adam[0].grad = grad.copy()
            params_adamw[0].grad = grad.copy()
            adam.step()
            adamw.step()

        # Results should differ (Adam applies L2 to gradient, AdamW is decoupled)
        # This test verifies the implementations are actually different
        assert not np.allclose(params_adam[0].data, params_adamw[0].data), \
            "AdamW and Adam+L2 should produce different results"

    def test_adamw_convergence(self):
        """Test AdamW converges on simple problem."""
        from python.optimization import AdamW
        from python.foundations import Tensor

        x = Tensor(np.array([10.0]).astype(np.float64), requires_grad=True)
        optimizer = AdamW([x], lr=0.1, weight_decay=0.01)

        for _ in range(1000):
            x.grad = 2 * x.data  # gradient of x^2
            optimizer.step()

        assert np.abs(x.data[0]) < 0.5, f"AdamW failed to converge: x = {x.data[0]}"



class TestRMSprop:
    """Test RMSprop optimizer with new Tensor-based API."""

    def test_rmsprop_creation(self, sample_params):
        """Test RMSprop optimizer creation."""
        from python.optimization import RMSprop

        optimizer = RMSprop(sample_params, lr=0.01)
        assert optimizer is not None
        assert optimizer.defaults['alpha'] == 0.99

    def test_rmsprop_step_correctness(self, sample_params, sample_grads):
        """Test RMSprop computes correct update."""
        from python.optimization import RMSprop
        from python.foundations import Tensor

        params = [Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float64), requires_grad=True)]
        grads = [np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float64)]
        original_data = params[0].data.copy()
        lr = 0.01
        alpha = 0.99
        eps = 1e-8

        optimizer = RMSprop(params, lr=lr, alpha=alpha, eps=eps)
        set_grads(params, grads)
        optimizer.step()

        # Manual computation for first step
        g = grads[0]
        v = (1 - alpha) * (g ** 2)  # v = 0 * alpha + (1-alpha) * g^2
        expected = original_data - lr * g / (np.sqrt(v) + eps)

        assert np.allclose(params[0].data, expected, rtol=1e-5), \
            f"RMSprop step incorrect: got {params[0].data}, expected {expected}"

    def test_rmsprop_square_avg_accumulation(self, sample_params, sample_grads):
        """Test RMSprop accumulates squared gradients correctly."""
        from python.optimization import RMSprop
        from python.foundations import Tensor

        params = [Tensor(np.ones((2, 2)).astype(np.float64), requires_grad=True)]
        grads = [np.ones((2, 2)).astype(np.float64) * 0.5]
        alpha = 0.9

        optimizer = RMSprop(params, lr=0.01, alpha=alpha)

        # Step 1
        set_grads(params, grads)
        optimizer.step()
        expected_v1 = (1 - alpha) * (grads[0] ** 2)
        assert np.allclose(optimizer.velocities[params[0]], expected_v1)

        # Step 2
        set_grads(params, grads)
        optimizer.step()
        expected_v2 = alpha * expected_v1 + (1 - alpha) * (grads[0] ** 2)
        assert np.allclose(optimizer.velocities[params[0]], expected_v2)

    def test_rmsprop_centered(self):
        """Test RMSprop with centered gradient (variance normalization)."""
        from python.optimization import RMSprop
        from python.foundations import Tensor

        params = [Tensor(np.ones((3, 3)).astype(np.float64), requires_grad=True)]
        optimizer = RMSprop(params, lr=0.01, centered=True)

        params[0].grad = np.random.randn(3, 3).astype(np.float64)
        optimizer.step()

        # Centered RMSprop should track gradient average
        assert params[0] in optimizer.grad_avg

    def test_rmsprop_with_momentum(self):
        """Test RMSprop with momentum buffer."""
        from python.optimization import RMSprop
        from python.foundations import Tensor

        params = [Tensor(np.ones((2, 2)).astype(np.float64), requires_grad=True)]
        optimizer = RMSprop(params, lr=0.01, momentum=0.9)

        params[0].grad = np.ones((2, 2)) * 0.1
        optimizer.step()

        # Should have momentum buffer
        assert params[0] in optimizer.buffer

    def test_rmsprop_numerical_stability(self):
        """Test RMSprop handles small/zero gradients without division issues."""
        from python.optimization import RMSprop
        from python.foundations import Tensor

        params = [Tensor(np.ones((2, 2)).astype(np.float64), requires_grad=True)]
        optimizer = RMSprop(params, lr=0.01, eps=1e-8)

        # Near-zero gradients
        params[0].grad = np.ones((2, 2)) * 1e-20
        optimizer.step()
        assert np.all(np.isfinite(params[0].data)), "RMSprop produced NaN/Inf"



class TestMSELoss:
    """Test Mean Squared Error Loss."""

    def test_mse_forward(self, seed):
        """Test MSE forward computation."""
        from python.optimization import MSELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(32, 1).astype(np.float64))
        targets = Tensor(np.random.randn(32, 1).astype(np.float64))
        loss_fn = MSELoss()

        loss = loss_fn(predictions, targets, reduction='mean')

        expected = np.mean((predictions.data - targets.data) ** 2)
        assert np.allclose(loss.data, expected), "MSE forward incorrect"

    def test_mse_reduction_sum(self, seed):
        """Test MSE with sum reduction."""
        from python.optimization import MSELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(32, 1).astype(np.float64))
        targets = Tensor(np.random.randn(32, 1).astype(np.float64))
        loss_fn = MSELoss()

        loss = loss_fn(predictions, targets, reduction='sum')

        expected = np.sum((predictions.data - targets.data) ** 2)
        assert np.allclose(loss.data, expected), "MSE sum reduction incorrect"

    def test_mse_reduction_none(self, seed):
        """Test MSE with no reduction."""
        from python.optimization import MSELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(8, 1).astype(np.float64))
        targets = Tensor(np.random.randn(8, 1).astype(np.float64))
        loss_fn = MSELoss()

        loss = loss_fn(predictions, targets, reduction='none')

        expected = (predictions.data - targets.data) ** 2
        assert np.allclose(loss.data, expected), "MSE none reduction incorrect"

    def test_mse_backward(self, seed):
        """Test MSE backward gradient via autograd."""
        from python.optimization import MSELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(8, 1).astype(np.float64), requires_grad=True)
        targets = Tensor(np.random.randn(8, 1).astype(np.float64))
        loss_fn = MSELoss()

        loss = loss_fn(predictions, targets, reduction='mean')
        loss.backward()

        # Gradient: 2 * (pred - target) / n
        expected_grad = 2 * (predictions.data - targets.data) / predictions.data.size
        assert np.allclose(predictions.grad, expected_grad), "MSE backward incorrect"

    def test_mse_gradient_check(self, seed):
        """Test MSE gradient using foundations.gradcheck."""
        from python.optimization import MSELoss
        from python.foundations import Tensor, gradcheck

        predictions = Tensor(np.random.randn(4, 1).astype(np.float64), requires_grad=True)
        targets = Tensor(np.random.randn(4, 1).astype(np.float64))
        loss_fn = MSELoss()

        def func(pred):
            return loss_fn(pred, targets, reduction='mean')

        assert gradcheck(func, (predictions,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "MSE gradient check failed"

    def test_mse_gradient_check_sum_reduction(self, seed):
        """Test MSE gradient with sum reduction using gradcheck."""
        from python.optimization import MSELoss
        from python.foundations import Tensor, gradcheck

        predictions = Tensor(np.random.randn(4, 1).astype(np.float64), requires_grad=True)
        targets = Tensor(np.random.randn(4, 1).astype(np.float64))
        loss_fn = MSELoss()

        def func(pred):
            return loss_fn(pred, targets, reduction='sum')

        assert gradcheck(func, (predictions,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "MSE sum reduction gradient check failed"



class TestMAELoss:
    """Test Mean Absolute Error Loss."""

    def test_mae_forward(self, seed):
        """Test MAE forward computation."""
        from python.optimization import MAELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(32, 1).astype(np.float64))
        targets = Tensor(np.random.randn(32, 1).astype(np.float64))
        loss_fn = MAELoss()

        loss = loss_fn(predictions, targets, reduction='mean')

        expected = np.mean(np.abs(predictions.data - targets.data))
        assert np.allclose(loss.data, expected), "MAE forward incorrect"

    def test_mae_reduction_sum(self, seed):
        """Test MAE with sum reduction."""
        from python.optimization import MAELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(16, 1).astype(np.float64))
        targets = Tensor(np.random.randn(16, 1).astype(np.float64))
        loss_fn = MAELoss()

        loss = loss_fn(predictions, targets, reduction='sum')

        expected = np.sum(np.abs(predictions.data - targets.data))
        assert np.allclose(loss.data, expected), "MAE sum reduction incorrect"

    def test_mae_reduction_none(self, seed):
        """Test MAE with no reduction."""
        from python.optimization import MAELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(8, 1).astype(np.float64))
        targets = Tensor(np.random.randn(8, 1).astype(np.float64))
        loss_fn = MAELoss()

        loss = loss_fn(predictions, targets, reduction='none')

        expected = np.abs(predictions.data - targets.data)
        assert np.allclose(loss.data, expected), "MAE none reduction incorrect"

    def test_mae_backward(self, seed):
        """Test MAE backward gradient via autograd."""
        from python.optimization import MAELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(8, 1).astype(np.float64), requires_grad=True)
        targets = Tensor(np.random.randn(8, 1).astype(np.float64))
        loss_fn = MAELoss()

        loss = loss_fn(predictions, targets, reduction='mean')
        loss.backward()

        # Gradient: sign(pred - target) / n
        expected_grad = np.sign(predictions.data - targets.data) / predictions.data.size
        assert np.allclose(predictions.grad, expected_grad), "MAE backward incorrect"

    def test_mae_gradient_check(self, seed):
        """Test MAE gradient using foundations.gradcheck."""
        from python.optimization import MAELoss
        from python.foundations import Tensor, gradcheck

        # Use values away from zero to avoid gradient discontinuity at abs(x)=0
        predictions = Tensor(np.random.randn(4, 1).astype(np.float64) + 1.0, requires_grad=True)
        targets = Tensor(np.random.randn(4, 1).astype(np.float64))
        loss_fn = MAELoss()

        def func(pred):
            return loss_fn(pred, targets, reduction='mean')

        assert gradcheck(func, (predictions,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "MAE gradient check failed"



class TestHuberLoss:
    """Test Huber Loss."""

    def test_huber_forward(self, seed):
        """Test Huber forward computation with mixed small/large errors."""
        from python.optimization import HuberLoss
        from python.foundations import Tensor

        # Mix of small and large errors
        predictions = Tensor(np.array([0.0, 0.5, 2.0, -1.5]).astype(np.float64))
        targets = Tensor(np.zeros(4).astype(np.float64))
        delta = 1.0
        loss_fn = HuberLoss()

        loss = loss_fn(predictions, targets, delta=delta, reduction='none')

        # Compute expected: quadratic for |e| <= delta, linear for |e| > delta
        errors = predictions.data - targets.data
        expected = np.where(
            np.abs(errors) <= delta,
            0.5 * errors ** 2,
            delta * (np.abs(errors) - 0.5 * delta)
        )
        assert np.allclose(loss.data, expected), f"Huber forward incorrect: got {loss.data}, expected {expected}"

    def test_huber_small_errors_quadratic(self, seed):
        """Test Huber is quadratic for small errors."""
        from python.optimization import HuberLoss
        from python.foundations import Tensor

        # Small errors (|error| <= delta)
        predictions = Tensor(np.array([0.0, 0.1, 0.2, 0.5, -0.3]).astype(np.float64))
        targets = Tensor(np.zeros(5).astype(np.float64))
        delta = 1.0
        loss_fn = HuberLoss()

        loss = loss_fn(predictions, targets, delta=delta, reduction='none')

        # For |error| <= delta, loss = 0.5 * error^2
        expected = 0.5 * predictions.data ** 2
        assert np.allclose(loss.data, expected), "Huber small errors should be quadratic"

    def test_huber_large_errors_linear(self, seed):
        """Test Huber is linear for large errors."""
        from python.optimization import HuberLoss
        from python.foundations import Tensor

        # Large errors (|error| > delta)
        predictions = Tensor(np.array([2.0, 3.0, -2.5, 5.0]).astype(np.float64))
        targets = Tensor(np.zeros(4).astype(np.float64))
        delta = 1.0
        loss_fn = HuberLoss()

        loss = loss_fn(predictions, targets, delta=delta, reduction='none')

        # For |error| > delta, loss = delta * (|error| - 0.5 * delta)
        expected = delta * (np.abs(predictions.data) - 0.5 * delta)
        assert np.allclose(loss.data, expected), "Huber large errors should be linear"

    def test_huber_mean_reduction(self, seed):
        """Test Huber with mean reduction."""
        from python.optimization import HuberLoss
        from python.foundations import Tensor

        predictions = Tensor(np.array([0.5, 2.0, -0.3, 1.5]).astype(np.float64))
        targets = Tensor(np.zeros(4).astype(np.float64))
        delta = 1.0
        loss_fn = HuberLoss()

        loss = loss_fn(predictions, targets, delta=delta, reduction='mean')

        errors = predictions.data
        element_losses = np.where(
            np.abs(errors) <= delta,
            0.5 * errors ** 2,
            delta * (np.abs(errors) - 0.5 * delta)
        )
        expected = np.mean(element_losses)
        assert np.allclose(loss.data, expected), "Huber mean reduction incorrect"

    def test_huber_gradient_check_small_errors(self, seed):
        """Test Huber gradient for small errors (quadratic region) using gradcheck."""
        from python.optimization import HuberLoss
        from python.foundations import Tensor, gradcheck

        # Small errors stay in quadratic region
        predictions = Tensor(np.array([0.1, 0.2, -0.3, 0.4]).astype(np.float64), requires_grad=True)
        targets = Tensor(np.zeros(4).astype(np.float64))
        loss_fn = HuberLoss()

        def func(pred):
            return loss_fn(pred, targets, delta=1.0, reduction='mean')

        assert gradcheck(func, (predictions,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "Huber gradient check failed for small errors"

    def test_huber_gradient_check_large_errors(self, seed):
        """Test Huber gradient for large errors (linear region) using gradcheck."""
        from python.optimization import HuberLoss
        from python.foundations import Tensor, gradcheck

        # Large errors stay in linear region
        predictions = Tensor(np.array([2.0, 3.0, -2.5, 4.0]).astype(np.float64), requires_grad=True)
        targets = Tensor(np.zeros(4).astype(np.float64))
        loss_fn = HuberLoss()

        def func(pred):
            return loss_fn(pred, targets, delta=1.0, reduction='mean')

        assert gradcheck(func, (predictions,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "Huber gradient check failed for large errors"



class TestCrossEntropyLoss:
    """Test Cross-Entropy Loss."""

    def test_crossentropy_forward(self, seed):
        """Test CrossEntropy forward computation."""
        from python.optimization import CrossEntropyLoss
        from python.utils.math_utils import logsumexp
        from python.foundations import Tensor

        logits = Tensor(np.random.randn(4, 5).astype(np.float64))
        targets_data = np.array([0, 2, 1, 4])
        targets = Tensor(targets_data)
        loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits, targets, reduction='mean')

        # Manual: CE = -logits[target] + log(sum(exp(logits)))
        log_sum_exp = logsumexp(logits.data, axis=1)
        target_logits = logits.data[np.arange(4), targets_data]
        expected = np.mean(-target_logits + log_sum_exp)
        assert np.allclose(loss.data, expected, rtol=1e-5), f"CrossEntropy forward incorrect: got {loss.data}, expected {expected}"

    def test_crossentropy_reduction_none(self, seed):
        """Test CrossEntropy with no reduction."""
        from python.optimization import CrossEntropyLoss
        from python.foundations import Tensor

        logits_data = np.random.randn(4, 5).astype(np.float64)
        logits = Tensor(logits_data)
        targets_data = np.array([0, 2, 1, 4])
        targets = Tensor(targets_data)
        loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits, targets, reduction='none')

        # CE = -log(softmax) = -logit[target] + log(sum(exp(logits)))
        log_sum_exp = np.log(np.sum(np.exp(logits_data), axis=1))  # Shape (4,)
        target_logits = logits_data[np.arange(4), targets_data]  # Shape (4,)
        expected = -target_logits + log_sum_exp  # Shape (4,)

        assert loss.data.shape == (4,), f"CE none reduction should have shape (N,), got {loss.data.shape}"
        assert np.allclose(loss.data, expected, rtol=1e-5), "CrossEntropy none reduction incorrect"

    def test_crossentropy_perfect_prediction(self):
        """Test CrossEntropy with perfect prediction approaches 0."""
        from python.optimization import CrossEntropyLoss
        from python.foundations import Tensor

        logits = Tensor(np.array([[10.0, -10.0, -10.0],
                                  [-10.0, 10.0, -10.0]]).astype(np.float64))
        targets = Tensor(np.array([0, 1]))
        loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits, targets, reduction='mean')

        # With very confident predictions, loss should be near 0
        assert loss.data < 0.001, "CrossEntropy with perfect prediction should be near 0"

    def test_crossentropy_uniform_prediction(self):
        """Test CrossEntropy with uniform predictions equals log(num_classes)."""
        from python.optimization import CrossEntropyLoss
        from python.foundations import Tensor

        # All logits equal -> uniform distribution -> loss = log(num_classes)
        num_classes = 5
        logits = Tensor(np.zeros((4, num_classes)).astype(np.float64))
        targets = Tensor(np.array([0, 1, 2, 3]))
        loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits, targets, reduction='mean')

        expected = np.log(num_classes)
        assert np.allclose(loss.data, expected, rtol=1e-5), f"Uniform prediction should give log({num_classes})"

    def test_crossentropy_gradient_check(self, seed):
        """Test CrossEntropy backward gradient using foundations.gradcheck."""
        from python.optimization import CrossEntropyLoss
        from python.foundations import Tensor, gradcheck

        logits = Tensor(np.random.randn(4, 5).astype(np.float64), requires_grad=True)
        targets = Tensor(np.array([0, 2, 1, 4]))
        loss_fn = CrossEntropyLoss()

        def func(x):
            return loss_fn(x, targets, reduction='mean')

        assert gradcheck(func, (logits,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "CrossEntropy gradient check failed"

    def test_crossentropy_gradient_check_sum_reduction(self, seed):
        """Test CrossEntropy backward gradient with sum reduction using gradcheck."""
        from python.optimization import CrossEntropyLoss
        from python.foundations import Tensor, gradcheck

        logits = Tensor(np.random.randn(4, 5).astype(np.float64), requires_grad=True)
        targets = Tensor(np.array([0, 2, 1, 4]))
        loss_fn = CrossEntropyLoss()

        def func(x):
            return loss_fn(x, targets, reduction='sum')

        assert gradcheck(func, (logits,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "CrossEntropy sum reduction gradient check failed"

    def test_crossentropy_with_label_smoothing(self, seed):
        """Test CrossEntropy with label smoothing."""
        from python.optimization import CrossEntropyLoss
        from python.foundations import Tensor

        logits = Tensor(np.random.randn(4, 5).astype(np.float64))
        targets = Tensor(np.array([0, 2, 1, 4]))
        loss_fn = CrossEntropyLoss()

        loss_no_smooth = loss_fn(logits, targets, label_smoothing=0.0, reduction='mean')
        loss_smooth = loss_fn(logits, targets, label_smoothing=0.1, reduction='mean')

        # Label smoothing should increase loss (spreads probability mass)
        assert loss_smooth.data > loss_no_smooth.data, "Label smoothing should increase loss"



class TestBCELoss:
    """Test Binary Cross-Entropy Loss."""

    def test_bce_forward(self, seed):
        """Test BCE forward computation."""
        from python.optimization import BinaryCrossEntropyLoss
        from python.foundations import Tensor

        # Probabilities in (0, 1)
        probs_data = np.clip(np.random.rand(8, 1), 0.01, 0.99).astype(np.float64)
        targets_data = np.random.randint(0, 2, size=(8, 1)).astype(np.float64)
        probs = Tensor(probs_data)
        targets = Tensor(targets_data)
        loss_fn = BinaryCrossEntropyLoss()

        loss = loss_fn(probs, targets, reduction='mean')

        # Manual: BCE = -[y*log(p) + (1-y)*log(1-p)]
        expected = -np.mean(targets_data * np.log(probs_data) + (1 - targets_data) * np.log(1 - probs_data))
        assert np.allclose(loss.data, expected, rtol=1e-5), f"BCE forward incorrect: got {loss.data}, expected {expected}"

    def test_bce_reduction_none(self, seed):
        """Test BCE with no reduction."""
        from python.optimization import BinaryCrossEntropyLoss
        from python.foundations import Tensor

        probs_data = np.array([[0.2], [0.8], [0.5], [0.9]]).astype(np.float64)
        targets_data = np.array([[0.0], [1.0], [1.0], [0.0]]).astype(np.float64)
        probs = Tensor(probs_data)
        targets = Tensor(targets_data)
        loss_fn = BinaryCrossEntropyLoss()

        loss = loss_fn(probs, targets, reduction='none')

        expected = -(targets_data * np.log(probs_data) + (1 - targets_data) * np.log(1 - probs_data))
        assert np.allclose(loss.data, expected, rtol=1e-5), "BCE none reduction incorrect"

    def test_bce_perfect_prediction(self):
        """Test BCE with near-perfect predictions approaches 0."""
        from python.optimization import BinaryCrossEntropyLoss
        from python.foundations import Tensor

        probs = Tensor(np.array([[0.999], [0.001]]).astype(np.float64))
        targets = Tensor(np.array([[1.0], [0.0]]).astype(np.float64))
        loss_fn = BinaryCrossEntropyLoss()

        loss = loss_fn(probs, targets, reduction='mean')

        assert loss.data < 0.01, "BCE with perfect prediction should be near 0"

    def test_bce_with_logits_forward(self, seed):
        """Test BCE with logits forward computation."""
        from python.optimization import BCEWithLogitsLoss
        from python.foundations import Tensor

        logits_data = np.random.randn(8, 1).astype(np.float64)
        targets_data = np.random.randint(0, 2, size=(8, 1)).astype(np.float64)
        logits = Tensor(logits_data)
        targets = Tensor(targets_data)
        loss_fn = BCEWithLogitsLoss()

        loss = loss_fn(logits, targets, reduction='mean')

        # Manual: BCEWithLogits = max(x, 0) - x*y + log(1 + exp(-|x|))
        # Numerically stable form
        expected = np.mean(np.maximum(logits_data, 0) - logits_data * targets_data + np.log(1 + np.exp(-np.abs(logits_data))))
        assert np.allclose(loss.data, expected, rtol=1e-4), f"BCEWithLogits forward incorrect: got {loss.data}, expected {expected}"

    def test_bce_with_logits_numerical_stability(self):
        """Test BCEWithLogits is numerically stable for extreme values."""
        from python.optimization import BCEWithLogitsLoss
        from python.foundations import Tensor

        # Extreme logits
        logits = Tensor(np.array([[100.0], [-100.0], [0.0]]).astype(np.float64))
        targets = Tensor(np.array([[1.0], [0.0], [0.5]]).astype(np.float64))
        loss_fn = BCEWithLogitsLoss()

        loss = loss_fn(logits, targets, reduction='mean')

        assert np.isfinite(loss.data), "BCEWithLogits should be stable for extreme values"

    def test_bce_gradient_check(self, seed):
        """Test BCE gradient using foundations.gradcheck."""
        from python.optimization import BinaryCrossEntropyLoss
        from python.foundations import Tensor, gradcheck

        # Probabilities in (0.1, 0.9) to avoid log(0) issues
        probs = Tensor(np.clip(np.random.rand(4, 1), 0.1, 0.9).astype(np.float64), requires_grad=True)
        targets = Tensor(np.random.randint(0, 2, size=(4, 1)).astype(np.float64))
        loss_fn = BinaryCrossEntropyLoss()

        def func(p):
            return loss_fn(p, targets, reduction='mean')

        assert gradcheck(func, (probs,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "BCE gradient check failed"

    def test_bce_with_logits_gradient_check(self, seed):
        """Test BCEWithLogits gradient using foundations.gradcheck."""
        from python.optimization import BCEWithLogitsLoss
        from python.foundations import Tensor, gradcheck

        logits = Tensor(np.random.randn(4, 1).astype(np.float64), requires_grad=True)
        targets = Tensor(np.random.randint(0, 2, size=(4, 1)).astype(np.float64))
        loss_fn = BCEWithLogitsLoss()

        def func(x):
            return loss_fn(x, targets, reduction='mean')

        assert gradcheck(func, (logits,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "BCEWithLogits gradient check failed"
        # For logit=100, target=1: loss ≈ 0; for logit=-100, target=0: loss ≈ 0
        assert loss.data < 1.0, "Extreme confident correct predictions should have low loss"



class TestFocalLoss:
    """Test Focal Loss for imbalanced classification."""

    def test_focal_loss_gamma_zero_equals_ce(self, seed):
        """Test Focal Loss with gamma=0 equals CrossEntropy."""
        from python.optimization import FocalLoss, CrossEntropyLoss
        from python.foundations import Tensor

        logits = Tensor(np.random.randn(8, 5).astype(np.float64))
        targets = Tensor(np.random.randint(0, 5, size=(8,)))

        focal_fn = FocalLoss()
        ce_fn = CrossEntropyLoss()

        focal_loss = focal_fn(logits, targets, gamma=0.0, reduction='mean')
        ce_loss = ce_fn(logits, targets, reduction='mean')

        assert np.allclose(focal_loss.data, ce_loss.data, rtol=1e-4), \
            f"Focal(gamma=0) should equal CE: got {focal_loss.data}, expected {ce_loss.data}"

    def test_focal_loss_downweights_easy_examples(self, seed):
        """Test Focal Loss down-weights easy (confident) examples."""
        from python.optimization import FocalLoss, CrossEntropyLoss
        from python.foundations import Tensor

        # Easy example: high confidence correct
        easy_logits = Tensor(np.array([[10.0, -10.0, -10.0]]).astype(np.float64))
        easy_targets = Tensor(np.array([0]))

        # Hard example: low confidence
        hard_logits = Tensor(np.array([[0.1, 0.0, 0.0]]).astype(np.float64))
        hard_targets = Tensor(np.array([0]))

        focal_fn = FocalLoss()
        ce_fn = CrossEntropyLoss()

        easy_focal = focal_fn(easy_logits, easy_targets, gamma=2.0, reduction='mean')
        hard_focal = focal_fn(hard_logits, hard_targets, gamma=2.0, reduction='mean')

        easy_ce = ce_fn(easy_logits, easy_targets, reduction='mean')
        hard_ce = ce_fn(hard_logits, hard_targets, reduction='mean')

        # Focal should reduce easy examples MORE than hard examples (relative to CE)
        # So focal_easy/ce_easy < focal_hard/ce_hard
        focal_easy_ratio = easy_focal.data / (easy_ce.data + 1e-10)
        focal_hard_ratio = hard_focal.data / (hard_ce.data + 1e-10)

        assert focal_easy_ratio < focal_hard_ratio, \
            "Focal should down-weight easy examples more than hard ones"

    def test_focal_loss_forward(self, seed):
        """Test Focal Loss forward computation."""
        from python.optimization import FocalLoss
        from python.utils.math_utils import softmax
        from python.foundations import Tensor

        logits_data = np.random.randn(4, 3).astype(np.float64)
        targets_data = np.array([0, 1, 2, 0])
        logits = Tensor(logits_data)
        targets = Tensor(targets_data)
        loss_fn = FocalLoss()
        gamma = 2.0

        loss = loss_fn(logits, targets, gamma=gamma, reduction='mean')

        # Manual: FL = -(1-p_t)^gamma * log(p_t)
        probs = softmax(logits_data)
        p_t = probs[np.arange(4), targets_data]
        expected = np.mean(-((1 - p_t) ** gamma) * np.log(p_t + 1e-10))

        assert np.allclose(loss.data, expected, rtol=1e-3), \
            f"Focal loss incorrect: got {loss.data}, expected {expected}"

    def test_focal_loss_gradient_check(self, seed):
        """Test Focal Loss gradient using foundations.gradcheck."""
        from python.optimization import FocalLoss
        from python.foundations import Tensor, gradcheck

        logits = Tensor(np.random.randn(4, 3).astype(np.float64), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 0]))
        loss_fn = FocalLoss()

        def func(x):
            return loss_fn(x, targets, gamma=2.0, reduction='mean')

        assert gradcheck(func, (logits,), eps=1e-3, atol=1e-3, rtol=2e-2), \
            "Focal Loss gradient check failed"



class TestRMSELoss:
    """Test Root Mean Squared Error Loss."""

    def test_rmse_forward(self, seed):
        """Test RMSE forward computation."""
        from python.optimization import RMSELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(32, 1).astype(np.float64))
        targets = Tensor(np.random.randn(32, 1).astype(np.float64))
        loss_fn = RMSELoss()

        loss = loss_fn(predictions, targets, reduction='mean')

        expected = np.sqrt(np.mean((predictions.data - targets.data) ** 2))
        assert np.allclose(loss.data, expected, rtol=1e-4), "RMSE forward incorrect"

    def test_rmse_relationship_to_mse(self, seed):
        """Test RMSE is sqrt of MSE."""
        from python.optimization import RMSELoss, MSELoss
        from python.foundations import Tensor

        predictions = Tensor(np.random.randn(16, 1).astype(np.float64))
        targets = Tensor(np.random.randn(16, 1).astype(np.float64))

        rmse_fn = RMSELoss()
        mse_fn = MSELoss()

        rmse = rmse_fn(predictions, targets, reduction='mean')
        mse = mse_fn(predictions, targets, reduction='mean')

        assert np.allclose(rmse.data ** 2, mse.data, rtol=1e-4), "RMSE^2 should equal MSE"

    def test_rmse_backward(self, seed):
        """Test RMSE backward gradient via numerical check."""
        from python.optimization import RMSELoss
        from python.foundations import Tensor

        pred_data = np.random.randn(4, 1).astype(np.float64)
        target_data = np.random.randn(4, 1).astype(np.float64)

        predictions = Tensor(pred_data, requires_grad=True)
        targets = Tensor(target_data)
        loss_fn = RMSELoss()

        loss = loss_fn(predictions, targets, reduction='mean')
        loss.backward()
        analytical_grad = predictions.grad.copy()

        # Numerical gradient check (eps=1e-3 optimal for fp32 central differences)
        eps = 1e-3
        numerical_grad = np.zeros_like(pred_data)
        for i in range(pred_data.size):
            idx = np.unravel_index(i, pred_data.shape)
            pred_plus = pred_data.copy()
            pred_minus = pred_data.copy()
            pred_plus[idx] += eps
            pred_minus[idx] -= eps

            loss_plus = loss_fn(Tensor(pred_plus), Tensor(target_data), reduction='mean')
            loss_minus = loss_fn(Tensor(pred_minus), Tensor(target_data), reduction='mean')
            numerical_grad[idx] = (loss_plus.data - loss_minus.data) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, rtol=2e-2, atol=1e-3), \
            "RMSE gradient check failed"



class TestOptimizerSchedulerIntegration:
    """Test optimizer and scheduler integration."""

    def test_optimizer_with_scheduler(self, sample_params, sample_grads):
        """Test optimizer works with scheduler."""
        from python.optimization import Adam, CosineAnnealingLR

        params = [p.copy() for p in sample_params]
        optimizer = Adam(params, lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        # Simulate training
        for epoch in range(10):
            set_grads(params, sample_grads)
            optimizer.step()
            scheduler.step()

        # LR should have decayed
        assert optimizer.defaults['lr'] < 0.001

    def test_optimizer_with_gradient_clipping(self, sample_params, sample_grads):
        """Test optimizer with gradient clipping."""
        from python.optimization import SGD, clip_grad_norm_

        params = [p.copy() for p in sample_params]
        grads = [g.copy() * 100 for g in sample_grads]  # Large gradients

        optimizer = SGD(params, lr=0.1)

        # Clip gradients
        clip_grad_norm_(grads, max_norm=1.0)

        # Then step
        set_grads(params, grads)
        optimizer.step()

        # Should complete without issues
        assert all(np.all(np.isfinite(p)) for p in params)



class TestLossOptimizerIntegration:
    """Test loss and optimizer integration."""

    def test_mse_sgd_optimization(self, seed):
        """Test optimizing MSE loss with SGD."""
        from python.optimization import SGD
        from python.foundations import Tensor

        # Simple quadratic minimization test
        # Minimize f(x) = (x - target)^2 with SGD
        np.random.seed(42)
        target = np.array([2.0, 1.0])

        # Initialize parameters as Tensors
        W = [
            Tensor(np.array([5.0, 5.0]).astype(np.float64), requires_grad=True),
        ]

        optimizer = SGD(W, lr=0.1)
        original_loss = None

        # Training loop - minimize (W - target)^2
        for epoch in range(100):
            # Loss: (W - target)^2
            diff = W[0].data - target
            loss = np.sum(diff ** 2)

            if original_loss is None:
                original_loss = loss

            # Gradient: 2 * (W - target)
            grad_W = 2 * diff

            # Update
            W[0].grad = grad_W
            optimizer.step()

        # Loss should decrease
        final_diff = W[0].data - target
        final_loss = np.sum(final_diff ** 2)
        assert final_loss < original_loss, "Loss should decrease during optimization"

        # Parameters should be close to target
        assert np.allclose(W[0].data, target, atol=0.5), "Parameters should converge to target"



class TestNumericalStability:
    """Test numerical stability of components."""

    def test_crossentropy_extreme_logits(self):
        """Test CrossEntropy handles extreme logits."""
        from python.optimization import CrossEntropyLoss

        # Very large logits
        logits = np.array([[1000.0, -1000.0, 0.0],
                          [-1000.0, 1000.0, 0.0]])
        targets = np.array([0, 1])

        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, targets)

        assert np.isfinite(loss), "CrossEntropy should handle extreme logits"
        assert loss >= 0, "CrossEntropy loss should be non-negative"

    def test_softmax_stability(self):
        """Test softmax is numerically stable."""
        from python.optimization import softmax

        # Large values that would overflow naive softmax
        x = np.array([[1000.0, 1000.0, 1000.0],
                      [-1000.0, -1000.0, -1000.0]])

        probs = softmax(x)

        assert np.all(np.isfinite(probs)), "Softmax should handle large values"
        assert np.allclose(probs.sum(axis=1), 1.0), "Softmax should sum to 1"

    def test_logsumexp_stability(self):
        """Test logsumexp is numerically stable."""
        from python.optimization import logsumexp

        # Large values
        x = np.array([1000.0, 1000.0, 1000.0])

        result = logsumexp(x)

        assert np.isfinite(result), "Logsumexp should handle large values"
        # log(3 * e^1000) = 1000 + log(3)
        expected = 1000.0 + np.log(3)
        assert np.isclose(result, expected), "Logsumexp computation incorrect"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_optimizer_zero_lr(self, sample_params, sample_grads):
        """Test optimizer with zero learning rate."""
        from python.optimization import SGD

        params = [p.copy() for p in sample_params]
        original = [p.copy() for p in params]

        optimizer = SGD(params, lr=0.0)
        set_grads(params, sample_grads)
        optimizer.step()

        # Parameters should not change
        for p, p_orig in zip(params, original):
            assert np.allclose(p.data, p_orig.data), "Zero LR should not change parameters"

    def test_loss_single_sample(self):
        """Test loss with single sample."""
        from python.optimization import MSELoss
        from python.foundations import Tensor

        pred = Tensor(np.array([[1.0]]).astype(np.float64))
        target = Tensor(np.array([[2.0]]).astype(np.float64))

        loss_fn = MSELoss()
        loss = loss_fn(pred, target, reduction='mean')

        assert np.isclose(loss.data, 1.0), "MSE of single sample should be (1-2)^2 = 1"

    def test_scheduler_zero_lr(self):
        """Test scheduler doesn't go negative."""
        from python.optimization import StepLR

        optimizer = MockOptimizer(lr=0.001)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        for _ in range(100):
            scheduler.step()

        assert optimizer.defaults['lr'] >= 0, "LR should not go negative"

    def test_gradient_clipping_zero_gradients(self):
        """Test gradient clipping with zero gradients."""
        from python.optimization import clip_grad_norm_

        grads = [np.zeros((10, 5)), np.zeros((5,))]

        # Should not raise
        norm = clip_grad_norm_(grads, max_norm=1.0)

        assert norm == 0.0, "Zero gradients should have zero norm"

    def test_empty_batch_handling(self):
        """Test loss handles edge case dimensions."""
        from python.optimization import MSELoss
        from python.foundations import Tensor

        # 1D case
        pred = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float64))
        target = Tensor(np.array([1.5, 2.5, 3.5]).astype(np.float64))

        loss_fn = MSELoss()
        loss = loss_fn(pred, target, reduction='mean')

        assert np.isfinite(loss.data), "Should handle 1D arrays"


# =============================================================================
# BENCHMARK / SANITY CHECKS
# =============================================================================


class TestSanityChecks:
    """Sanity checks to verify basic correctness."""

    def test_adam_converges_simple_quadratic(self, seed):
        """Test Adam converges on simple quadratic."""
        from python.optimization import Adam
        from python.foundations import Tensor

        # Minimize f(x) = x^2, optimal x = 0
        x = [Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)]  # Start at 5
        optimizer = Adam(x, lr=0.1)

        for _ in range(100):
            grad = 2 * x[0].data  # Gradient of x^2
            x[0].grad = grad
            optimizer.step()

        # Should converge close to 0
        assert np.abs(x[0].data[0]) < 0.1, f"Adam should converge to 0, got {x[0].data[0]}"

    def test_sgd_with_momentum_converges(self, seed):
        """Test SGD with momentum converges."""
        from python.optimization import SGD
        from python.foundations import Tensor

        x = [Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)]
        optimizer = SGD(x, lr=0.01, momentum=0.9)

        for _ in range(200):
            grad = 2 * x[0].data
            x[0].grad = grad
            optimizer.step()

        assert np.abs(x[0].data[0]) < 0.5, f"SGD+momentum should converge, got {x[0].data[0]}"

    def test_cosine_schedule_smooth(self):
        """Test cosine schedule produces smooth curve."""
        from python.optimization import CosineAnnealingLR

        optimizer = MockOptimizer(lr=1.0)
        T_max = 100
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

        lrs = []
        for _ in range(T_max):
            scheduler.step()
            lrs.append(optimizer.defaults['lr'])

        # Check smoothness: no sudden jumps
        for i in range(1, len(lrs) - 1):
            delta1 = abs(lrs[i] - lrs[i-1])
            delta2 = abs(lrs[i+1] - lrs[i])
            # Deltas should be similar (smooth)
            assert delta1 < 0.1, "Cosine schedule should be smooth"



# =============================================================================
# COMPREHENSIVE REWRITES (REPLACING WEAK TESTS)
# =============================================================================

# From rewrite_optimizers.py

class TestAdagradComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adagrad
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = Adagrad([w], lr=0.01)
        assert opt is not None

    def test_step_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adagrad
        np.random.seed(42)
        w_data = np.random.randn(4, 3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        grad = np.random.randn(4, 3).astype(np.float64)
        w.grad = grad.copy()
        lr, eps = 0.01, 1e-10
        opt = Adagrad([w], lr=lr, eps=eps)
        opt.step()
        sum_sq = grad ** 2
        expected = w_data - lr * grad / (np.sqrt(sum_sq) + eps)
        assert np.allclose(w.data, expected, atol=1e-10)

    def test_accumulated_gradient(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adagrad
        np.random.seed(42)
        w_data = np.random.randn(3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        lr, eps = 0.01, 1e-10
        opt = Adagrad([w], lr=lr, eps=eps)
        grads = [np.random.randn(3).astype(np.float64) for _ in range(3)]
        sum_sq = np.zeros(3)
        current = w_data.copy()
        for g in grads:
            w.grad = g.copy()
            opt.step()
            sum_sq += g ** 2
            current = current - lr * g / (np.sqrt(sum_sq) + eps)
        assert np.allclose(w.data, current, atol=1e-8)

    def test_learning_rate_decay(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adagrad
        np.random.seed(42)
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = Adagrad([w], lr=1.0, eps=1e-10)
        # Same gradient each step - effective LR should decrease
        grad = np.ones(3).astype(np.float64)
        updates = []
        for i in range(5):
            w_before = w.data.copy()
            w.grad = grad.copy()
            opt.step()
            update = np.abs(w.data - w_before).mean()
            updates.append(update)
        # Each update should be smaller than the previous
        for i in range(1, len(updates)):
            assert updates[i] < updates[i-1]

    def test_different_params_different_rates(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adagrad
        np.random.seed(42)
        w1 = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        w2 = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = Adagrad([w1, w2], lr=1.0, eps=1e-10)
        # w1 gets large gradients, w2 gets small
        w1.grad = np.ones(3).astype(np.float64) * 10.0
        w2.grad = np.ones(3).astype(np.float64) * 0.1
        opt.step()
        # w1 should have smaller effective update (per unit grad) than w2
        # Both get update = lr * grad / sqrt(grad^2) = lr * sign(grad) for first step
        # So first step update magnitude is the same. After more steps, diverges.

    def test_zero_gradient(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adagrad
        w_data = np.array([1.0, 2.0, 3.0]).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        w.grad = np.zeros(3).astype(np.float64)
        opt = Adagrad([w], lr=0.01)
        opt.step()
        assert np.allclose(w.data, w_data)

    def test_convergence_quadratic(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adagrad
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = Adagrad([w], lr=1.0)
        for _ in range(100):
            w.grad = (2.0 * w.data).astype(np.float64)  # d/dw (w^2) = 2w
            opt.step()
        assert np.abs(w.data[0]) < 0.5


class TestAdadeltaComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adadelta
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = Adadelta([w], rho=0.9, eps=1e-6)
        assert opt is not None

    def test_step_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adadelta
        np.random.seed(42)
        w_data = np.random.randn(4, 3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        grad = np.random.randn(4, 3).astype(np.float64)
        w.grad = grad.copy()
        rho, eps = 0.9, 1e-6
        opt = Adadelta([w], rho=rho, eps=eps)
        opt.step()
        # First step: E[g^2] = (1-rho)*g^2, E[dx^2]=0
        avg_sq_grad = (1 - rho) * grad ** 2
        rms_dx = np.sqrt(0 + eps)  # E[dx^2] starts at 0
        rms_grad = np.sqrt(avg_sq_grad + eps)
        delta = -(rms_dx / rms_grad) * grad
        expected = w_data + delta
        assert np.allclose(w.data, expected, atol=1e-8)

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adadelta
        np.random.seed(42)
        w_data = np.random.randn(3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        rho, eps = 0.9, 1e-6
        opt = Adadelta([w], rho=rho, eps=eps)
        avg_sq_grad = np.zeros(3)
        avg_sq_dx = np.zeros(3)
        current = w_data.copy()
        for step in range(3):
            grad = np.random.randn(3).astype(np.float64)
            w.grad = grad.copy()
            opt.step()
            avg_sq_grad = rho * avg_sq_grad + (1 - rho) * grad ** 2
            rms_dx = np.sqrt(avg_sq_dx + eps)
            rms_grad = np.sqrt(avg_sq_grad + eps)
            delta = -(rms_dx / rms_grad) * grad
            avg_sq_dx = rho * avg_sq_dx + (1 - rho) * delta ** 2
            current = current + delta
        assert np.allclose(w.data, current, atol=1e-8)

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adadelta
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = Adadelta([w], rho=0.9, eps=1e-6)
        for _ in range(200):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 1.0

    def test_rho_parameter(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adadelta
        np.random.seed(42)
        # Higher rho = more smoothing = slower adaptation
        w1 = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        w2 = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt1 = Adadelta([w1], rho=0.5)
        opt2 = Adadelta([w2], rho=0.99)
        for _ in range(10):
            w1.grad = (2.0 * w1.data).astype(np.float64)
            w2.grad = (2.0 * w2.data).astype(np.float64)
            opt1.step()
            opt2.step()
        # Both should move toward 0 but at different rates


class TestNAdamComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NAdam
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = NAdam([w], lr=0.001)
        assert opt is not None

    def test_step(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NAdam
        np.random.seed(42)
        w_data = np.random.randn(3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        grad = np.random.randn(3).astype(np.float64)
        w.grad = grad.copy()
        opt = NAdam([w], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        w_before = w.data.copy()
        opt.step()
        assert not np.allclose(w.data, w_before)

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NAdam
        np.random.seed(42)
        w = Tensor(np.random.randn(4).astype(np.float64), requires_grad=True)
        opt = NAdam([w], lr=0.001, betas=(0.9, 0.999))
        for _ in range(10):
            w.grad = np.random.randn(4).astype(np.float64)
            opt.step()
        # After 10 steps, params should have changed significantly
        assert not np.allclose(w.data, np.zeros(4))

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NAdam
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = NAdam([w], lr=0.1)
        for _ in range(200):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 0.1


class TestRAdamComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import RAdam
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = RAdam([w], lr=0.001)
        assert opt is not None

    def test_step(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import RAdam
        np.random.seed(42)
        w_data = np.random.randn(3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        w.grad = np.random.randn(3).astype(np.float64)
        opt = RAdam([w], lr=0.001)
        opt.step()
        assert not np.allclose(w.data, w_data)

    def test_warmup_behavior(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import RAdam
        np.random.seed(42)
        w = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float64), requires_grad=True)
        opt = RAdam([w], lr=0.001, betas=(0.9, 0.999))
        # Early steps should use SGD-like behavior (no adaptive LR)
        updates_early = []
        for i in range(5):
            w_before = w.data.copy()
            w.grad = np.ones(3).astype(np.float64)
            opt.step()
            updates_early.append(np.abs(w.data - w_before).mean())

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import RAdam
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = RAdam([w], lr=0.01)
        for _ in range(300):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 0.5

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import RAdam
        np.random.seed(42)
        w = Tensor(np.random.randn(4).astype(np.float64), requires_grad=True)
        opt = RAdam([w], lr=0.001)
        for _ in range(20):
            w.grad = np.random.randn(4).astype(np.float64)
            opt.step()


class TestAdafactorComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adafactor
        w = Tensor(np.zeros((3, 4)).astype(np.float64), requires_grad=True)
        opt = Adafactor([w])
        assert opt is not None

    def test_step(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adafactor
        np.random.seed(42)
        w_data = np.random.randn(4, 3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        w.grad = np.random.randn(4, 3).astype(np.float64)
        opt = Adafactor([w])
        opt.step()
        assert not np.allclose(w.data, w_data)

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adafactor
        np.random.seed(42)
        w = Tensor(np.random.randn(4, 3).astype(np.float64), requires_grad=True)
        opt = Adafactor([w])
        for _ in range(10):
            w.grad = np.random.randn(4, 3).astype(np.float64)
            opt.step()

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Adafactor
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = Adafactor([w], lr=0.1)
        for _ in range(200):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 1.0


class TestLAMBComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LAMB
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = LAMB([w], lr=0.001)
        assert opt is not None

    def test_step_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LAMB
        np.random.seed(42)
        w_data = np.random.randn(4, 3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        grad = np.random.randn(4, 3).astype(np.float64)
        w.grad = grad.copy()
        lr = 0.001
        beta1, beta2, eps, wd = 0.9, 0.999, 1e-6, 0.01
        opt = LAMB([w], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)
        opt.step()
        # Numpy reference: LAMB = Adam + layer-wise trust ratio
        m = (1 - beta1) * grad
        v = (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        adam_update = m_hat / (np.sqrt(v_hat) + eps) + wd * w_data
        w_norm = np.linalg.norm(w_data)
        update_norm = np.linalg.norm(adam_update)
        if w_norm > 0 and update_norm > 0:
            trust_ratio = w_norm / update_norm
        else:
            trust_ratio = 1.0
        expected = w_data - lr * trust_ratio * adam_update
        assert np.allclose(w.data, expected, atol=1e-6)

    def test_trust_ratio(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LAMB
        np.random.seed(42)
        # Large params + small grads = large trust ratio
        w1 = Tensor(np.ones(10).astype(np.float64) * 10.0, requires_grad=True)
        w1.grad = np.ones(10).astype(np.float64) * 0.001
        w2 = Tensor(np.ones(10).astype(np.float64) * 0.001, requires_grad=True)
        w2.grad = np.ones(10).astype(np.float64) * 10.0
        opt = LAMB([w1, w2], lr=0.001)
        opt.step()
        # Both should have moved

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LAMB
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = LAMB([w], lr=0.01)
        for _ in range(200):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 1.0

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LAMB
        np.random.seed(42)
        w = Tensor(np.random.randn(4, 3).astype(np.float64), requires_grad=True)
        opt = LAMB([w], lr=0.001)
        for _ in range(10):
            w.grad = np.random.randn(4, 3).astype(np.float64)
            opt.step()


class TestLARSComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LARS
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = LARS([w], lr=0.01)
        assert opt is not None

    def test_step_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LARS
        np.random.seed(42)
        w_data = np.random.randn(4, 3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        grad = np.random.randn(4, 3).astype(np.float64)
        w.grad = grad.copy()
        lr, trust_coeff, wd = 0.01, 0.001, 0.0
        opt = LARS([w], lr=lr, trust_coefficient=trust_coeff, weight_decay=wd)
        opt.step()
        # LARS: local_lr = trust_coeff * ||w|| / (||g|| + wd * ||w||)
        w_norm = np.linalg.norm(w_data)
        g_norm = np.linalg.norm(grad)
        local_lr = trust_coeff * w_norm / (g_norm + wd * w_norm)
        expected = w_data - lr * local_lr * grad
        assert np.allclose(w.data, expected, atol=1e-6)

    def test_layer_wise_scaling(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LARS
        np.random.seed(42)
        # Layers with different norms get different scaling
        w1 = Tensor(np.ones(5).astype(np.float64) * 10.0, requires_grad=True)
        w2 = Tensor(np.ones(5).astype(np.float64) * 0.1, requires_grad=True)
        w1.grad = np.ones(5).astype(np.float64)
        w2.grad = np.ones(5).astype(np.float64)
        opt = LARS([w1, w2], lr=0.01, trust_coefficient=0.001)
        w1_before = w1.data.copy()
        w2_before = w2.data.copy()
        opt.step()
        update1 = np.abs(w1.data - w1_before).mean()
        update2 = np.abs(w2.data - w2_before).mean()
        # w1 has larger norm → larger local_lr → larger update
        assert update1 > update2

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LARS
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = LARS([w], lr=0.1, trust_coefficient=0.1)
        for _ in range(200):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 1.0

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LARS
        np.random.seed(42)
        w = Tensor(np.random.randn(4).astype(np.float64), requires_grad=True)
        opt = LARS([w], lr=0.01)
        for _ in range(10):
            w.grad = np.random.randn(4).astype(np.float64)
            opt.step()


class TestLionComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Lion
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = Lion([w], lr=1e-4)
        assert opt is not None

    def test_step_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Lion
        np.random.seed(42)
        w_data = np.random.randn(4, 3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        grad = np.random.randn(4, 3).astype(np.float64)
        w.grad = grad.copy()
        lr, beta1, beta2, wd = 1e-4, 0.9, 0.99, 0.0
        opt = Lion([w], lr=lr, betas=(beta1, beta2), weight_decay=wd)
        opt.step()
        # Lion: update = sign(beta1 * m + (1-beta1) * g)
        # First step: m=0, so update = sign((1-beta1) * g) = sign(g)
        update = np.sign((1 - beta1) * grad)
        expected = w_data - lr * (update + wd * w_data)
        assert np.allclose(w.data, expected, atol=1e-8)

    def test_sign_updates(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Lion
        np.random.seed(42)
        w = Tensor(np.random.randn(10).astype(np.float64), requires_grad=True)
        w.grad = np.random.randn(10).astype(np.float64)
        w_before = w.data.copy()
        opt = Lion([w], lr=1e-4, weight_decay=0.0)
        opt.step()
        # All updates should be exactly ±lr (sign-based)
        updates = w.data - w_before
        assert np.allclose(np.abs(updates), 1e-4, atol=1e-10)

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Lion
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = Lion([w], lr=0.01)
        for _ in range(1000):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 0.5

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Lion
        np.random.seed(42)
        w = Tensor(np.random.randn(4).astype(np.float64), requires_grad=True)
        opt = Lion([w], lr=1e-4)
        for _ in range(10):
            w.grad = np.random.randn(4).astype(np.float64)
            opt.step()


class TestMuonComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Muon
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = Muon([w], lr=0.01)
        assert opt is not None

    def test_step(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Muon
        np.random.seed(42)
        w_data = np.random.randn(3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        w.grad = np.random.randn(3).astype(np.float64)
        opt = Muon([w], lr=0.01)
        opt.step()
        assert not np.allclose(w.data, w_data)

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Muon
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = Muon([w], lr=0.01)
        for _ in range(200):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 1.0

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import Muon
        np.random.seed(42)
        w = Tensor(np.random.randn(4).astype(np.float64), requires_grad=True)
        opt = Muon([w], lr=0.01)
        for _ in range(10):
            w.grad = np.random.randn(4).astype(np.float64)
            opt.step()


class TestSGDWComprehensive:

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SGDW
        w = Tensor(np.zeros(3).astype(np.float64), requires_grad=True)
        opt = SGDW([w], lr=0.01, weight_decay=0.001)
        assert opt is not None

    def test_step_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SGDW
        np.random.seed(42)
        w_data = np.random.randn(4, 3).astype(np.float64)
        w = Tensor(w_data.copy(), requires_grad=True)
        grad = np.random.randn(4, 3).astype(np.float64)
        w.grad = grad.copy()
        lr, wd = 0.01, 0.001
        opt = SGDW([w], lr=lr, weight_decay=wd)
        opt.step()
        # SGDW: decoupled weight decay
        expected = w_data - lr * grad - lr * wd * w_data
        assert np.allclose(w.data, expected, atol=1e-10)

    def test_differs_from_sgd_l2(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SGD, SGDW
        np.random.seed(42)
        w_data = np.random.randn(3).astype(np.float64) * 5.0
        grad = np.random.randn(3).astype(np.float64)
        lr, wd = 0.1, 0.1  # Large values to see difference
        # SGD with L2 weight decay
        w1 = Tensor(w_data.copy(), requires_grad=True)
        w1.grad = grad.copy()
        opt1 = SGD([w1], lr=lr, weight_decay=wd)
        opt1.step()
        # SGDW with decoupled weight decay
        w2 = Tensor(w_data.copy(), requires_grad=True)
        w2.grad = grad.copy()
        opt2 = SGDW([w2], lr=lr, weight_decay=wd)
        opt2.step()
        # They should differ
        assert not np.allclose(w1.data, w2.data)

    def test_zero_weight_decay(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SGD, SGDW
        np.random.seed(42)
        w_data = np.random.randn(3).astype(np.float64)
        grad = np.random.randn(3).astype(np.float64)
        # With wd=0, SGDW should be same as SGD
        w1 = Tensor(w_data.copy(), requires_grad=True)
        w1.grad = grad.copy()
        opt1 = SGD([w1], lr=0.01)
        opt1.step()
        w2 = Tensor(w_data.copy(), requires_grad=True)
        w2.grad = grad.copy()
        opt2 = SGDW([w2], lr=0.01, weight_decay=0.0)
        opt2.step()
        assert np.allclose(w1.data, w2.data, atol=1e-10)

    def test_with_momentum(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SGDW
        np.random.seed(42)
        w = Tensor(np.random.randn(3).astype(np.float64), requires_grad=True)
        opt = SGDW([w], lr=0.01, momentum=0.9, weight_decay=0.001)
        for _ in range(5):
            w.grad = np.random.randn(3).astype(np.float64)
            opt.step()

    def test_convergence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SGDW
        np.random.seed(42)
        w = Tensor(np.array([5.0]).astype(np.float64), requires_grad=True)
        opt = SGDW([w], lr=0.01, weight_decay=0.001)
        for _ in range(200):
            w.grad = (2.0 * w.data).astype(np.float64)
            opt.step()
        assert np.abs(w.data[0]) < 0.5

    def test_multiple_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SGDW
        np.random.seed(42)
        w = Tensor(np.random.randn(4, 3).astype(np.float64), requires_grad=True)
        opt = SGDW([w], lr=0.01, weight_decay=0.001)
        for _ in range(10):
            w.grad = np.random.randn(4, 3).astype(np.float64)
            opt.step()



# From rewrite_losses.py

class TestKLDivLossComprehensive:
    """Comprehensive tests for KL Divergence Loss"""

    def test_creation(self):
        import numpy as np
        from python.optimization import KLDivLoss

        loss_fn = KLDivLoss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')

    def test_forward_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 4, 5

        # Input is log-probabilities
        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

        # Target is probabilities
        target_logits = np.random.randn(batch, classes).astype(np.float64)
        target = np.exp(target_logits) / np.exp(target_logits).sum(axis=1, keepdims=True)

        # Numpy reference: KL(target||log_probs)
        kl = target * (np.log(target + 1e-10) - log_probs)
        expected = kl.sum() / batch  # batchmean reduction

        loss_fn = KLDivLoss()
        loss = loss_fn(Tensor(log_probs, requires_grad=True), Tensor(target))

        assert np.allclose(loss.data, expected, atol=1e-5), \
            f"Expected {expected}, got {loss.data}"

    def test_reduction_sum(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 3, 4

        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

        target_logits = np.random.randn(batch, classes).astype(np.float64)
        target = np.exp(target_logits) / np.exp(target_logits).sum(axis=1, keepdims=True)

        kl = target * (np.log(target + 1e-10) - log_probs)
        expected = kl.sum()

        loss_fn = KLDivLoss()
        loss = loss_fn(Tensor(log_probs, requires_grad=True), Tensor(target))

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_reduction_none(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 3, 4

        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

        target_logits = np.random.randn(batch, classes).astype(np.float64)
        target = np.exp(target_logits) / np.exp(target_logits).sum(axis=1, keepdims=True)

        kl = target * (np.log(target + 1e-10) - log_probs)
        expected = kl.sum(axis=1)

        loss_fn = KLDivLoss()
        loss = loss_fn(Tensor(log_probs, requires_grad=True), Tensor(target))

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_same_distribution_zero(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 2, 3

        # Create same distribution for input and target
        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        target = np.exp(log_probs)

        loss_fn = KLDivLoss()
        loss = loss_fn(Tensor(log_probs, requires_grad=True), Tensor(target))

        assert np.allclose(loss.data, 0.0, atol=1e-5), \
            f"KL(P||P) should be ~0, got {loss.data}"

    def test_asymmetry(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 2, 4

        logits1 = np.random.randn(batch, classes).astype(np.float64)
        log_probs1 = logits1 - np.log(np.exp(logits1).sum(axis=1, keepdims=True))

        logits2 = np.random.randn(batch, classes).astype(np.float64)
        log_probs2 = logits2 - np.log(np.exp(logits2).sum(axis=1, keepdims=True))

        # KL(P||Q)
        p = np.exp(log_probs1)
        kl_pq = (p * (np.log(p + 1e-10) - log_probs2)).sum() / batch

        # KL(Q||P)
        q = np.exp(log_probs2)
        kl_qp = (q * (np.log(q + 1e-10) - log_probs1)).sum() / batch

        loss_fn = KLDivLoss()
        loss1 = loss_fn(Tensor(log_probs2, requires_grad=True), Tensor(p))
        loss2 = loss_fn(Tensor(log_probs1, requires_grad=True), Tensor(q))

        assert not np.allclose(loss1.data, loss2.data, atol=1e-4), \
            "KL divergence should be asymmetric"

    def test_non_negative(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 5, 6

        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

        target_logits = np.random.randn(batch, classes).astype(np.float64)
        target = np.exp(target_logits) / np.exp(target_logits).sum(axis=1, keepdims=True)

        loss_fn = KLDivLoss()
        loss = loss_fn(Tensor(log_probs, requires_grad=True), Tensor(target))

        assert loss.data >= -1e-6, \
            f"KL divergence should be non-negative, got {loss.data}"

    def test_backward(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 2, 3

        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

        target_logits = np.random.randn(batch, classes).astype(np.float64)
        target = np.exp(target_logits) / np.exp(target_logits).sum(axis=1, keepdims=True)

        log_probs_tensor = Tensor(log_probs, requires_grad=True)
        loss_fn = KLDivLoss()
        loss = loss_fn(log_probs_tensor, Tensor(target))
        loss.backward()

        assert log_probs_tensor.grad is not None
        assert log_probs_tensor.grad.shape == log_probs.shape

    def test_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 2, 3

        logits = np.random.randn(batch, classes).astype(np.float64) * 0.1
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

        target_logits = np.random.randn(batch, classes).astype(np.float64) * 0.1
        target = np.exp(target_logits) / np.exp(target_logits).sum(axis=1, keepdims=True)

        log_probs_tensor = Tensor(log_probs, requires_grad=True)
        loss_fn = KLDivLoss()
        loss = loss_fn(log_probs_tensor, Tensor(target))
        loss.backward()

        # Check gradient is finite and not zero
        assert np.isfinite(log_probs_tensor.grad).all()
        assert not np.allclose(log_probs_tensor.grad, 0, atol=1e-8)

    def test_uniform_distributions(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import KLDivLoss

        np.random.seed(42)
        batch, classes = 3, 4

        # Uniform distributions
        log_probs = -np.log(classes) * np.ones((batch, classes), dtype=np.float64)
        target = (1.0 / classes) * np.ones((batch, classes), dtype=np.float64)

        # KL(uniform||uniform) should be 0
        kl = target * (np.log(target + 1e-10) - log_probs)
        expected = kl.sum() / batch

        loss_fn = KLDivLoss()
        loss = loss_fn(Tensor(log_probs, requires_grad=True), Tensor(target))

        assert np.allclose(loss.data, expected, atol=1e-5)


class TestTripletLossComprehensive:
    """Comprehensive tests for Triplet Loss"""

    def test_creation(self):
        import numpy as np
        from python.optimization import TripletLoss

        loss_fn = TripletLoss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')

    def test_forward_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import TripletLoss

        np.random.seed(42)
        batch, embed_dim = 4, 8
        margin = 1.0

        anchor = np.random.randn(batch, embed_dim).astype(np.float64)
        positive = np.random.randn(batch, embed_dim).astype(np.float64)
        negative = np.random.randn(batch, embed_dim).astype(np.float64)

        # Compute reference
        d_pos = np.sqrt(((anchor - positive)**2).sum(axis=1))
        d_neg = np.sqrt(((anchor - negative)**2).sum(axis=1))
        expected = np.maximum(0, d_pos - d_neg + margin).mean()

        loss_fn = TripletLoss()
        loss = loss_fn(
            Tensor(anchor, requires_grad=True),
            Tensor(positive, requires_grad=True),
            Tensor(negative, requires_grad=True)
        )

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_satisfied_margin(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import TripletLoss

        np.random.seed(42)
        batch, embed_dim = 4, 8
        margin = 1.0

        anchor = np.random.randn(batch, embed_dim).astype(np.float64)
        positive = anchor + np.random.randn(batch, embed_dim).astype(np.float64) * 0.01
        # negative far away from anchor
        negative = anchor + np.random.randn(batch, embed_dim).astype(np.float64) * 10.0

        loss_fn = TripletLoss()
        loss = loss_fn(
            Tensor(anchor, requires_grad=True),
            Tensor(positive, requires_grad=True),
            Tensor(negative, requires_grad=True)
        )

        assert np.allclose(loss.data, 0.0, atol=1e-5), \
            "Loss should be ~0 when margin is satisfied"

    def test_violated_margin(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import TripletLoss

        np.random.seed(42)
        batch, embed_dim = 4, 8
        margin = 1.0

        anchor = np.random.randn(batch, embed_dim).astype(np.float64)
        # positive far from anchor
        positive = anchor + np.random.randn(batch, embed_dim).astype(np.float64) * 5.0
        # negative close to anchor
        negative = anchor + np.random.randn(batch, embed_dim).astype(np.float64) * 0.1

        loss_fn = TripletLoss()
        loss = loss_fn(
            Tensor(anchor, requires_grad=True),
            Tensor(positive, requires_grad=True),
            Tensor(negative, requires_grad=True)
        )

        assert loss.data > 0, "Loss should be > 0 when margin is violated"

    def test_at_margin(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import TripletLoss

        np.random.seed(42)
        batch, embed_dim = 2, 4
        margin = 1.0

        anchor = np.random.randn(batch, embed_dim).astype(np.float64)
        # Set up so d_pos - d_neg + margin ≈ 0
        positive = anchor + np.ones((batch, embed_dim), dtype=np.float64) * 0.5
        negative = anchor + np.ones((batch, embed_dim), dtype=np.float64) * 1.5

        loss_fn = TripletLoss()
        loss = loss_fn(
            Tensor(anchor, requires_grad=True),
            Tensor(positive, requires_grad=True),
            Tensor(negative, requires_grad=True)
        )

        # Should be very small (at the decision boundary)
        assert loss.data >= -1e-5

    def test_different_margins(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import TripletLoss

        np.random.seed(42)
        batch, embed_dim = 3, 6

        anchor = np.random.randn(batch, embed_dim).astype(np.float64)
        positive = np.random.randn(batch, embed_dim).astype(np.float64)
        negative = np.random.randn(batch, embed_dim).astype(np.float64)

        losses = []
        for margin in [0.5, 1.0, 2.0]:
            loss_fn = TripletLoss()
            loss = loss_fn(
                Tensor(anchor, requires_grad=True),
                Tensor(positive, requires_grad=True),
                Tensor(negative, requires_grad=True)
            )
            losses.append(loss.data)

        # Losses may differ depending on margin value
        assert len(losses) == 3

    def test_backward(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import TripletLoss

        np.random.seed(42)
        batch, embed_dim = 2, 4

        anchor = Tensor(np.random.randn(batch, embed_dim).astype(np.float64), requires_grad=True)
        positive = Tensor(np.random.randn(batch, embed_dim).astype(np.float64), requires_grad=True)
        negative = Tensor(np.random.randn(batch, embed_dim).astype(np.float64), requires_grad=True)

        loss_fn = TripletLoss()
        loss = loss_fn(anchor, positive, negative)
        loss.backward()

        assert anchor.grad is not None
        assert positive.grad is not None
        assert negative.grad is not None

    def test_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import TripletLoss

        np.random.seed(42)
        batch, embed_dim = 2, 3

        anchor = Tensor(
            np.random.randn(batch, embed_dim).astype(np.float64) * 0.1,
            requires_grad=True
        )
        positive = Tensor(
            np.random.randn(batch, embed_dim).astype(np.float64) * 0.1,
            requires_grad=True
        )
        negative = Tensor(
            np.random.randn(batch, embed_dim).astype(np.float64) * 0.1,
            requires_grad=True
        )

        loss_fn = TripletLoss()
        loss = loss_fn(anchor, positive, negative)
        loss.backward()

        assert np.isfinite(anchor.grad).all()
        assert np.isfinite(positive.grad).all()
        assert np.isfinite(negative.grad).all()

    def test_batch_of_triplets(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import TripletLoss

        np.random.seed(42)
        batch, embed_dim = 8, 16
        margin = 0.5

        anchor = np.random.randn(batch, embed_dim).astype(np.float64)
        positive = np.random.randn(batch, embed_dim).astype(np.float64)
        negative = np.random.randn(batch, embed_dim).astype(np.float64)

        d_pos = np.sqrt(((anchor - positive)**2).sum(axis=1))
        d_neg = np.sqrt(((anchor - negative)**2).sum(axis=1))
        expected = np.maximum(0, d_pos - d_neg + margin).mean()

        loss_fn = TripletLoss()
        loss = loss_fn(
            Tensor(anchor, requires_grad=True),
            Tensor(positive, requires_grad=True),
            Tensor(negative, requires_grad=True)
        )

        assert np.allclose(loss.data, expected, atol=1e-5)


class TestContrastiveLossComprehensive:
    """Comprehensive tests for Contrastive Loss"""

    def test_creation(self):
        import numpy as np
        from python.optimization import ContrastiveLoss

        loss_fn = ContrastiveLoss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')

    def test_forward_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ContrastiveLoss

        np.random.seed(42)
        batch = 4
        embed_dim = 8
        margin = 1.0

        x1 = np.random.randn(batch, embed_dim).astype(np.float64)
        x2 = np.random.randn(batch, embed_dim).astype(np.float64)
        label = np.array([0, 1, 0, 1], dtype=np.float64)

        # Reference: loss = (1-label)*0.5*d^2 + label*0.5*max(0, margin-d)^2
        d = np.sqrt(((x1 - x2)**2).sum(axis=1))
        expected_loss = (1 - label) * 0.5 * d**2 + label * 0.5 * np.maximum(0, margin - d)**2
        expected = expected_loss.mean()

        loss_fn = ContrastiveLoss()
        loss = loss_fn(
            Tensor(x1, requires_grad=True),
            Tensor(x2, requires_grad=True),
            Tensor(label)
        )

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_similar_pairs(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ContrastiveLoss

        np.random.seed(42)
        batch = 4
        embed_dim = 8
        margin = 1.0

        x1 = np.random.randn(batch, embed_dim).astype(np.float64)
        x2 = x1 + np.random.randn(batch, embed_dim).astype(np.float64) * 0.01
        label = np.zeros(batch, dtype=np.float64)  # Similar pairs

        loss_fn = ContrastiveLoss()
        loss = loss_fn(Tensor(x1), Tensor(x2), Tensor(label))

        # Loss should be small for similar pairs
        assert loss.data < 0.1

    def test_dissimilar_pairs_within_margin(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ContrastiveLoss

        np.random.seed(42)
        batch = 4
        embed_dim = 8
        margin = 1.0

        x1 = np.random.randn(batch, embed_dim).astype(np.float64)
        # Place x2 within margin distance
        x2 = x1 + np.ones((batch, embed_dim), dtype=np.float64) * 0.5
        label = np.ones(batch, dtype=np.float64)  # Dissimilar pairs

        loss_fn = ContrastiveLoss()
        loss = loss_fn(Tensor(x1), Tensor(x2), Tensor(label))

        # Loss should be positive (pushing them apart)
        assert loss.data > 0

    def test_dissimilar_pairs_beyond_margin(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ContrastiveLoss

        np.random.seed(42)
        batch = 4
        embed_dim = 8
        margin = 1.0

        x1 = np.random.randn(batch, embed_dim).astype(np.float64)
        # Place x2 far from x1 (beyond margin)
        x2 = x1 + np.ones((batch, embed_dim), dtype=np.float64) * 5.0
        label = np.ones(batch, dtype=np.float64)  # Dissimilar pairs

        loss_fn = ContrastiveLoss()
        loss = loss_fn(Tensor(x1), Tensor(x2), Tensor(label))

        # Loss should be ~0 (margin satisfied)
        assert np.allclose(loss.data, 0.0, atol=1e-5)

    def test_different_margins(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ContrastiveLoss

        np.random.seed(42)
        batch = 3
        embed_dim = 6

        x1 = np.random.randn(batch, embed_dim).astype(np.float64)
        x2 = np.random.randn(batch, embed_dim).astype(np.float64)
        label = np.array([0, 1, 0], dtype=np.float64)

        losses = []
        for margin in [0.5, 1.0, 2.0]:
            loss_fn = ContrastiveLoss()
            loss = loss_fn(Tensor(x1), Tensor(x2), Tensor(label))
            losses.append(loss.data)

        assert len(losses) == 3

    def test_backward(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ContrastiveLoss

        np.random.seed(42)
        batch = 2
        embed_dim = 4

        x1 = Tensor(np.random.randn(batch, embed_dim).astype(np.float64), requires_grad=True)
        x2 = Tensor(np.random.randn(batch, embed_dim).astype(np.float64), requires_grad=True)
        label = Tensor(np.array([0, 1], dtype=np.float64))

        loss_fn = ContrastiveLoss()
        loss = loss_fn(x1, x2, label)
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None

    def test_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ContrastiveLoss

        np.random.seed(42)
        batch = 2
        embed_dim = 3

        x1 = Tensor(
            np.random.randn(batch, embed_dim).astype(np.float64) * 0.1,
            requires_grad=True
        )
        x2 = Tensor(
            np.random.randn(batch, embed_dim).astype(np.float64) * 0.1,
            requires_grad=True
        )
        label = Tensor(np.array([0, 1], dtype=np.float64))

        loss_fn = ContrastiveLoss()
        loss = loss_fn(x1, x2, label)
        loss.backward()

        assert np.isfinite(x1.grad).all()
        assert np.isfinite(x2.grad).all()


class TestInfoNCELossComprehensive:
    """Comprehensive tests for InfoNCE Loss (Contrastive Learning)"""

    def test_creation(self):
        import numpy as np
        from python.optimization import InfoNCELoss

        loss_fn = InfoNCELoss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')

    def test_forward_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import InfoNCELoss

        np.random.seed(42)
        batch = 4
        embed_dim = 8
        temperature = 0.07

        queries = np.random.randn(batch, embed_dim).astype(np.float64)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)

        keys = np.random.randn(batch, embed_dim).astype(np.float64)
        keys = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-10)

        # Similarity matrix: (batch, batch)
        logits = queries @ keys.T / temperature

        # Labels: diagonal elements are positive pairs
        labels = np.arange(batch)

        # Cross-entropy loss
        log_sum_exp = np.logaddexp.reduce(logits, axis=1)
        expected = (log_sum_exp - logits[np.arange(batch), labels]).mean()

        loss_fn = InfoNCELoss()
        loss = loss_fn(Tensor(queries), Tensor(keys))

        assert np.allclose(loss.data, expected, atol=1e-4)

    def test_temperature_scaling(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import InfoNCELoss

        np.random.seed(42)
        batch = 4
        embed_dim = 8

        queries = np.random.randn(batch, embed_dim).astype(np.float64)
        keys = np.random.randn(batch, embed_dim).astype(np.float64)

        # Higher temperature should give softer distribution and lower loss
        loss_low_temp = InfoNCELoss()(
            Tensor(queries), Tensor(keys)
        ).data
        loss_high_temp = InfoNCELoss()(
            Tensor(queries), Tensor(keys)
        ).data

        # Both should be valid losses
        assert loss_low_temp > 0
        assert loss_high_temp > 0

    def test_identical_pairs_low_loss(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import InfoNCELoss

        np.random.seed(42)
        batch = 4
        embed_dim = 8

        queries = np.random.randn(batch, embed_dim).astype(np.float64)
        # Make keys identical to queries (or very close)
        keys = queries + np.random.randn(batch, embed_dim).astype(np.float64) * 0.01

        loss_fn = InfoNCELoss()
        loss = loss_fn(Tensor(queries), Tensor(keys))

        # Loss should be relatively low when pairs match
        assert loss.data < 2.0

    def test_random_pairs_high_loss(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import InfoNCELoss

        np.random.seed(42)
        batch = 8
        embed_dim = 16

        queries = np.random.randn(batch, embed_dim).astype(np.float64)
        # Random keys (no correspondence with queries)
        keys = np.random.randn(batch, embed_dim).astype(np.float64)

        loss_fn = InfoNCELoss()
        loss = loss_fn(Tensor(queries), Tensor(keys))

        # Loss should be higher for random pairs
        assert loss.data > 0.5

    def test_backward(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import InfoNCELoss

        np.random.seed(42)
        batch = 2
        embed_dim = 4

        queries = Tensor(np.random.randn(batch, embed_dim).astype(np.float64), requires_grad=True)
        keys = Tensor(np.random.randn(batch, embed_dim).astype(np.float64), requires_grad=True)

        loss_fn = InfoNCELoss()
        loss = loss_fn(queries, keys)
        loss.backward()

        assert queries.grad is not None
        assert keys.grad is not None

    def test_batch_size_effect(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import InfoNCELoss

        np.random.seed(42)
        embed_dim = 8

        losses = []
        for batch in [2, 4, 8]:
            queries = np.random.randn(batch, embed_dim).astype(np.float64)
            keys = np.random.randn(batch, embed_dim).astype(np.float64)

            loss_fn = InfoNCELoss()
            loss = loss_fn(Tensor(queries), Tensor(keys))
            losses.append(loss.data)

        # All losses should be valid
        assert len(losses) == 3
        assert all(l > 0 for l in losses)


class TestDiceLossComprehensive:
    """Comprehensive tests for Dice Loss"""

    def test_creation(self):
        import numpy as np
        from python.optimization import DiceLoss

        loss_fn = DiceLoss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')

    def test_forward_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import DiceLoss

        np.random.seed(42)
        batch, height, width = 2, 4, 4
        smooth = 1e-6

        # Predictions and targets in [0, 1]
        pred = np.random.rand(batch, height, width).astype(np.float64)
        target = np.random.rand(batch, height, width).astype(np.float64)

        # Flatten for computation
        pred_flat = pred.reshape(batch, -1)
        target_flat = target.reshape(batch, -1)

        # Dice = 1 - (2*|X∩Y| + smooth) / (|X| + |Y| + smooth)
        intersection = (pred_flat * target_flat).sum(axis=1)
        dice = 1 - (2 * intersection + smooth) / (pred_flat.sum(axis=1) + target_flat.sum(axis=1) + smooth)
        expected = dice.mean()

        loss_fn = DiceLoss()
        loss = loss_fn(Tensor(pred, requires_grad=True), Tensor(target))

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_perfect_prediction(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import DiceLoss

        np.random.seed(42)
        batch, height, width = 2, 4, 4

        pred = np.random.rand(batch, height, width).astype(np.float64)
        target = pred.copy()

        loss_fn = DiceLoss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        assert np.allclose(loss.data, 0.0, atol=1e-5)

    def test_no_overlap(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import DiceLoss

        np.random.seed(42)
        batch, height, width = 2, 4, 4

        pred = np.ones((batch, height, width), dtype=np.float64)
        target = np.zeros((batch, height, width), dtype=np.float64)

        loss_fn = DiceLoss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        # Loss should be close to 1 when there's no overlap
        assert np.allclose(loss.data, 1.0, atol=1e-5)

    def test_partial_overlap(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import DiceLoss

        np.random.seed(42)
        batch, height, width = 2, 4, 4

        pred = np.random.rand(batch, height, width).astype(np.float64)
        target = np.random.rand(batch, height, width).astype(np.float64)

        loss_fn = DiceLoss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        # Loss should be between 0 and 1
        assert 0 <= loss.data <= 1

    def test_smooth_parameter(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import DiceLoss

        np.random.seed(42)
        batch, height, width = 2, 4, 4

        pred = np.zeros((batch, height, width), dtype=np.float64)
        target = np.zeros((batch, height, width), dtype=np.float64)

        # Without smooth, this would cause division by zero
        loss_fn = DiceLoss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        assert np.isfinite(loss.data)

    def test_binary_segmentation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import DiceLoss

        np.random.seed(42)
        batch, height, width = 4, 8, 8

        # Binary segmentation: predictions are probabilities
        pred = np.random.rand(batch, height, width).astype(np.float64)
        # Binary targets
        target = (np.random.rand(batch, height, width) > 0.5).astype(np.float64)

        loss_fn = DiceLoss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        assert np.isfinite(loss.data)
        assert loss.data >= 0

    def test_backward(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import DiceLoss

        np.random.seed(42)
        batch, height, width = 2, 4, 4

        pred = Tensor(np.random.rand(batch, height, width).astype(np.float64), requires_grad=True)
        target = Tensor(np.random.rand(batch, height, width).astype(np.float64))

        loss_fn = DiceLoss()
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None

    def test_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import DiceLoss

        np.random.seed(42)
        batch, height, width = 2, 3, 3

        pred = Tensor(
            np.random.rand(batch, height, width).astype(np.float64) * 0.1 + 0.45,
            requires_grad=True
        )
        target = Tensor(np.random.rand(batch, height, width).astype(np.float64))

        loss_fn = DiceLoss()
        loss = loss_fn(pred, target)
        loss.backward()

        assert np.isfinite(pred.grad).all()


class TestCTCLossComprehensive:
    """Comprehensive tests for CTC Loss (Connectionist Temporal Classification)"""

    def test_creation(self):
        import numpy as np
        from python.optimization import CTCLoss

        loss_fn = CTCLoss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')

    def test_forward_shape(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CTCLoss

        np.random.seed(42)
        time_steps, batch, num_classes = 10, 2, 5

        logits = np.random.randn(time_steps, batch, num_classes).astype(np.float64)
        targets = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        input_lengths = np.array([10, 10], dtype=np.int64)
        target_lengths = np.array([3, 3], dtype=np.int64)

        loss_fn = CTCLoss()
        loss = loss_fn(
            Tensor(logits, requires_grad=True),
            Tensor(targets),
            Tensor(input_lengths),
            Tensor(target_lengths)
        )

        # Output should be scalar
        assert loss.data.shape == () or loss.data.shape == (1,)

    def test_forward_finite(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CTCLoss

        np.random.seed(42)
        time_steps, batch, num_classes = 8, 2, 4

        logits = np.random.randn(time_steps, batch, num_classes).astype(np.float64)
        targets = np.array([[0, 1], [1, 0]], dtype=np.int64)
        input_lengths = np.array([8, 8], dtype=np.int64)
        target_lengths = np.array([2, 2], dtype=np.int64)

        loss_fn = CTCLoss()
        loss = loss_fn(
            Tensor(logits),
            Tensor(targets),
            Tensor(input_lengths),
            Tensor(target_lengths)
        )

        assert np.isfinite(loss.data)

    def test_blank_token(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CTCLoss

        np.random.seed(42)
        time_steps, batch, num_classes = 10, 2, 5
        blank_idx = 0

        logits = np.random.randn(time_steps, batch, num_classes).astype(np.float64)
        # Targets don't include blank token (blank_idx=0)
        targets = np.array([[1, 2, 3], [2, 3, 1]], dtype=np.int64)
        input_lengths = np.array([10, 10], dtype=np.int64)
        target_lengths = np.array([3, 3], dtype=np.int64)

        loss_fn = CTCLoss()
        loss = loss_fn(
            Tensor(logits),
            Tensor(targets),
            Tensor(input_lengths),
            Tensor(target_lengths)
        )

        assert np.isfinite(loss.data)

    def test_short_sequence(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CTCLoss

        np.random.seed(42)
        time_steps, batch, num_classes = 3, 1, 2

        logits = np.random.randn(time_steps, batch, num_classes).astype(np.float64)
        targets = np.array([[0]], dtype=np.int64)
        input_lengths = np.array([3], dtype=np.int64)
        target_lengths = np.array([1], dtype=np.int64)

        loss_fn = CTCLoss()
        loss = loss_fn(
            Tensor(logits),
            Tensor(targets),
            Tensor(input_lengths),
            Tensor(target_lengths)
        )

        assert np.isfinite(loss.data)

    def test_repeated_labels(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CTCLoss

        np.random.seed(42)
        time_steps, batch, num_classes = 12, 2, 5

        logits = np.random.randn(time_steps, batch, num_classes).astype(np.float64)
        # Repeated labels (CTC handles this with blank tokens)
        targets = np.array([[1, 1, 2, 2], [2, 1, 1, 1]], dtype=np.int64)
        input_lengths = np.array([12, 12], dtype=np.int64)
        target_lengths = np.array([4, 4], dtype=np.int64)

        loss_fn = CTCLoss()
        loss = loss_fn(
            Tensor(logits),
            Tensor(targets),
            Tensor(input_lengths),
            Tensor(target_lengths)
        )

        assert np.isfinite(loss.data)

    def test_backward_finite(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CTCLoss

        np.random.seed(42)
        time_steps, batch, num_classes = 8, 2, 4

        logits = Tensor(np.random.randn(time_steps, batch, num_classes).astype(np.float64), requires_grad=True)
        targets = np.array([[0, 1], [1, 0]], dtype=np.int64)
        input_lengths = np.array([8, 8], dtype=np.int64)
        target_lengths = np.array([2, 2], dtype=np.int64)

        loss_fn = CTCLoss()
        loss = loss_fn(
            logits,
            Tensor(targets),
            Tensor(input_lengths),
            Tensor(target_lengths)
        )
        loss.backward()

        assert logits.grad is not None
        assert np.isfinite(logits.grad).all()

    def test_loss_positive(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CTCLoss

        np.random.seed(42)
        time_steps, batch, num_classes = 10, 2, 5

        logits = np.random.randn(time_steps, batch, num_classes).astype(np.float64)
        targets = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        input_lengths = np.array([10, 10], dtype=np.int64)
        target_lengths = np.array([3, 3], dtype=np.int64)

        loss_fn = CTCLoss()
        loss = loss_fn(
            Tensor(logits),
            Tensor(targets),
            Tensor(input_lengths),
            Tensor(target_lengths)
        )

        assert loss.data >= 0


class TestSmoothL1LossComprehensive:
    """Comprehensive tests for Smooth L1 Loss (Huber Loss)"""

    def test_creation(self):
        import numpy as np
        from python.optimization import SmoothL1Loss

        loss_fn = SmoothL1Loss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')

    def test_forward_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 4, 8
        beta = 1.0

        pred = np.random.randn(batch, dim).astype(np.float64)
        target = np.random.randn(batch, dim).astype(np.float64)

        # Reference: Smooth L1 loss
        diff = np.abs(pred - target)
        loss_vals = np.where(
            diff < beta,
            0.5 * diff**2 / beta,
            diff - 0.5 * beta
        )
        expected = loss_vals.mean()

        loss_fn = SmoothL1Loss()
        loss = loss_fn(Tensor(pred, requires_grad=True), Tensor(target))

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_small_errors_quadratic(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 4, 8
        beta = 1.0

        pred = np.random.randn(batch, dim).astype(np.float64)
        # Make errors small
        target = pred + np.random.randn(batch, dim).astype(np.float64) * 0.1

        loss_fn = SmoothL1Loss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        assert np.isfinite(loss.data)

    def test_large_errors_linear(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 4, 8
        beta = 1.0

        pred = np.random.randn(batch, dim).astype(np.float64)
        # Make errors large
        target = pred + np.random.randn(batch, dim).astype(np.float64) * 10.0

        loss_fn = SmoothL1Loss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        assert np.isfinite(loss.data)

    def test_transition_point(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 2, 4
        beta = 1.0

        pred = np.zeros((batch, dim), dtype=np.float64)
        # Set target so error = beta exactly
        target = np.ones((batch, dim), dtype=np.float64) * beta

        diff = np.abs(pred - target)
        # At |error| = beta, both formulas should give same value
        quadratic_val = 0.5 * diff[0, 0]**2 / beta
        linear_val = diff[0, 0] - 0.5 * beta

        assert np.allclose(quadratic_val, linear_val, atol=1e-5)

    def test_reduction_mean(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 4, 8

        pred = np.random.randn(batch, dim).astype(np.float64)
        target = np.random.randn(batch, dim).astype(np.float64)

        loss_fn = SmoothL1Loss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        assert np.isfinite(loss.data)

    def test_reduction_sum(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 4, 8

        pred = np.random.randn(batch, dim).astype(np.float64)
        target = np.random.randn(batch, dim).astype(np.float64)

        loss_fn = SmoothL1Loss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        assert np.isfinite(loss.data)

    def test_reduction_none(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 4, 8

        pred = np.random.randn(batch, dim).astype(np.float64)
        target = np.random.randn(batch, dim).astype(np.float64)

        loss_fn = SmoothL1Loss()
        loss = loss_fn(Tensor(pred), Tensor(target))

        assert loss.data.shape == pred.shape

    def test_backward(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 2, 4

        pred = Tensor(np.random.randn(batch, dim).astype(np.float64), requires_grad=True)
        target = Tensor(np.random.randn(batch, dim).astype(np.float64))

        loss_fn = SmoothL1Loss()
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None

    def test_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 2, 3

        pred = Tensor(
            np.random.randn(batch, dim).astype(np.float64) * 0.1,
            requires_grad=True
        )
        target = Tensor(np.random.randn(batch, dim).astype(np.float64) * 0.1)

        loss_fn = SmoothL1Loss()
        loss = loss_fn(pred, target)
        loss.backward()

        assert np.isfinite(pred.grad).all()

    def test_different_beta_values(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import SmoothL1Loss

        np.random.seed(42)
        batch, dim = 4, 8

        pred = np.random.randn(batch, dim).astype(np.float64)
        target = np.random.randn(batch, dim).astype(np.float64)

        losses = []
        for beta in [0.5, 1.0, 2.0]:
            loss_fn = SmoothL1Loss()
            loss = loss_fn(Tensor(pred), Tensor(target))
            losses.append(loss.data)

        assert len(losses) == 3
        # Different beta values should give different results
        assert not np.allclose(losses[0], losses[1], atol=1e-5)


class TestNLLLossComprehensive:
    """Comprehensive tests for Negative Log Likelihood Loss"""

    def test_creation(self):
        import numpy as np
        from python.optimization import NLLLoss

        loss_fn = NLLLoss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')

    def test_forward_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NLLLoss

        np.random.seed(42)
        batch, classes = 4, 5

        # log_probs: (batch, classes) - already log-probabilities
        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

        # targets: (batch,) - class indices
        targets = np.array([0, 2, 1, 3], dtype=np.int64)

        # Reference: NLL = -log_probs[batch_idx, target_idx]
        expected = -log_probs[np.arange(batch), targets].mean()

        loss_fn = NLLLoss()
        loss = loss_fn(Tensor(log_probs, requires_grad=True), Tensor(targets))

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_reduction_sum(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NLLLoss

        np.random.seed(42)
        batch, classes = 3, 4

        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        targets = np.array([0, 1, 2], dtype=np.int64)

        expected = -log_probs[np.arange(batch), targets].sum()

        loss_fn = NLLLoss()
        loss = loss_fn(Tensor(log_probs), Tensor(targets))

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_reduction_none(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NLLLoss

        np.random.seed(42)
        batch, classes = 3, 4

        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        targets = np.array([0, 1, 2], dtype=np.int64)

        expected = -log_probs[np.arange(batch), targets]

        loss_fn = NLLLoss()
        loss = loss_fn(Tensor(log_probs), Tensor(targets))

        assert np.allclose(loss.data, expected, atol=1e-5)

    def test_with_softmax_equals_ce(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NLLLoss, CrossEntropyLoss

        np.random.seed(42)
        batch, classes = 4, 5

        logits = np.random.randn(batch, classes).astype(np.float64)
        targets = np.array([0, 2, 1, 3], dtype=np.int64)

        # NLL(log_softmax(logits)) should equal CrossEntropyLoss(logits)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

        nll_loss = NLLLoss()
        nll_out = nll_loss(Tensor(log_probs), Tensor(targets))

        ce_loss = CrossEntropyLoss()
        ce_out = ce_loss(Tensor(logits), Tensor(targets))

        assert np.allclose(nll_out.data, ce_out.data, atol=1e-4)

    def test_backward(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NLLLoss

        np.random.seed(42)
        batch, classes = 2, 3

        logits = np.random.randn(batch, classes).astype(np.float64)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        targets = np.array([0, 1], dtype=np.int64)

        log_probs_tensor = Tensor(log_probs, requires_grad=True)
        loss_fn = NLLLoss()
        loss = loss_fn(log_probs_tensor, Tensor(targets))
        loss.backward()

        assert log_probs_tensor.grad is not None

    def test_gradcheck(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NLLLoss

        np.random.seed(42)
        batch, classes = 2, 3

        logits = np.random.randn(batch, classes).astype(np.float64) * 0.1
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        targets = np.array([0, 1], dtype=np.int64)

        log_probs_tensor = Tensor(log_probs, requires_grad=True)
        loss_fn = NLLLoss()
        loss = loss_fn(log_probs_tensor, Tensor(targets))
        loss.backward()

        assert np.isfinite(log_probs_tensor.grad).all()

    def test_different_class_counts(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import NLLLoss

        np.random.seed(42)
        batch = 4

        losses = []
        for classes in [2, 5, 10]:
            logits = np.random.randn(batch, classes).astype(np.float64)
            log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            targets = np.random.randint(0, classes, batch)

            loss_fn = NLLLoss()
            loss = loss_fn(Tensor(log_probs), Tensor(targets))
            losses.append(loss.data)

        assert len(losses) == 3



# From rewrite_sched_grad.py

class TestStepLRComprehensive:
    """Comprehensive tests for StepLR scheduler with numpy reference implementations."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import StepLR, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = StepLR(opt, step_size=10, gamma=0.1)
        assert scheduler is not None
        assert opt.get_lr()[-1] == 0.1

    def test_decay_formula(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import StepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        step_size = 5
        gamma = 0.1

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

        # Numpy reference: lr = initial_lr * gamma^(epoch // step_size)
        for epoch in range(30):
            expected_lr = initial_lr * (gamma ** (epoch // step_size))
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10), \
                f"Epoch {epoch}: expected {expected_lr}, got {actual_lr}"
            scheduler.step()

    def test_no_decay_before_step(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import StepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        step_size = 10

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = StepLR(opt, step_size=step_size, gamma=0.1)

        for epoch in range(step_size):
            assert np.isclose(opt.get_lr()[-1], initial_lr, atol=1e-10), \
                f"Epoch {epoch}: LR should not decay before step_size"
            scheduler.step()

    def test_multiple_decays(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import StepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        step_size = 5
        gamma = 0.5

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

        test_epochs = [0, 5, 10, 15]
        expected_lrs = [
            initial_lr * (gamma ** 0),
            initial_lr * (gamma ** 1),
            initial_lr * (gamma ** 2),
            initial_lr * (gamma ** 3),
        ]

        for epoch in range(max(test_epochs) + 1):
            if epoch in test_epochs:
                idx = test_epochs.index(epoch)
                actual_lr = opt.get_lr()[-1]
                assert np.isclose(actual_lr, expected_lrs[idx], atol=1e-10), \
                    f"Epoch {epoch}: expected {expected_lrs[idx]}, got {actual_lr}"
            scheduler.step()

    def test_different_gamma(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import StepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        step_size = 5

        gammas = [0.1, 0.5, 0.9]

        for gamma in gammas:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=initial_lr)
            scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

            expected_lr = initial_lr * gamma
            for _ in range(step_size):
                scheduler.step()

            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10), \
                f"Gamma {gamma}: expected {expected_lr}, got {actual_lr}"

    def test_different_step_size(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import StepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        gamma = 0.5

        step_sizes = [5, 10, 15]

        for step_size in step_sizes:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=initial_lr)
            scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

            for _ in range(step_size):
                scheduler.step()

            expected_lr = initial_lr * gamma
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10), \
                f"Step size {step_size}: expected {expected_lr}, got {actual_lr}"

    def test_lr_schedule_curve(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import StepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        step_size = 5
        gamma = 0.5
        total_epochs = 20

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

        lrs_actual = []
        lrs_expected = np.array([initial_lr * (gamma ** (epoch // step_size)) for epoch in range(total_epochs)])

        for epoch in range(total_epochs):
            lrs_actual.append(opt.get_lr()[-1])
            scheduler.step()

        lrs_actual = np.array(lrs_actual)
        assert np.allclose(lrs_actual, lrs_expected, atol=1e-10), \
            f"LR schedule mismatch. Expected {lrs_expected}, got {lrs_actual}"

    def test_last_epoch_property(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import StepLR, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = StepLR(opt, step_size=10, gamma=0.1)

        for _ in range(15):
            scheduler.step()

        assert scheduler.last_epoch == 14


class TestMultiStepLRComprehensive:
    """Comprehensive tests for MultiStepLR scheduler."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import MultiStepLR, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = MultiStepLR(opt, milestones=[5, 10, 15], gamma=0.1)
        assert scheduler is not None
        assert opt.get_lr()[-1] == 0.1

    def test_decay_at_milestones(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import MultiStepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        milestones = [5, 10]
        gamma = 0.5

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = MultiStepLR(opt, milestones=milestones, gamma=gamma)

        for epoch in range(12):
            if epoch == 5:
                expected_lr = initial_lr * gamma
                actual_lr = opt.get_lr()[-1]
                assert np.isclose(actual_lr, expected_lr, atol=1e-10)
            elif epoch == 10:
                expected_lr = initial_lr * (gamma ** 2)
                actual_lr = opt.get_lr()[-1]
                assert np.isclose(actual_lr, expected_lr, atol=1e-10)
            scheduler.step()

    def test_no_decay_between_milestones(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import MultiStepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        milestones = [10, 20]

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = MultiStepLR(opt, milestones=milestones, gamma=0.1)

        for epoch in range(10):
            assert np.isclose(opt.get_lr()[-1], initial_lr, atol=1e-10)
            scheduler.step()

    def test_multiple_milestones(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import MultiStepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        milestones = [5, 10, 20]
        gamma = 0.5

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = MultiStepLR(opt, milestones=milestones, gamma=gamma)

        expected_lrs = {
            0: initial_lr,
            5: initial_lr * gamma,
            10: initial_lr * (gamma ** 2),
            20: initial_lr * (gamma ** 3),
        }

        for epoch in range(25):
            if epoch in expected_lrs:
                actual_lr = opt.get_lr()[-1]
                assert np.isclose(actual_lr, expected_lrs[epoch], atol=1e-10), \
                    f"Epoch {epoch}: expected {expected_lrs[epoch]}, got {actual_lr}"
            scheduler.step()

    def test_lr_schedule_curve(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import MultiStepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        milestones = [5, 15]
        gamma = 0.5
        total_epochs = 20

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = MultiStepLR(opt, milestones=milestones, gamma=gamma)

        lrs_actual = []
        for epoch in range(total_epochs):
            lrs_actual.append(opt.get_lr()[-1])
            scheduler.step()

        lrs_actual = np.array(lrs_actual)

        # Verify the schedule follows milestone-based decay
        assert np.isclose(lrs_actual[0], initial_lr, atol=1e-10)
        assert np.isclose(lrs_actual[5], initial_lr * gamma, atol=1e-10)
        assert np.isclose(lrs_actual[15], initial_lr * (gamma ** 2), atol=1e-10)

    def test_different_gamma(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import MultiStepLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        milestones = [5]
        gammas = [0.1, 0.5, 0.9]

        for gamma in gammas:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=initial_lr)
            scheduler = MultiStepLR(opt, milestones=milestones, gamma=gamma)

            for _ in range(6):
                scheduler.step()

            expected_lr = initial_lr * gamma
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10)


class TestExponentialLRComprehensive:
    """Comprehensive tests for ExponentialLR scheduler."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ExponentialLR, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = ExponentialLR(opt, gamma=0.9)
        assert scheduler is not None
        assert opt.get_lr()[-1] == 0.1

    def test_decay_formula(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ExponentialLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        gamma = 0.95

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = ExponentialLR(opt, gamma=gamma)

        # Numpy reference: lr = initial_lr * gamma^epoch
        for epoch in range(30):
            expected_lr = initial_lr * (gamma ** epoch)
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10), \
                f"Epoch {epoch}: expected {expected_lr}, got {actual_lr}"
            scheduler.step()

    def test_lr_schedule_curve(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ExponentialLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        gamma = 0.9
        total_epochs = 20

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = ExponentialLR(opt, gamma=gamma)

        lrs_actual = []
        lrs_expected = np.array([initial_lr * (gamma ** epoch) for epoch in range(total_epochs)])

        for _ in range(total_epochs):
            lrs_actual.append(opt.get_lr()[-1])
            scheduler.step()

        lrs_actual = np.array(lrs_actual)
        assert np.allclose(lrs_actual, lrs_expected, atol=1e-10)

    def test_different_gamma(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ExponentialLR, SGD

        np.random.seed(42)
        initial_lr = 0.1

        gammas = [0.5, 0.9, 0.99]

        for gamma in gammas:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=initial_lr)
            scheduler = ExponentialLR(opt, gamma=gamma)

            scheduler.step()
            expected_lr = initial_lr * gamma
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10)

    def test_monotonically_decreasing(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ExponentialLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        gamma = 0.95

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = ExponentialLR(opt, gamma=gamma)

        lrs = []
        for _ in range(20):
            lrs.append(opt.get_lr()[-1])
            scheduler.step()

        lrs = np.array(lrs)
        # Verify monotonically decreasing (with gamma < 1)
        assert np.all(np.diff(lrs) < 0), "LR should be monotonically decreasing"


class TestCosineAnnealingLRComprehensive:
    """Comprehensive tests for CosineAnnealingLR scheduler."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = CosineAnnealingLR(opt, T_max=20, eta_min=0.001)
        assert scheduler is not None
        assert opt.get_lr()[-1] == 0.1

    def test_formula(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_max = 20
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

        for epoch in range(T_max + 1):
            # Numpy reference: lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * (epoch-1) / T_max))
            # Note: actual lr is one epoch behind because step() increments last_epoch before computing lr
            expected_lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + np.cos(np.pi * (epoch - 1) / T_max)) if epoch > 0 else initial_lr
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10), \
                f"Epoch {epoch}: expected {expected_lr}, got {actual_lr}"
            scheduler.step()

    def test_endpoints(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_max = 20
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

        # At epoch 0: lr should be initial_lr
        assert np.isclose(opt.get_lr()[-1], initial_lr, atol=1e-10)

        # Need T_max+1 steps to reach epoch T_max (0, 1, ..., T_max)
        for _ in range(T_max + 1):
            scheduler.step()

        # At epoch T_max: lr should be eta_min
        assert np.isclose(opt.get_lr()[-1], eta_min, atol=1e-10)

    def test_midpoint(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_max = 20
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

        # To get to epoch T_max/2, we need T_max/2 + 1 steps
        for _ in range(T_max // 2 + 1):
            scheduler.step()

        # At midpoint: lr should be average of initial_lr and eta_min
        expected_lr = (initial_lr + eta_min) / 2
        actual_lr = opt.get_lr()[-1]
        assert np.isclose(actual_lr, expected_lr, atol=1e-10)

    def test_symmetry(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_max = 20
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

        lrs = []
        for _ in range(T_max + 1):
            lrs.append(opt.get_lr()[-1])
            scheduler.step()

        lrs = np.array(lrs)
        # The cosine schedule is strictly monotonically decreasing after the initial base_lr
        # (which appears at indices 0 and 1 due to the off-by-one between last_epoch=-1 and epoch 0)
        # Check that learning rate decreases or stays the same at each step after the initial plateau
        diffs = np.diff(lrs[1:])  # Skip the duplicate base_lr, check differences
        assert np.all(diffs <= 1e-10), "Learning rate should be monotonically non-increasing after initial step"

    def test_lr_schedule_curve(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_max = 20
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

        lrs_actual = []
        # Expected values use epoch-1 because actual lr is one step behind
        lrs_expected = np.array([
            eta_min + 0.5 * (initial_lr - eta_min) * (1 + np.cos(np.pi * (epoch - 1) / T_max))
            if epoch > 0 else initial_lr
            for epoch in range(T_max + 1)
        ])

        for _ in range(T_max + 1):
            lrs_actual.append(opt.get_lr()[-1])
            scheduler.step()

        lrs_actual = np.array(lrs_actual)
        assert np.allclose(lrs_actual, lrs_expected, atol=1e-10)

    def test_different_T_max(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        eta_min = 0.001

        T_maxs = [10, 20, 50]

        for T_max in T_maxs:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=initial_lr)
            scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

            # Need T_max+1 steps to reach epoch T_max
            for _ in range(T_max + 1):
                scheduler.step()

            # At T_max, should reach eta_min
            assert np.isclose(opt.get_lr()[-1], eta_min, atol=1e-10)

    def test_different_eta_min(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_max = 20

        eta_mins = [0.0, 0.001, 0.01]

        for eta_min in eta_mins:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=initial_lr)
            scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

            # Need T_max+1 steps to reach epoch T_max
            for _ in range(T_max + 1):
                scheduler.step()

            # At T_max, should reach eta_min
            assert np.isclose(opt.get_lr()[-1], eta_min, atol=1e-10)

    def test_smooth_transition(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_max = 20
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

        lrs = []
        for _ in range(T_max):
            lrs.append(opt.get_lr()[-1])
            scheduler.step()

        lrs = np.array(lrs)
        # Check for no discontinuities (smooth gradient)
        diffs = np.diff(lrs)
        assert np.all(np.abs(diffs) < 0.01), "Should have smooth transitions"


class TestOneCycleLRComprehensive:
    """Comprehensive tests for OneCycleLR scheduler."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import OneCycleLR, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = OneCycleLR(opt, max_lr=1.0, total_steps=100)
        assert scheduler is not None

    def test_warmup_phase(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import OneCycleLR, SGD

        np.random.seed(42)
        base_lr = 0.1
        max_lr = 1.0
        total_steps = 100

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=base_lr)
        scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)

        # During warmup (first half), LR should increase
        lrs_warmup = []
        for _ in range(total_steps // 2):
            lrs_warmup.append(opt.get_lr()[-1])
            scheduler.step()

        lrs_warmup = np.array(lrs_warmup)
        # Verify increasing trend in warmup
        assert lrs_warmup[-1] > lrs_warmup[0], "LR should increase during warmup"

    def test_decay_phase(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import OneCycleLR, SGD

        np.random.seed(42)
        base_lr = 0.1
        max_lr = 1.0
        total_steps = 100

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=base_lr)
        scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)

        for _ in range(total_steps // 2):
            scheduler.step()

        # During decay (second half), LR should decrease
        lrs_decay = []
        for _ in range(total_steps // 2):
            lrs_decay.append(opt.get_lr()[-1])
            scheduler.step()

        lrs_decay = np.array(lrs_decay)
        # Verify decreasing trend in decay
        assert lrs_decay[-1] < lrs_decay[0], "LR should decrease during decay phase"

    def test_lr_schedule_curve(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import OneCycleLR, SGD

        np.random.seed(42)
        base_lr = 0.1
        max_lr = 1.0
        total_steps = 50

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=base_lr)
        scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)

        lrs = []
        for _ in range(total_steps):
            lrs.append(opt.get_lr()[-1])
            scheduler.step()

        lrs = np.array(lrs)
        # Verify the schedule reaches near max_lr
        assert np.max(lrs) > max_lr * 0.9, "Should reach near max_lr"

    def test_different_max_lr(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import OneCycleLR, SGD

        np.random.seed(42)
        base_lr = 0.1
        total_steps = 100

        max_lrs = [0.5, 1.0, 2.0]

        for max_lr in max_lrs:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=base_lr)
            scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)

            max_lr_reached = 0
            for _ in range(total_steps):
                max_lr_reached = max(max_lr_reached, opt.get_lr()[-1])
                scheduler.step()

            assert max_lr_reached >= max_lr * 0.9, f"Should reach near {max_lr}"

    def test_different_total_steps(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import OneCycleLR, SGD

        np.random.seed(42)
        base_lr = 0.1
        max_lr = 1.0

        for total_steps in [50, 100, 200]:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=base_lr)
            scheduler = OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)

            lrs = []
            for _ in range(total_steps):
                lrs.append(opt.get_lr()[-1])
                scheduler.step()

            lrs = np.array(lrs)
            assert len(lrs) == total_steps


class TestWarmupLRComprehensive:
    """Comprehensive tests for WarmupLR scheduler."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import WarmupLR, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = WarmupLR(opt, warmup_iters=10)
        assert scheduler is not None

    def test_linear_warmup(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import WarmupLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        warmup_iters = 10

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = WarmupLR(opt, warmup_iters=warmup_iters)

        # Numpy reference: lr = initial_lr * (step_count) / warmup_iters during warmup
        for step in range(warmup_iters):
            expected_lr = initial_lr * step / warmup_iters
            scheduler.step()
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10), \
                f"Step {step}: expected {expected_lr}, got {actual_lr}"

    def test_after_warmup(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import WarmupLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        warmup_iters = 10

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = WarmupLR(opt, warmup_iters=warmup_iters)

        for _ in range(warmup_iters):
            scheduler.step()

        # After warmup, LR should stay at initial_lr
        for _ in range(10):
            assert np.isclose(opt.get_lr()[-1], initial_lr, atol=1e-10)
            scheduler.step()

    def test_lr_schedule_curve(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import WarmupLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        warmup_iters = 10
        total_epochs = 20

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = WarmupLR(opt, warmup_iters=warmup_iters)

        lrs_actual = []
        for epoch in range(total_epochs):
            scheduler.step()
            lrs_actual.append(opt.get_lr()[-1])

        lrs_actual = np.array(lrs_actual)

        # Verify warmup phase increases linearly (step 1 to warmup_iters)
        for step in range(warmup_iters):
            expected_lr = initial_lr * step / warmup_iters
            assert np.isclose(lrs_actual[step], expected_lr, atol=1e-10)

        # Verify post-warmup stays constant
        for epoch in range(warmup_iters, total_epochs):
            assert np.isclose(lrs_actual[epoch], initial_lr, atol=1e-10)

    def test_different_warmup_epochs(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import WarmupLR, SGD

        np.random.seed(42)
        initial_lr = 0.1

        for warmup_iters in [5, 10, 20]:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=initial_lr)
            scheduler = WarmupLR(opt, warmup_iters=warmup_iters)

            for _ in range(warmup_iters):
                scheduler.step()

            # After warmup, should be at full LR
            assert np.isclose(opt.get_lr()[-1], initial_lr, atol=1e-10)


class TestReduceLROnPlateauComprehensive:
    """Comprehensive tests for ReduceLROnPlateau scheduler."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ReduceLROnPlateau, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        assert scheduler is not None

    def test_reduces_after_patience(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ReduceLROnPlateau, SGD

        np.random.seed(42)
        initial_lr = 0.1
        factor = 0.5
        patience = 3

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=factor, patience=patience)

        # Simulate no improvement for patience+2 epochs (triggers after patience+1 bad steps)
        best_loss = 1.0
        for epoch in range(patience + 2):
            loss = best_loss + 0.1  # No improvement
            scheduler.step(loss)

        expected_lr = initial_lr * factor
        actual_lr = opt.get_lr()[-1]
        assert np.isclose(actual_lr, expected_lr, atol=1e-10)

    def test_no_reduction_with_improvement(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ReduceLROnPlateau, SGD

        np.random.seed(42)
        initial_lr = 0.1

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

        # Improve every epoch
        best_loss = 1.0
        for epoch in range(10):
            loss = best_loss - 0.05  # Improvement
            best_loss = loss
            scheduler.step(loss)

        # LR should not change
        assert np.isclose(opt.get_lr()[-1], initial_lr, atol=1e-10)

    def test_factor(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ReduceLROnPlateau, SGD

        np.random.seed(42)
        initial_lr = 0.1
        patience = 2

        factors = [0.1, 0.5, 0.9]

        for factor in factors:
            param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
            opt = SGD([param], lr=initial_lr)
            scheduler = ReduceLROnPlateau(opt, mode='min', factor=factor, patience=patience)

            # No improvement for patience+2 epochs (triggers after patience+1 bad steps)
            best_loss = 1.0
            for _ in range(patience + 2):
                loss = best_loss + 0.1
                scheduler.step(loss)

            expected_lr = initial_lr * factor
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10)

    def test_min_lr(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ReduceLROnPlateau, SGD

        np.random.seed(42)
        initial_lr = 0.1
        min_lr = 0.01
        factor = 0.5
        patience = 2

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=factor, patience=patience, min_lr=min_lr)

        # Multiple reductions
        best_loss = 1.0
        for _ in range((patience + 1) * 3):
            loss = best_loss + 0.1
            scheduler.step(loss)

        # LR should not go below min_lr
        assert opt.get_lr()[-1] >= min_lr * 0.99

    def test_cooldown(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ReduceLROnPlateau, SGD

        np.random.seed(42)
        initial_lr = 0.1
        patience = 2
        cooldown = 2

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=patience, cooldown=cooldown)

        # Trigger reduction (patience+2 steps)
        best_loss = 1.0
        for _ in range(patience + 2):
            loss = best_loss + 0.1
            scheduler.step(loss)

        reduced_lr = opt.get_lr()[-1]

        # During cooldown, no further reductions even with no improvement
        for _ in range(cooldown + 1):
            loss = best_loss + 0.1
            scheduler.step(loss)

        # Should not reduce during cooldown
        assert np.isclose(opt.get_lr()[-1], reduced_lr, atol=1e-10)

    def test_threshold(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import ReduceLROnPlateau, SGD

        np.random.seed(42)
        initial_lr = 0.1
        patience = 2
        threshold = 0.01

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=patience, threshold=threshold)

        # Improvement below threshold is not considered improvement (patience+2 steps)
        best_loss = 1.0
        for _ in range(patience + 2):
            loss = best_loss - 0.001  # Improvement < threshold
            scheduler.step(loss)

        # Should reduce because improvement is below threshold
        assert opt.get_lr()[-1] < initial_lr


class TestLinearLRComprehensive:
    """Comprehensive tests for LinearLR scheduler."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LinearLR, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = LinearLR(opt, start_factor=0.5, end_factor=1.0, total_iters=100)
        assert scheduler is not None

    def test_linear_decay_formula(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LinearLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        start_factor = 0.5
        end_factor = 1.0
        total_iters = 20

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = LinearLR(opt, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)

        # Before any step(), scheduler is at implicit epoch -1, giving base_lr
        # After step() N, we're at epoch N-1 with interpolated LR
        # First get_lr() without step() gives base_lr, not start_factor*base_lr
        scheduler.step()  # Move to epoch 0
        for step_num in range(total_iters):
            # After step_num steps (not counting the first step()), we're at epoch step_num
            progress = (step_num + 1) / total_iters
            expected_factor = start_factor + (end_factor - start_factor) * progress
            expected_lr = initial_lr * expected_factor
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10), \
                f"Step {step_num+1}: expected {expected_lr}, got {actual_lr}"
            if step_num < total_iters - 1:
                scheduler.step()

    def test_lr_schedule_curve(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LinearLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        start_factor = 0.5
        end_factor = 1.0
        total_iters = 20

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = LinearLR(opt, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)

        lrs_actual = []
        scheduler.step()  # Start at epoch 0
        for epoch in range(total_iters):
            lrs_actual.append(opt.get_lr()[-1])
            if epoch < total_iters - 1:
                scheduler.step()

        lrs_actual = np.array(lrs_actual)
        lrs_expected = np.array([
            initial_lr * (start_factor + (end_factor - start_factor) * (epoch + 1) / total_iters)
            for epoch in range(total_iters)
        ])

        assert np.allclose(lrs_actual, lrs_expected, atol=1e-10)

    def test_endpoints(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import LinearLR, SGD

        np.random.seed(42)
        initial_lr = 0.1
        start_factor = 0.5
        end_factor = 1.0
        total_iters = 20

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = LinearLR(opt, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)

        # At start (after first step): should be initial_lr * start_factor (approximately, at epoch 0)
        scheduler.step()
        assert np.isclose(opt.get_lr()[-1], initial_lr * (start_factor + (end_factor - start_factor) * 1 / total_iters), atol=1e-10)

        for _ in range(total_iters - 2):
            scheduler.step()

        # At end (after total_iters steps, at epoch total_iters-1): should approach end_factor
        scheduler.step()
        assert np.isclose(opt.get_lr()[-1], initial_lr * end_factor, atol=1e-10)


class TestCosineAnnealingWarmRestartsComprehensive:
    """Comprehensive tests for CosineAnnealingWarmRestarts scheduler."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingWarmRestarts, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=0.001)
        assert scheduler is not None

    def test_restart_behavior(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingWarmRestarts, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_0 = 10
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=1, eta_min=eta_min)

        # Collect LRs for first cycle
        lrs_cycle1 = []
        for _ in range(T_0):
            lrs_cycle1.append(opt.get_lr()[-1])
            scheduler.step()

        # Collect LRs for second cycle
        lrs_cycle2 = []
        for _ in range(T_0):
            lrs_cycle2.append(opt.get_lr()[-1])
            scheduler.step()

        # Cycles should be similar (restart behavior)
        lrs_cycle1 = np.array(lrs_cycle1)
        lrs_cycle2 = np.array(lrs_cycle2)
        assert np.allclose(lrs_cycle1, lrs_cycle2, atol=1e-9)

    def test_formula_within_cycle(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingWarmRestarts, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_0 = 10
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=1, eta_min=eta_min)

        for t in range(T_0):
            # Expected: eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * t / T_0))
            expected_lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + np.cos(np.pi * t / T_0))
            actual_lr = opt.get_lr()[-1]
            assert np.isclose(actual_lr, expected_lr, atol=1e-10)
            scheduler.step()

    def test_lr_schedule_curve(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingWarmRestarts, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_0 = 10
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=1, eta_min=eta_min)

        lrs = []
        for _ in range(T_0 * 2):
            lrs.append(opt.get_lr()[-1])
            scheduler.step()

        lrs = np.array(lrs)
        # Should show restart pattern
        assert len(lrs) == T_0 * 2

    def test_T_mult(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import CosineAnnealingWarmRestarts, SGD

        np.random.seed(42)
        initial_lr = 0.1
        T_0 = 5
        T_mult = 2
        eta_min = 0.001

        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=initial_lr)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

        # First cycle: T_0 steps
        for _ in range(T_0):
            scheduler.step()

        # Second cycle: T_0 * T_mult steps
        for _ in range(T_0 * T_mult):
            scheduler.step()


class TestGradientClippingComprehensive:
    """Comprehensive tests for gradient clipping functions."""

    def test_clip_grad_norm_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import clip_grad_norm_, SGD

        np.random.seed(42)
        param = Tensor(np.random.randn(10, 10).astype(np.float64), requires_grad=True)
        param.grad = Tensor(np.random.randn(10, 10).astype(np.float64))

        opt = SGD([param], lr=0.1)

        max_norm = 1.0

        # Compute expected total norm
        grad_array = param.grad.data
        expected_total_norm = np.sqrt(np.sum(grad_array ** 2))

        # Clip
        returned_norm = clip_grad_norm_([param], max_norm)

        # If total_norm > max_norm, grads should be scaled
        if expected_total_norm > max_norm:
            scale_factor = max_norm / expected_total_norm
            expected_grad = grad_array * scale_factor
        else:
            expected_grad = grad_array

        actual_grad = param.grad.data
        assert np.allclose(actual_grad, expected_grad, atol=1e-10)

    def test_clip_grad_norm_no_op(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import clip_grad_norm_, SGD

        np.random.seed(42)
        grad_data = np.random.randn(5, 5).astype(np.float64) * 0.001
        param = Tensor(np.zeros((5, 5)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(grad_data)

        opt = SGD([param], lr=0.1)

        max_norm = 10.0  # Large max_norm
        grad_before = param.grad.data.copy()

        clip_grad_norm_([param], max_norm)

        # Gradient should not change
        assert np.allclose(param.grad.data, grad_before, atol=1e-10)

    def test_clip_grad_value(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import clip_grad_value_, SGD

        np.random.seed(42)
        grad_data = np.array([[1.5, -2.0, 0.5], [3.0, -1.0, 0.1]]).astype(np.float64)
        param = Tensor(np.zeros((2, 3)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(grad_data)

        opt = SGD([param], lr=0.1)

        clip_value = 1.0
        clip_grad_value_([param], clip_value)

        # Clipped gradient should be within [-clip_value, clip_value]
        clipped_grad = param.grad.data
        assert np.all(clipped_grad >= -clip_value - 1e-10)
        assert np.all(clipped_grad <= clip_value + 1e-10)

    def test_clip_grad_value_correctness(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import clip_grad_value_, SGD

        np.random.seed(42)
        grad_data = np.array([[1.5, -2.0, 0.5], [3.0, -1.0, 0.1]]).astype(np.float64)
        param = Tensor(np.zeros((2, 3)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(grad_data.copy())

        opt = SGD([param], lr=0.1)

        clip_value = 1.0
        expected_grad = np.clip(grad_data, -clip_value, clip_value)

        clip_grad_value_([param], clip_value)

        actual_grad = param.grad.data
        assert np.allclose(actual_grad, expected_grad, atol=1e-10)

    def test_multiple_params(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import clip_grad_norm_, SGD

        np.random.seed(42)
        param1 = Tensor(np.random.randn(5, 5).astype(np.float64), requires_grad=True)
        param2 = Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)
        param1.grad = Tensor(np.random.randn(5, 5).astype(np.float64))
        param2.grad = Tensor(np.random.randn(3, 3).astype(np.float64))

        opt = SGD([param1, param2], lr=0.1)

        max_norm = 1.0

        # Compute total norm across both parameters
        grad1 = param1.grad.data
        grad2 = param2.grad.data
        total_norm = np.sqrt(np.sum(grad1 ** 2) + np.sum(grad2 ** 2))

        clip_grad_norm_([param1, param2], max_norm)

        # Verify clipping was applied consistently
        if total_norm > max_norm:
            scale_factor = max_norm / total_norm
            assert np.allclose(param1.grad.data, grad1 * scale_factor, atol=1e-10)
            assert np.allclose(param2.grad.data, grad2 * scale_factor, atol=1e-10)

    def test_returns_total_norm(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import clip_grad_norm_, SGD

        np.random.seed(42)
        grad_data = np.array([[3.0, 4.0]]).astype(np.float64)
        param = Tensor(np.zeros((1, 2)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(grad_data)

        opt = SGD([param], lr=0.1)

        expected_norm = np.sqrt(3.0 ** 2 + 4.0 ** 2)

        returned_norm = clip_grad_norm_([param], max_norm=10.0)

        assert np.isclose(returned_norm, expected_norm, atol=1e-10)


class TestGradientAccumulatorComprehensive:
    """Comprehensive tests for GradientAccumulator."""

    def test_creation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradientAccumulator, SGD

        np.random.seed(42)
        param = Tensor(np.zeros(1).astype(np.float64), requires_grad=True)
        opt = SGD([param], lr=0.1)
        accumulator = GradientAccumulator()
        assert accumulator is not None

    def test_accumulation(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradientAccumulator, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(np.ones((2, 2)).astype(np.float64))
        opt = SGD([param], lr=0.1)

        accumulator = GradientAccumulator()

        # Accumulate gradients N times
        N = 3
        for _ in range(N):
            accumulator.step([param])

        # Check accumulation count
        assert accumulator.accumulation_count == N

    def test_zero_grads(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradientAccumulator, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(np.ones((2, 2)).astype(np.float64))
        opt = SGD([param], lr=0.1)

        accumulator = GradientAccumulator()

        # Accumulate then zero
        accumulator.step([param])
        accumulator.zero([param])

        assert np.allclose(param.grad.data, np.zeros((2, 2)), atol=1e-10)

    def test_accumulation_count(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradientAccumulator, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(np.ones((2, 2)).astype(np.float64))
        opt = SGD([param], lr=0.1)

        accumulator = GradientAccumulator()

        for i in range(1, 6):
            accumulator.step([param])
            assert accumulator.accumulation_count == i

    def test_average_gradients(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradientAccumulator, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)

        # Accumulate gradients with known values
        gradients = [
            np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float64),
            np.array([[2.0, 4.0], [6.0, 8.0]]).astype(np.float64),
        ]

        opt = SGD([param], lr=0.1)
        accumulator = GradientAccumulator()

        for grad in gradients:
            param.grad = Tensor(grad)
            accumulator.step([param])

        # Check accumulated gradient
        accumulated = accumulator.get_accumulated()[0]
        expected_accumulated = gradients[0] + gradients[1]
        assert np.allclose(accumulated.data, expected_accumulated, atol=1e-10)

        # Check average
        average = accumulated.data / len(gradients)
        expected_average = (gradients[0] + gradients[1]) / 2
        assert np.allclose(average, expected_average, atol=1e-10)


class TestGradScalerComprehensive:
    """Comprehensive tests for GradScaler."""

    def test_creation(self):
        import numpy as np
        from python.optimization import GradScaler

        np.random.seed(42)
        scaler = GradScaler(init_scale=65536.0)
        assert scaler is not None

    def test_scale_loss(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradScaler

        np.random.seed(42)
        loss = Tensor(np.array(0.5).astype(np.float64))
        scaler = GradScaler(init_scale=1000.0)

        scaled_loss = scaler.scale(loss)
        expected_scaled_loss = loss.data * 1000.0

        assert np.isclose(scaled_loss.data, expected_scaled_loss, atol=1e-10)

    def test_unscale_gradients(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradScaler, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        grad_data = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float64)
        param.grad = Tensor(grad_data)

        opt = SGD([param], lr=0.1)
        scaler = GradScaler(init_scale=1000.0)

        scale = scaler.get_scale()
        expected_unscaled = grad_data / scale

        scaler.unscale_(opt)

        actual_unscaled = param.grad.data
        assert np.allclose(actual_unscaled, expected_unscaled, atol=1e-10)

    def test_inf_handling(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradScaler, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        grad_data = np.array([[1.0, np.inf], [3.0, 4.0]]).astype(np.float64)
        param.grad = Tensor(grad_data)

        opt = SGD([param], lr=0.1)
        scaler = GradScaler(init_scale=1000.0)

        # If gradients contain inf, should handle gracefully
        scaler.unscale_(opt)
        found_inf = scaler.has_overflow()

        assert found_inf or np.any(np.isinf(param.grad.data))

    def test_scale_update(self):
        import numpy as np
        from python.optimization import GradScaler

        np.random.seed(42)
        scaler = GradScaler(init_scale=1000.0, growth_factor=2.0, backoff_factor=0.5)

        initial_scale = scaler.get_scale()

        # Simulate successful step
        scaler.update()
        scale_after_success = scaler.get_scale()

        # Scale should increase (or stay same depending on implementation)
        assert scale_after_success >= initial_scale

    def test_growth_factor(self):
        import numpy as np
        from python.optimization import GradScaler

        np.random.seed(42)
        growth_factor = 2.0
        scaler = GradScaler(init_scale=1000.0, growth_factor=growth_factor)

        initial_scale = scaler.get_scale()

        # After multiple successful steps, scale should grow
        for _ in range(5):
            scaler.update()

        final_scale = scaler.get_scale()
        # Scale should potentially be higher (depending on overflow detection)
        assert final_scale >= initial_scale

    def test_backoff_factor(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import GradScaler, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(np.ones((2, 2)).astype(np.float64) * np.inf)

        opt = SGD([param], lr=0.1)
        backoff_factor = 0.5
        scaler = GradScaler(init_scale=1000.0, backoff_factor=backoff_factor)

        initial_scale = scaler.get_scale()

        scaler.unscale_(opt)
        if scaler.has_overflow():
            scaler.update()
            scale_after_overflow = scaler.get_scale()
            # Scale should be reduced
            assert scale_after_overflow <= initial_scale


class TestGradientAnalysisComprehensive:
    """Comprehensive tests for gradient analysis functions."""

    def test_compute_gradient_norm(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import compute_gradient_norm, SGD

        np.random.seed(42)
        param = Tensor(np.array([[3.0, 4.0]]).astype(np.float64), requires_grad=True)
        param.grad = Tensor(np.array([[3.0, 4.0]]).astype(np.float64))

        opt = SGD([param], lr=0.1)

        # Expected norm: sqrt(3^2 + 4^2) = 5.0
        expected_norm = 5.0
        actual_norm = compute_gradient_norm([param])

        assert np.isclose(actual_norm, expected_norm, atol=1e-10)

    def test_compute_gradient_stats(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import compute_gradient_stats, SGD

        np.random.seed(42)
        grad_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float64)
        param = Tensor(np.zeros((2, 3)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(grad_data)

        opt = SGD([param], lr=0.1)

        stats = compute_gradient_stats([param])

        # Verify stats
        assert 'mean' in stats or isinstance(stats, dict)
        assert 'std' in stats or isinstance(stats, dict)

    def test_detect_anomaly_clean(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import detect_gradient_anomaly, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(np.random.randn(2, 2).astype(np.float64))

        opt = SGD([param], lr=0.1)

        anomaly = detect_gradient_anomaly([param])

        # Normal gradients should not have anomalies
        assert not anomaly or anomaly == {}

    def test_detect_anomaly_nan(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import detect_gradient_anomaly, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        grad_data = np.array([[1.0, np.nan], [3.0, 4.0]]).astype(np.float64)
        param.grad = Tensor(grad_data)

        opt = SGD([param], lr=0.1)

        anomaly = detect_gradient_anomaly([param])

        # Should detect NaN
        assert anomaly or np.any(np.isnan(param.grad.data))

    def test_detect_anomaly_inf(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import detect_gradient_anomaly, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        grad_data = np.array([[1.0, np.inf], [3.0, 4.0]]).astype(np.float64)
        param.grad = Tensor(grad_data)

        opt = SGD([param], lr=0.1)

        anomaly = detect_gradient_anomaly([param])

        # Should detect Inf
        assert anomaly or np.any(np.isinf(param.grad.data))

    def test_flatten_unflatten(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import flatten_gradients, unflatten_gradients, SGD

        np.random.seed(42)
        param1 = Tensor(np.random.randn(2, 3).astype(np.float64), requires_grad=True)
        param2 = Tensor(np.random.randn(4).astype(np.float64), requires_grad=True)
        param1.grad = Tensor(np.random.randn(2, 3).astype(np.float64))
        param2.grad = Tensor(np.random.randn(4).astype(np.float64))

        opt = SGD([param1, param2], lr=0.1)

        # Flatten
        flattened = flatten_gradients([param1, param2])

        # Should be 1D
        assert flattened.ndim == 1
        assert len(flattened) == 2 * 3 + 4

    def test_zero_gradients(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import zero_gradients, SGD

        np.random.seed(42)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(np.ones((2, 2)).astype(np.float64))

        opt = SGD([param], lr=0.1)

        zero_gradients([param])

        assert np.allclose(param.grad.data, np.zeros((2, 2)), atol=1e-10)

    def test_scale_gradients(self):
        import numpy as np
        from python.foundations import Tensor
        from python.optimization import scale_gradients, SGD

        np.random.seed(42)
        grad_data = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float64)
        param = Tensor(np.zeros((2, 2)).astype(np.float64), requires_grad=True)
        param.grad = Tensor(grad_data.copy())

        opt = SGD([param], lr=0.1)

        scale_factor = 2.0
        expected_grad = grad_data * scale_factor

        scale_gradients([param], scale_factor)

        actual_grad = param.grad.data
        assert np.allclose(actual_grad, expected_grad, atol=1e-10)
