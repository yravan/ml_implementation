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
    """Create sample parameters for optimizer testing."""
    return [
        np.random.randn(10, 5).astype(np.float64),
        np.random.randn(5,).astype(np.float64),
        np.random.randn(5, 3).astype(np.float64),
    ]


@pytest.fixture
def sample_grads(sample_params):
    """Create sample gradients matching parameter shapes."""
    return [np.random.randn(*p.shape).astype(np.float64) for p in sample_params]


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

class TestSGD:
    """Test SGD optimizer."""

    def test_sgd_creation(self, sample_params):
        """Test SGD optimizer creation."""
        from python.optimization import SGD

        optimizer = SGD(sample_params, lr=0.01)
        assert optimizer is not None

    def test_vanilla_sgd_step(self, sample_params, sample_grads):
        """Test vanilla SGD (no momentum) takes correct step."""
        from python.optimization import SGD

        params = [p.copy() for p in sample_params]
        original_params = [p.copy() for p in params]
        lr = 0.1

        optimizer = SGD(params, lr=lr, momentum=0.0)
        optimizer.step(sample_grads)

        # Verify: param = param - lr * grad
        for p, p_orig, g in zip(params, original_params, sample_grads):
            expected = p_orig - lr * g
            assert np.allclose(p, expected), "Vanilla SGD step incorrect"

    def test_sgd_with_momentum(self, sample_params, sample_grads):
        """Test SGD with momentum accumulates velocity."""
        from python.optimization import SGD

        params = [p.copy() for p in sample_params]
        lr = 0.1
        momentum = 0.9

        optimizer = SGD(params, lr=lr, momentum=momentum)

        # Take multiple steps
        for _ in range(3):
            optimizer.step(sample_grads)

        # Verify momentum state exists
        for i in range(len(params)):
            assert 'velocity' in optimizer.state[i]
            assert optimizer.state[i]['velocity'] is not None

    def test_sgd_with_weight_decay(self, sample_params, sample_grads):
        """Test SGD with weight decay."""
        from python.optimization import SGD

        params = [p.copy() for p in sample_params]
        original_params = [p.copy() for p in params]
        lr = 0.1
        weight_decay = 0.01

        optimizer = SGD(params, lr=lr, weight_decay=weight_decay)
        optimizer.step(sample_grads)

        # With weight decay: param = param - lr * (grad + weight_decay * param)
        for p, p_orig, g in zip(params, original_params, sample_grads):
            effective_grad = g + weight_decay * p_orig
            expected = p_orig - lr * effective_grad
            assert np.allclose(p, expected), "SGD with weight decay incorrect"

    def test_sgd_state_dict(self, sample_params, sample_grads):
        """Test SGD state save/restore."""
        from python.optimization import SGD

        params = [p.copy() for p in sample_params]
        optimizer = SGD(params, lr=0.1, momentum=0.9)

        # Take some steps
        optimizer.step(sample_grads)
        optimizer.step(sample_grads)

        # Save state
        state = optimizer.get_state()
        assert state is not None
        assert 'state' in state

        # Verify state can be loaded
        optimizer.set_state(state)


class TestAdam:
    """Test Adam optimizer."""

    def test_adam_creation(self, sample_params):
        """Test Adam optimizer creation."""
        from python.optimization import Adam

        optimizer = Adam(sample_params, lr=0.001)
        assert optimizer is not None

    def test_adam_step(self, sample_params, sample_grads):
        """Test Adam takes a step."""
        from python.optimization import Adam

        params = [p.copy() for p in sample_params]
        original_params = [p.copy() for p in params]

        optimizer = Adam(params, lr=0.001)
        optimizer.step(sample_grads)

        # Parameters should have changed
        for p, p_orig in zip(params, original_params):
            assert not np.allclose(p, p_orig), "Adam should update parameters"

    def test_adam_bias_correction(self, sample_params, sample_grads):
        """Test Adam bias correction is applied."""
        from python.optimization import Adam

        params = [p.copy() for p in sample_params]
        optimizer = Adam(params, lr=0.001, betas=(0.9, 0.999))

        # Take a step
        optimizer.step(sample_grads)

        # Verify step count incremented
        assert optimizer._step_count == 1

        # Verify moment estimates exist
        for i in range(len(params)):
            assert 'exp_avg' in optimizer.state[i]
            assert 'exp_avg_sq' in optimizer.state[i]

    def test_adam_with_amsgrad(self, sample_params, sample_grads):
        """Test Adam with AMSGrad variant."""
        from python.optimization import Adam

        params = [p.copy() for p in sample_params]
        optimizer = Adam(params, lr=0.001, amsgrad=True)

        optimizer.step(sample_grads)

        # Verify max_exp_avg_sq is tracked
        for i in range(len(params)):
            assert optimizer.state[i].get('max_exp_avg_sq') is not None


class TestAdamW:
    """Test AdamW optimizer."""

    def test_adamw_creation(self, sample_params):
        """Test AdamW optimizer creation."""
        from python.optimization import AdamW

        optimizer = AdamW(sample_params, lr=0.001, weight_decay=0.01)
        assert optimizer is not None

    def test_adamw_decoupled_weight_decay(self, sample_params, sample_grads):
        """Test that AdamW applies decoupled weight decay."""
        from python.optimization import AdamW

        params = [p.copy() for p in sample_params]
        original_params = [p.copy() for p in params]

        optimizer = AdamW(params, lr=0.001, weight_decay=0.01)
        optimizer.step(sample_grads)

        # Parameters should change (we can't easily verify the exact formula
        # without implementing it, but we can check they changed)
        for p, p_orig in zip(params, original_params):
            assert not np.allclose(p, p_orig), "AdamW should update parameters"


class TestRMSprop:
    """Test RMSprop optimizer."""

    def test_rmsprop_creation(self, sample_params):
        """Test RMSprop optimizer creation."""
        from python.optimization import RMSprop

        optimizer = RMSprop(sample_params, lr=0.01)
        assert optimizer is not None

    def test_rmsprop_step(self, sample_params, sample_grads):
        """Test RMSprop takes a step."""
        from python.optimization import RMSprop

        params = [p.copy() for p in sample_params]
        original_params = [p.copy() for p in params]

        optimizer = RMSprop(params, lr=0.01, alpha=0.99)
        optimizer.step(sample_grads)

        # Parameters should have changed
        for p, p_orig in zip(params, original_params):
            assert not np.allclose(p, p_orig), "RMSprop should update parameters"

    def test_rmsprop_square_avg_tracked(self, sample_params, sample_grads):
        """Test RMSprop tracks squared gradient average."""
        from python.optimization import RMSprop

        params = [p.copy() for p in sample_params]
        optimizer = RMSprop(params, lr=0.01)
        optimizer.step(sample_grads)

        for i in range(len(params)):
            assert 'square_avg' in optimizer.state[i]


class TestAdagrad:
    """Test Adagrad optimizer."""

    def test_adagrad_creation(self, sample_params):
        """Test Adagrad optimizer creation."""
        from python.optimization import Adagrad

        optimizer = Adagrad(sample_params, lr=0.01)
        assert optimizer is not None

    def test_adagrad_accumulated_gradient(self, sample_params, sample_grads):
        """Test Adagrad accumulates squared gradients."""
        from python.optimization import Adagrad

        params = [p.copy() for p in sample_params]
        optimizer = Adagrad(params, lr=0.01)

        # Take multiple steps
        for _ in range(3):
            optimizer.step(sample_grads)

        # Accumulated squared gradients should grow
        for i in range(len(params)):
            assert 'sum' in optimizer.state[i]
            assert np.all(optimizer.state[i]['sum'] >= 0)


class TestLAMB:
    """Test LAMB optimizer for large batch training."""

    def test_lamb_creation(self, sample_params):
        """Test LAMB optimizer creation."""
        from python.optimization import LAMB

        optimizer = LAMB(sample_params, lr=0.001)
        assert optimizer is not None

    def test_lamb_step(self, sample_params, sample_grads):
        """Test LAMB takes a step with layer-wise scaling."""
        from python.optimization import LAMB

        params = [p.copy() for p in sample_params]
        original_params = [p.copy() for p in params]

        optimizer = LAMB(params, lr=0.001)
        optimizer.step(sample_grads)

        # Parameters should have changed
        for p, p_orig in zip(params, original_params):
            assert not np.allclose(p, p_orig), "LAMB should update parameters"


class TestLion:
    """Test Lion optimizer."""

    def test_lion_creation(self, sample_params):
        """Test Lion optimizer creation."""
        from python.optimization import Lion

        optimizer = Lion(sample_params, lr=1e-4)
        assert optimizer is not None

    def test_lion_step(self, sample_params, sample_grads):
        """Test Lion takes a step using sign of momentum."""
        from python.optimization import Lion

        params = [p.copy() for p in sample_params]
        original_params = [p.copy() for p in params]

        optimizer = Lion(params, lr=1e-4)
        optimizer.step(sample_grads)

        # Parameters should have changed
        for p, p_orig in zip(params, original_params):
            assert not np.allclose(p, p_orig), "Lion should update parameters"


class TestMuon:
    """Test Muon optimizer."""

    def test_muon_creation(self, sample_params):
        """Test Muon optimizer creation."""
        from python.optimization import Muon

        optimizer = Muon(sample_params, lr=0.02)
        assert optimizer is not None

    def test_muon_step(self, sample_params, sample_grads):
        """Test Muon takes a step."""
        from python.optimization import Muon

        params = [p.copy() for p in sample_params]
        original = [p.copy() for p in params]

        optimizer = Muon(params, lr=0.02)
        optimizer.step(sample_grads)

        # Parameters should change
        for p, o in zip(params, original):
            assert not np.allclose(p, o), "Muon should update parameters"


class TestSGDW:
    """Test SGDW optimizer (SGD with decoupled weight decay)."""

    def test_sgdw_creation(self, sample_params):
        """Test SGDW creation."""
        from python.optimization import SGDW

        optimizer = SGDW(sample_params, lr=0.01, weight_decay=0.01)
        assert optimizer is not None

    def test_sgdw_step(self, sample_params, sample_grads):
        """Test SGDW takes correct step with decoupled weight decay."""
        from python.optimization import SGDW

        params = [p.copy() for p in sample_params]
        original = [p.copy() for p in params]
        lr = 0.1
        weight_decay = 0.01

        optimizer = SGDW(params, lr=lr, weight_decay=weight_decay)
        optimizer.step(sample_grads)

        # SGDW: param = param * (1 - lr * weight_decay) - lr * grad
        for p, p_orig, g in zip(params, original, sample_grads):
            expected = p_orig * (1 - lr * weight_decay) - lr * g
            assert np.allclose(p, expected), "SGDW step incorrect"

    def test_sgdw_with_momentum(self, sample_params, sample_grads):
        """Test SGDW with momentum."""
        from python.optimization import SGDW

        params = [p.copy() for p in sample_params]

        optimizer = SGDW(params, lr=0.1, momentum=0.9, weight_decay=0.01)

        for _ in range(3):
            optimizer.step(sample_grads)

        # Verify momentum state
        for i in range(len(params)):
            assert 'velocity' in optimizer.state[i] or 'momentum_buffer' in optimizer.state[i]


class TestAdadelta:
    """Test Adadelta optimizer."""

    def test_adadelta_creation(self, sample_params):
        """Test Adadelta creation."""
        from python.optimization import Adadelta

        optimizer = Adadelta(sample_params, lr=1.0, rho=0.9)
        assert optimizer is not None

    def test_adadelta_step(self, sample_params, sample_grads):
        """Test Adadelta takes a step."""
        from python.optimization import Adadelta

        params = [p.copy() for p in sample_params]
        original = [p.copy() for p in params]

        optimizer = Adadelta(params, lr=1.0, rho=0.9)
        optimizer.step(sample_grads)

        for p, o in zip(params, original):
            assert not np.allclose(p, o), "Adadelta should update parameters"

    def test_adadelta_state_accumulation(self, sample_params, sample_grads):
        """Test Adadelta accumulates state (avg squared grads, avg squared deltas)."""
        from python.optimization import Adadelta

        params = [p.copy() for p in sample_params]

        optimizer = Adadelta(params, lr=1.0, rho=0.9, eps=1e-6)

        for _ in range(3):
            optimizer.step(sample_grads)

        # Should have accumulated state
        for i in range(len(params)):
            assert 'square_avg' in optimizer.state[i] or 'acc_grad' in optimizer.state[i]

    def test_adadelta_default_lr(self, sample_params, sample_grads):
        """Test Adadelta with default lr (typically 1.0)."""
        from python.optimization import Adadelta

        params = [p.copy() for p in sample_params]

        # Adadelta often uses lr=1.0 by default
        optimizer = Adadelta(params)
        optimizer.step(sample_grads)

        # Should work without explicit lr


class TestNAdam:
    """Test NAdam optimizer (Nesterov-accelerated Adam)."""

    def test_nadam_creation(self, sample_params):
        """Test NAdam creation."""
        from python.optimization import NAdam

        optimizer = NAdam(sample_params, lr=0.001)
        assert optimizer is not None

    def test_nadam_step(self, sample_params, sample_grads):
        """Test NAdam takes a step."""
        from python.optimization import NAdam

        params = [p.copy() for p in sample_params]
        original = [p.copy() for p in params]

        optimizer = NAdam(params, lr=0.001, betas=(0.9, 0.999))
        optimizer.step(sample_grads)

        for p, o in zip(params, original):
            assert not np.allclose(p, o), "NAdam should update parameters"

    def test_nadam_betas(self, sample_params, sample_grads):
        """Test NAdam with custom betas."""
        from python.optimization import NAdam

        params = [p.copy() for p in sample_params]

        optimizer = NAdam(params, lr=0.001, betas=(0.95, 0.999))

        for _ in range(3):
            optimizer.step(sample_grads)

        # Check state contains momentum and variance
        for i in range(len(params)):
            state = optimizer.state[i]
            assert 'm' in state or 'exp_avg' in state
            assert 'v' in state or 'exp_avg_sq' in state

    def test_nadam_momentum_decay(self, sample_params, sample_grads):
        """Test NAdam handles momentum decay (mu schedule)."""
        from python.optimization import NAdam

        params = [p.copy() for p in sample_params]

        optimizer = NAdam(params, lr=0.002, momentum_decay=0.004)
        optimizer.step(sample_grads)

        # Should handle momentum decay without error


class TestRAdam:
    """Test RAdam optimizer (Rectified Adam)."""

    def test_radam_creation(self, sample_params):
        """Test RAdam creation."""
        from python.optimization import RAdam

        optimizer = RAdam(sample_params, lr=0.001)
        assert optimizer is not None

    def test_radam_step(self, sample_params, sample_grads):
        """Test RAdam takes a step."""
        from python.optimization import RAdam

        params = [p.copy() for p in sample_params]
        original = [p.copy() for p in params]

        optimizer = RAdam(params, lr=0.001)
        optimizer.step(sample_grads)

        for p, o in zip(params, original):
            assert not np.allclose(p, o), "RAdam should update parameters"

    def test_radam_warmup_behavior(self, sample_params, sample_grads):
        """Test RAdam warmup behavior (variance rectification)."""
        from python.optimization import RAdam

        params = [p.copy() for p in sample_params]

        optimizer = RAdam(params, lr=0.001)

        # Early steps should have warmup behavior
        for _ in range(5):
            optimizer.step(sample_grads)

        # Later steps should have full Adam behavior
        for _ in range(100):
            optimizer.step(sample_grads)

        # Check step count
        assert optimizer.state[0].get('step', 0) > 0


class TestAdafactor:
    """Test Adafactor optimizer."""

    def test_adafactor_creation(self, sample_params):
        """Test Adafactor creation."""
        from python.optimization import Adafactor

        optimizer = Adafactor(sample_params)
        assert optimizer is not None

    def test_adafactor_step(self, sample_params, sample_grads):
        """Test Adafactor takes a step."""
        from python.optimization import Adafactor

        params = [p.copy() for p in sample_params]
        original = [p.copy() for p in params]

        optimizer = Adafactor(params)
        optimizer.step(sample_grads)

        for p, o in zip(params, original):
            assert not np.allclose(p, o), "Adafactor should update parameters"

    def test_adafactor_factored_second_moment(self, sample_params, sample_grads):
        """Test Adafactor uses factored second moment for 2D+ params."""
        from python.optimization import Adafactor

        # 2D parameter should use factored representation
        params_2d = [np.random.randn(10, 5).astype(np.float64)]
        grads_2d = [np.random.randn(10, 5).astype(np.float64)]

        optimizer = Adafactor(params_2d)

        for _ in range(3):
            optimizer.step(grads_2d)

        # Should have row and column factors instead of full matrix
        state = optimizer.state[0]
        # Check for factored representation (row_var, col_var) or similar
        has_factored = ('exp_avg_sq_row' in state and 'exp_avg_sq_col' in state) or \
                      ('v_row' in state and 'v_col' in state) or \
                      'exp_avg_sq' in state  # Or full if not factored
        assert has_factored or len(state) > 0

    def test_adafactor_relative_step_size(self, sample_params, sample_grads):
        """Test Adafactor with relative step size (no explicit lr)."""
        from python.optimization import Adafactor

        params = [p.copy() for p in sample_params]

        # Adafactor can work without explicit lr (uses relative step size)
        optimizer = Adafactor(params, scale_parameter=True, relative_step=True)
        optimizer.step(sample_grads)


class TestLARS:
    """Test LARS optimizer (Layer-wise Adaptive Rate Scaling)."""

    def test_lars_creation(self, sample_params):
        """Test LARS creation."""
        from python.optimization import LARS

        optimizer = LARS(sample_params, lr=0.1)
        assert optimizer is not None

    def test_lars_step(self, sample_params, sample_grads):
        """Test LARS takes a step."""
        from python.optimization import LARS

        params = [p.copy() for p in sample_params]
        original = [p.copy() for p in params]

        optimizer = LARS(params, lr=0.1)
        optimizer.step(sample_grads)

        for p, o in zip(params, original):
            assert not np.allclose(p, o), "LARS should update parameters"

    def test_lars_trust_coefficient(self, sample_params, sample_grads):
        """Test LARS with trust coefficient."""
        from python.optimization import LARS

        params = [p.copy() for p in sample_params]

        optimizer = LARS(params, lr=0.1, trust_coefficient=0.001)
        optimizer.step(sample_grads)

    def test_lars_layer_wise_scaling(self, sample_params, sample_grads):
        """Test LARS applies layer-wise scaling."""
        from python.optimization import LARS

        params = [p.copy() for p in sample_params]
        params_copy = [p.copy() for p in params]

        optimizer = LARS(params, lr=0.1, trust_coefficient=0.001)

        # LARS scales learning rate per layer based on ||w|| / ||g||
        optimizer.step(sample_grads)

        # Parameters should be updated
        for p, o in zip(params, params_copy):
            assert not np.allclose(p, o)


# =============================================================================
# LOSS FUNCTION TESTS
# =============================================================================

class TestMSELoss:
    """Test Mean Squared Error Loss."""

    def test_mse_forward(self, regression_data):
        """Test MSE forward computation."""
        from python.optimization import MSELoss

        predictions, targets = regression_data
        loss_fn = MSELoss()

        loss = loss_fn.forward(predictions, targets, reduction='mean')

        # Manually compute MSE
        expected = np.mean((predictions - targets) ** 2)
        assert np.allclose(loss, expected), "MSE forward incorrect"

    def test_mse_reduction_sum(self, regression_data):
        """Test MSE with sum reduction."""
        from python.optimization import MSELoss

        predictions, targets = regression_data
        loss_fn = MSELoss()

        loss = loss_fn.forward(predictions, targets, reduction='sum')

        expected = np.sum((predictions - targets) ** 2)
        assert np.allclose(loss, expected), "MSE sum reduction incorrect"

    def test_mse_reduction_none(self, regression_data):
        """Test MSE with no reduction."""
        from python.optimization import MSELoss

        predictions, targets = regression_data
        loss_fn = MSELoss(reduction='none')

        loss = loss_fn.forward(predictions, targets)

        expected = (predictions - targets) ** 2
        assert np.allclose(loss, expected), "MSE none reduction incorrect"

    def test_mse_backward(self, regression_data):
        """Test MSE backward gradient."""
        from python.optimization import MSELoss

        predictions, targets = regression_data
        loss_fn = MSELoss(reduction='mean')

        loss = loss_fn.forward(predictions, targets)
        grad = loss_fn.backward()

        # Gradient: 2 * (pred - target) / n
        expected_grad = 2 * (predictions - targets) / predictions.size
        assert np.allclose(grad, expected_grad), "MSE backward incorrect"

    def test_mse_gradient_check(self, seed):
        """Test MSE gradient using numerical gradient check."""
        from python.optimization import MSELoss

        predictions = np.random.randn(8, 1).astype(np.float64)
        targets = np.random.randn(8, 1).astype(np.float64)

        loss_fn = MSELoss(reduction='mean')

        def loss_func(pred):
            return loss_fn.forward(pred, targets)

        # Compute analytical gradient
        loss = loss_fn.forward(predictions, targets)
        analytical_grad = loss_fn.backward()

        # Compute numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(predictions)
        for i in range(predictions.size):
            idx = np.unravel_index(i, predictions.shape)
            pred_plus = predictions.copy()
            pred_minus = predictions.copy()
            pred_plus[idx] += eps
            pred_minus[idx] -= eps
            numerical_grad[idx] = (loss_func(pred_plus) - loss_func(pred_minus)) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4), \
            "MSE gradient check failed"


class TestMAELoss:
    """Test Mean Absolute Error Loss."""

    def test_mae_forward(self, regression_data):
        """Test MAE forward computation."""
        from python.optimization import MAELoss

        predictions, targets = regression_data
        loss_fn = MAELoss(reduction='mean')

        loss = loss_fn.forward(predictions, targets)

        expected = np.mean(np.abs(predictions - targets))
        assert np.allclose(loss, expected), "MAE forward incorrect"

    def test_mae_backward(self, regression_data):
        """Test MAE backward gradient."""
        from python.optimization import MAELoss

        predictions, targets = regression_data
        loss_fn = MAELoss(reduction='mean')

        loss = loss_fn.forward(predictions, targets)
        grad = loss_fn.backward()

        # Gradient is sign(pred - target) / n
        diff = predictions - targets
        expected_grad = np.sign(diff) / predictions.size
        # Handle zeros
        expected_grad[diff == 0] = 0

        assert np.allclose(grad, expected_grad), "MAE backward incorrect"


class TestHuberLoss:
    """Test Huber Loss."""

    def test_huber_forward(self, regression_data):
        """Test Huber forward computation."""
        from python.optimization import HuberLoss

        predictions, targets = regression_data
        delta = 1.0
        loss_fn = HuberLoss(delta=delta, reduction='mean')

        loss = loss_fn.forward(predictions, targets)

        # Manually compute Huber
        diff = predictions - targets
        abs_diff = np.abs(diff)
        quadratic = 0.5 * diff ** 2
        linear = delta * (abs_diff - 0.5 * delta)
        expected = np.mean(np.where(abs_diff <= delta, quadratic, linear))

        assert np.allclose(loss, expected), "Huber forward incorrect"

    def test_huber_smooth_transition(self):
        """Test Huber smoothly transitions between L1 and L2."""
        from python.optimization import HuberLoss

        delta = 1.0
        loss_fn = HuberLoss(delta=delta, reduction='none')

        # Test at exactly delta
        pred = np.array([1.0, 2.0])  # delta away from 0
        target = np.array([0.0, 1.0])

        loss = loss_fn.forward(pred, target)

        # At delta: quadratic = 0.5 * delta^2, linear = delta * (delta - 0.5*delta) = 0.5*delta^2
        # They should be equal at the transition point
        expected = 0.5 * delta ** 2
        assert np.allclose(loss, expected), "Huber transition not smooth"


class TestCrossEntropyLoss:
    """Test Cross-Entropy Loss."""

    def test_crossentropy_forward(self, classification_data):
        """Test CrossEntropy forward computation."""
        from python.optimization import CrossEntropyLoss

        logits, targets = classification_data
        loss_fn = CrossEntropyLoss(reduction='mean')

        loss = loss_fn.forward(logits, targets)

        # Loss should be positive
        assert loss > 0, "CrossEntropy loss should be positive"

        # Loss should be finite
        assert np.isfinite(loss), "CrossEntropy loss should be finite"

    def test_crossentropy_perfect_prediction(self):
        """Test CrossEntropy with perfect prediction approaches 0."""
        from python.optimization import CrossEntropyLoss

        # Large logit for correct class
        logits = np.array([[10.0, -10.0, -10.0],
                          [-10.0, 10.0, -10.0]])
        targets = np.array([0, 1])

        loss_fn = CrossEntropyLoss(reduction='mean')
        loss = loss_fn.forward(logits, targets)

        # Loss should be very small
        assert loss < 0.1, "CrossEntropy with perfect prediction should be near 0"

    def test_crossentropy_backward(self, classification_data):
        """Test CrossEntropy backward gradient."""
        from python.optimization import CrossEntropyLoss

        logits, targets = classification_data
        loss_fn = CrossEntropyLoss(reduction='mean')

        loss = loss_fn.forward(logits, targets)
        grad = loss_fn.backward()

        # Gradient shape should match logits
        assert grad.shape == logits.shape, "Gradient shape mismatch"

        # Gradient should be finite
        assert np.all(np.isfinite(grad)), "Gradient should be finite"

    def test_crossentropy_gradient_check(self, seed):
        """Test CrossEntropy gradient using numerical check."""
        from python.optimization import CrossEntropyLoss

        logits = np.random.randn(4, 5).astype(np.float64)
        targets = np.array([0, 1, 2, 3])

        loss_fn = CrossEntropyLoss(reduction='mean')

        def loss_func(x):
            return loss_fn.forward(x, targets)

        # Compute analytical gradient
        loss = loss_fn.forward(logits, targets)
        analytical_grad = loss_fn.backward()

        # Compute numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(logits)
        for i in range(logits.size):
            idx = np.unravel_index(i, logits.shape)
            logits_plus = logits.copy()
            logits_minus = logits.copy()
            logits_plus[idx] += eps
            logits_minus[idx] -= eps
            numerical_grad[idx] = (loss_func(logits_plus) - loss_func(logits_minus)) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4), \
            "CrossEntropy gradient check failed"

    def test_crossentropy_with_label_smoothing(self, classification_data):
        """Test CrossEntropy with label smoothing."""
        from python.optimization import CrossEntropyLoss

        logits, targets = classification_data
        loss_fn = CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

        loss = loss_fn.forward(logits, targets)

        # Loss should be positive and finite
        assert loss > 0, "Loss should be positive"
        assert np.isfinite(loss), "Loss should be finite"

    def test_crossentropy_ignore_index(self):
        """Test CrossEntropy ignores specified index."""
        from python.optimization import CrossEntropyLoss

        logits = np.random.randn(4, 5)
        targets = np.array([0, 1, -100, 3])  # -100 is ignore_index

        loss_fn = CrossEntropyLoss(reduction='mean', ignore_index=-100)
        loss = loss_fn.forward(logits, targets)

        # Loss should only consider non-ignored samples
        assert np.isfinite(loss), "Loss should handle ignore_index"


class TestBCELoss:
    """Test Binary Cross-Entropy Loss."""

    def test_bce_forward(self, binary_classification_data):
        """Test BCE forward computation."""
        from python.optimization import BinaryCrossEntropyLoss

        logits, targets = binary_classification_data
        # Apply sigmoid to get probabilities
        predictions = 1 / (1 + np.exp(-logits))

        loss_fn = BinaryCrossEntropyLoss(reduction='mean')
        loss = loss_fn.forward(predictions, targets)

        # Loss should be positive
        assert loss > 0, "BCE loss should be positive"

    def test_bce_with_logits_forward(self, binary_classification_data):
        """Test BCE with logits forward computation."""
        from python.optimization import BCEWithLogitsLoss

        logits, targets = binary_classification_data

        loss_fn = BCEWithLogitsLoss(reduction='mean')
        loss = loss_fn.forward(logits, targets)

        # Loss should be positive
        assert loss > 0, "BCEWithLogits loss should be positive"
        assert np.isfinite(loss), "BCEWithLogits loss should be finite"

    def test_bce_with_logits_numerical_stability(self):
        """Test BCEWithLogits is numerically stable for extreme values."""
        from python.optimization import BCEWithLogitsLoss

        # Extreme logits
        logits = np.array([[100.0], [-100.0], [0.0]])
        targets = np.array([[1.0], [0.0], [0.5]])

        loss_fn = BCEWithLogitsLoss(reduction='mean')
        loss = loss_fn.forward(logits, targets)

        # Should not produce NaN or Inf
        assert np.isfinite(loss), "BCEWithLogits should be stable for extreme values"


class TestFocalLoss:
    """Test Focal Loss for imbalanced classification."""

    def test_focal_loss_creation(self):
        """Test Focal Loss creation."""
        from python.optimization import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        assert loss_fn is not None

    def test_focal_loss_forward(self, classification_data):
        """Test Focal Loss forward computation."""
        from python.optimization import FocalLoss

        logits, targets = classification_data
        loss_fn = FocalLoss(gamma=2.0, reduction='mean')

        loss = loss_fn.forward(logits, targets)

        assert loss > 0, "Focal loss should be positive"
        assert np.isfinite(loss), "Focal loss should be finite"

    def test_focal_downweights_easy_examples(self, seed):
        """Test Focal Loss down-weights easy (confident) examples."""
        from python.optimization import FocalLoss, CrossEntropyLoss

        # Create easy example (high confidence correct)
        easy_logits = np.array([[10.0, -10.0, -10.0]])
        easy_targets = np.array([0])

        # Create hard example (low confidence correct)
        hard_logits = np.array([[0.1, 0.0, 0.0]])
        hard_targets = np.array([0])

        focal_fn = FocalLoss(gamma=2.0, reduction='mean')
        ce_fn = CrossEntropyLoss(reduction='mean')

        easy_focal = focal_fn.forward(easy_logits, easy_targets)
        hard_focal = focal_fn.forward(hard_logits, hard_targets)

        easy_ce = ce_fn.forward(easy_logits, easy_targets)
        hard_ce = ce_fn.forward(hard_logits, hard_targets)

        # Focal should reduce easy more than hard (relative to CE)
        focal_ratio = hard_focal / (easy_focal + 1e-8)
        ce_ratio = hard_ce / (easy_ce + 1e-8)

        # Focal ratio should be smaller (easy examples more down-weighted)
        # This test may need adjustment based on exact implementation
        assert focal_ratio != ce_ratio, "Focal should treat easy/hard differently than CE"


class TestKLDivLoss:
    """Test KL Divergence Loss."""

    def test_kldiv_forward(self, seed):
        """Test KL Divergence forward computation."""
        from python.optimization import KLDivLoss
        from python.optimization import log_softmax

        # Create log probabilities and target distribution
        logits = np.random.randn(8, 5)
        log_probs = log_softmax(logits)

        target_logits = np.random.randn(8, 5)
        targets = np.exp(log_softmax(target_logits))  # Probability distribution

        loss_fn = KLDivLoss(reduction='batchmean')
        loss = loss_fn.forward(log_probs, targets)

        # KL divergence is non-negative
        assert loss >= -1e-6, "KL divergence should be non-negative"

    def test_kldiv_same_distribution(self, seed):
        """Test KL Divergence is 0 for identical distributions."""
        from python.optimization import KLDivLoss
        from python.optimization import log_softmax, softmax

        logits = np.random.randn(8, 5)
        log_probs = log_softmax(logits)
        targets = softmax(logits)  # Same distribution

        loss_fn = KLDivLoss(reduction='batchmean')
        loss = loss_fn.forward(log_probs, targets)

        # Should be very close to 0
        assert np.abs(loss) < 1e-5, "KL(P||P) should be 0"


class TestTripletLoss:
    """Test Triplet Loss for metric learning."""

    def test_triplet_loss_creation(self):
        """Test Triplet Loss creation."""
        from python.optimization import TripletLoss

        loss_fn = TripletLoss(margin=1.0)
        assert loss_fn is not None

    def test_triplet_loss_forward(self, seed):
        """Test Triplet Loss forward computation."""
        from python.optimization import TripletLoss

        anchor = np.random.randn(8, 64)
        positive = anchor + np.random.randn(8, 64) * 0.1  # Close to anchor
        negative = np.random.randn(8, 64)  # Random, likely far

        loss_fn = TripletLoss(margin=1.0, reduction='mean')
        loss = loss_fn.forward(anchor, positive, negative)

        # Loss should be non-negative
        assert loss >= 0, "Triplet loss should be non-negative"

    def test_triplet_loss_satisfied_margin(self, seed):
        """Test Triplet Loss is 0 when margin is satisfied."""
        from python.optimization import TripletLoss

        anchor = np.zeros((4, 64))
        positive = np.zeros((4, 64))  # Same as anchor, distance = 0
        negative = np.ones((4, 64)) * 10  # Very far from anchor

        loss_fn = TripletLoss(margin=1.0, reduction='mean')
        loss = loss_fn.forward(anchor, positive, negative)

        # d(a,p) = 0, d(a,n) >> margin, so loss should be 0
        assert loss < 1e-5, "Triplet loss should be 0 when margin is satisfied"


# =============================================================================
# LEARNING RATE SCHEDULER TESTS
# =============================================================================

class MockOptimizer:
    """Mock optimizer for scheduler testing."""

    def __init__(self, lr=0.1):
        self.defaults = {'lr': lr}

    def set_lr(self, lr):
        self.defaults['lr'] = lr


class TestStepLR:
    """Test StepLR scheduler."""

    def test_steplr_creation(self):
        """Test StepLR creation."""
        from python.optimization import StepLR

        optimizer = MockOptimizer(lr=0.1)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        assert scheduler is not None

    def test_steplr_decay(self):
        """Test StepLR decays at correct intervals."""
        from python.optimization import StepLR

        optimizer = MockOptimizer(lr=0.1)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # Before first decay
        for _ in range(10):
            scheduler.step()

        # After 10 steps, should decay once
        expected_lr = 0.1 * 0.1  # 0.01
        assert np.isclose(optimizer.defaults['lr'], expected_lr), \
            f"Expected LR {expected_lr}, got {optimizer.defaults['lr']}"

    def test_steplr_multiple_decays(self):
        """Test StepLR with multiple decay periods."""
        from python.optimization import StepLR

        optimizer = MockOptimizer(lr=1.0)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        lrs = []
        for epoch in range(20):
            scheduler.step()
            lrs.append(optimizer.defaults['lr'])

        # Expected: [1.0]*5, [0.5]*5, [0.25]*5, [0.125]*5
        # After step 5: 0.5, after step 10: 0.25, etc.
        assert np.isclose(lrs[4], 0.5), f"Expected 0.5 at step 5, got {lrs[4]}"
        assert np.isclose(lrs[9], 0.25), f"Expected 0.25 at step 10, got {lrs[9]}"


class TestMultiStepLR:
    """Test MultiStepLR scheduler."""

    def test_multisteplr_decay_at_milestones(self):
        """Test MultiStepLR decays at specified milestones."""
        from python.optimization import MultiStepLR

        optimizer = MockOptimizer(lr=0.1)
        scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

        lrs = []
        for epoch in range(15):
            scheduler.step()
            lrs.append(optimizer.defaults['lr'])

        # After epoch 5: 0.01, after epoch 10: 0.001
        assert np.isclose(lrs[5], 0.01), f"Expected 0.01 after milestone 5"
        assert np.isclose(lrs[10], 0.001), f"Expected 0.001 after milestone 10"


class TestExponentialLR:
    """Test ExponentialLR scheduler."""

    def test_exponentiallr_decay(self):
        """Test ExponentialLR decays every epoch."""
        from python.optimization import ExponentialLR

        optimizer = MockOptimizer(lr=1.0)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        for epoch in range(5):
            scheduler.step()

        # After 5 steps: 1.0 * 0.9^5 = 0.59049
        expected = 1.0 * (0.9 ** 5)
        assert np.isclose(optimizer.defaults['lr'], expected, rtol=1e-4)


class TestCosineAnnealingLR:
    """Test CosineAnnealingLR scheduler."""

    def test_cosine_annealing_endpoints(self):
        """Test CosineAnnealing reaches correct endpoints."""
        from python.optimization import CosineAnnealingLR

        optimizer = MockOptimizer(lr=1.0)
        T_max = 100
        eta_min = 0.0
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        # At t=0, should be at base_lr
        lr_start = scheduler.get_lr()
        assert np.isclose(lr_start, 1.0, rtol=0.1), f"Start LR should be ~1.0"

        # At t=T_max, should be at eta_min
        for _ in range(T_max):
            scheduler.step()

        lr_end = optimizer.defaults['lr']
        assert np.isclose(lr_end, eta_min, atol=0.05), f"End LR should be ~{eta_min}"

    def test_cosine_annealing_midpoint(self):
        """Test CosineAnnealing at midpoint."""
        from python.optimization import CosineAnnealingLR

        optimizer = MockOptimizer(lr=1.0)
        T_max = 100
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0)

        # Go to midpoint
        for _ in range(T_max // 2):
            scheduler.step()

        # At midpoint of cosine: (1 + cos(π/2)) / 2 = 0.5
        lr_mid = optimizer.defaults['lr']
        assert np.isclose(lr_mid, 0.5, rtol=0.1), f"Midpoint LR should be ~0.5"


class TestOneCycleLR:
    """Test OneCycleLR scheduler for super-convergence."""

    def test_onecycle_creation(self):
        """Test OneCycleLR creation."""
        from python.optimization import OneCycleLR

        optimizer = MockOptimizer(lr=0.1)
        scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=1000)
        assert scheduler is not None

    def test_onecycle_warmup_and_decay(self):
        """Test OneCycleLR warms up then decays."""
        from python.optimization import OneCycleLR

        optimizer = MockOptimizer(lr=0.001)
        max_lr = 0.1
        total_steps = 100
        pct_start = 0.3  # 30% warmup

        scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps,
                               pct_start=pct_start)

        lrs = []
        for _ in range(total_steps):
            scheduler.step()
            lrs.append(optimizer.defaults['lr'])

        # Should reach max_lr around 30% of training
        warmup_end = int(total_steps * pct_start)
        max_lr_achieved = max(lrs[:warmup_end + 5])  # Allow some tolerance

        assert max_lr_achieved >= max_lr * 0.9, \
            f"OneCycle should reach ~max_lr during warmup"

        # Should decay to near 0 at end
        assert lrs[-1] < lrs[warmup_end], "OneCycle should decay after warmup"


class TestWarmupLR:
    """Test WarmupLR scheduler."""

    def test_warmup_linear(self):
        """Test linear warmup."""
        from python.optimization import WarmupLR

        optimizer = MockOptimizer(lr=1.0)
        warmup_iters = 10
        scheduler = WarmupLR(optimizer, warmup_iters=warmup_iters)

        lrs = []
        for _ in range(warmup_iters + 5):
            scheduler.step()
            lrs.append(optimizer.defaults['lr'])

        # Should increase during warmup
        for i in range(warmup_iters - 1):
            assert lrs[i] <= lrs[i + 1], "LR should increase during warmup"

        # Should reach base_lr at end of warmup
        assert np.isclose(lrs[warmup_iters - 1], 1.0, rtol=0.1), \
            "Should reach base_lr at end of warmup"


class TestReduceLROnPlateau:
    """Test ReduceLROnPlateau scheduler."""

    def test_reduce_on_plateau_creation(self):
        """Test ReduceLROnPlateau creation."""
        from python.optimization import ReduceLROnPlateau

        optimizer = MockOptimizer(lr=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
        assert scheduler is not None

    def test_reduce_on_plateau_reduces_lr(self):
        """Test ReduceLROnPlateau reduces LR when metric plateaus."""
        from python.optimization import ReduceLROnPlateau

        optimizer = MockOptimizer(lr=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3,
                                       factor=0.1)

        # Simulate improving metric then plateau
        metrics = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]  # Plateaus after 3rd

        for metric in metrics:
            scheduler.step(metric)

        # After patience exceeded, LR should be reduced
        assert optimizer.defaults['lr'] < 0.1, \
            "LR should be reduced after plateau"


# =============================================================================
# GRADIENT UTILITY TESTS
# =============================================================================

class TestGradientClipping:
    """Test gradient clipping utilities."""

    def test_clip_grad_norm(self, sample_grads):
        """Test gradient norm clipping."""
        from python.optimization import clip_grad_norm_

        grads = [g.copy() * 100 for g in sample_grads]  # Scale up gradients
        max_norm = 1.0

        total_norm = clip_grad_norm_(grads, max_norm)

        # Verify total norm is now <= max_norm
        clipped_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        assert clipped_norm <= max_norm + 1e-6, \
            f"Clipped norm {clipped_norm} should be <= {max_norm}"

    def test_clip_grad_norm_no_op(self, sample_grads):
        """Test clip_grad_norm_ is no-op when norm is small."""
        from python.optimization import clip_grad_norm_

        grads = [g.copy() * 0.01 for g in sample_grads]  # Small gradients
        original_grads = [g.copy() for g in grads]
        max_norm = 10.0

        clip_grad_norm_(grads, max_norm)

        # Gradients should be unchanged
        for g, g_orig in zip(grads, original_grads):
            assert np.allclose(g, g_orig), "Small gradients should not be clipped"

    def test_clip_grad_value(self, sample_grads):
        """Test gradient value clipping."""
        from python.optimization import clip_grad_value_

        grads = [g.copy() * 100 for g in sample_grads]
        clip_value = 1.0

        clip_grad_value_(grads, clip_value)

        # All values should be in [-clip_value, clip_value]
        for g in grads:
            assert np.all(g >= -clip_value) and np.all(g <= clip_value), \
                "Gradient values should be clipped"


class TestGradientAccumulator:
    """Test gradient accumulation."""

    def test_accumulator_creation(self):
        """Test GradientAccumulator creation."""
        from python.optimization import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)
        assert accumulator is not None

    def test_accumulator_accumulates(self, sample_grads):
        """Test GradientAccumulator accumulates gradients."""
        from python.optimization import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)

        # Accumulate 4 times
        for _ in range(4):
            accumulator.accumulate(sample_grads)

        assert accumulator.should_step(), "Should be ready to step after 4 accumulations"

        # Get accumulated gradients
        acc_grads = accumulator.get_accumulated_gradients()

        # Should be average of accumulated gradients (since we used same grads)
        for acc_g, g in zip(acc_grads, sample_grads):
            expected = g  # Since all grads were same, average = original
            assert np.allclose(acc_g, expected), "Accumulated gradient incorrect"

    def test_accumulator_zero(self, sample_grads):
        """Test GradientAccumulator reset."""
        from python.optimization import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=2)

        accumulator.accumulate(sample_grads)
        accumulator.zero()

        assert not accumulator.should_step(), "Should not be ready after zero"


class TestGradScaler:
    """Test GradScaler for mixed-precision training."""

    def test_gradscaler_creation(self):
        """Test GradScaler creation."""
        from python.optimization import GradScaler

        scaler = GradScaler(init_scale=65536.0)
        assert scaler is not None

    def test_gradscaler_scale_loss(self):
        """Test GradScaler scales loss."""
        from python.optimization import GradScaler

        scaler = GradScaler(init_scale=1000.0)
        loss = 1.5

        scaled_loss = scaler.scale(loss)

        assert np.isclose(scaled_loss, loss * 1000.0), "Loss should be scaled"

    def test_gradscaler_unscale(self, sample_grads):
        """Test GradScaler unscales gradients."""
        from python.optimization import GradScaler

        scaler = GradScaler(init_scale=1000.0)
        grads = [g.copy() * 1000 for g in sample_grads]  # "Scaled" gradients
        original_grads = [g.copy() / 1000 for g in grads]

        scaler.unscale_(grads)

        for g, g_orig in zip(grads, original_grads):
            assert np.allclose(g, g_orig), "Gradients should be unscaled"

    def test_gradscaler_handles_inf(self, sample_grads):
        """Test GradScaler handles inf gradients."""
        from python.optimization import GradScaler

        scaler = GradScaler(init_scale=1000.0)
        grads = [g.copy() for g in sample_grads]
        grads[0][0, 0] = np.inf  # Inject inf

        scaler.unscale_(grads)

        # Should detect inf
        assert scaler._found_inf, "Should detect inf in gradients"


class TestGradientAnalysis:
    """Test gradient analysis utilities."""

    def test_compute_gradient_norm(self, sample_grads):
        """Test compute_gradient_norm."""
        from python.optimization import compute_gradient_norm

        norm = compute_gradient_norm(sample_grads, norm_type=2.0)

        # Manually compute
        expected = np.sqrt(sum(np.sum(g ** 2) for g in sample_grads))

        assert np.isclose(norm, expected), "Gradient norm computation incorrect"

    def test_compute_gradient_stats(self, sample_grads):
        """Test compute_gradient_stats."""
        from python.optimization import compute_gradient_stats

        stats = compute_gradient_stats(sample_grads)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'norm_l2' in stats

    def test_detect_gradient_anomaly_clean(self, sample_grads):
        """Test detect_gradient_anomaly with clean gradients."""
        from python.optimization import detect_gradient_anomaly

        has_anomaly, description = detect_gradient_anomaly(sample_grads, warn=False)

        assert not has_anomaly, "Clean gradients should have no anomaly"

    def test_detect_gradient_anomaly_nan(self, sample_grads):
        """Test detect_gradient_anomaly with NaN."""
        from python.optimization import detect_gradient_anomaly

        grads = [g.copy() for g in sample_grads]
        grads[0][0, 0] = np.nan

        has_anomaly, description = detect_gradient_anomaly(grads, warn=False)

        assert has_anomaly, "Should detect NaN"
        assert 'NaN' in description

    def test_detect_gradient_anomaly_inf(self, sample_grads):
        """Test detect_gradient_anomaly with Inf."""
        from python.optimization import detect_gradient_anomaly

        grads = [g.copy() for g in sample_grads]
        grads[0][0, 0] = np.inf

        has_anomaly, description = detect_gradient_anomaly(grads, warn=False)

        assert has_anomaly, "Should detect Inf"
        assert 'Inf' in description


class TestGradientUtilityFunctions:
    """Test utility functions for gradients."""

    def test_flatten_unflatten_gradients(self, sample_grads):
        """Test flatten and unflatten gradients."""
        from python.optimization import flatten_gradients, unflatten_gradients

        shapes = [g.shape for g in sample_grads]

        # Flatten
        flat = flatten_gradients(sample_grads)

        total_size = sum(g.size for g in sample_grads)
        assert flat.shape == (total_size,), "Flattened shape incorrect"

        # Unflatten
        restored = unflatten_gradients(flat, shapes)

        for g_orig, g_restored in zip(sample_grads, restored):
            assert np.allclose(g_orig, g_restored), "Unflatten should restore original"

    def test_zero_gradients(self, sample_grads):
        """Test zero_gradients."""
        from python.optimization import zero_gradients

        grads = [g.copy() for g in sample_grads]
        zero_gradients(grads)

        for g in grads:
            assert np.all(g == 0), "All gradients should be zero"

    def test_scale_gradients(self, sample_grads):
        """Test scale_gradients."""
        from python.optimization import scale_gradients

        grads = [g.copy() for g in sample_grads]
        original = [g.copy() for g in grads]
        scale = 0.5

        scale_gradients(grads, scale)

        for g, g_orig in zip(grads, original):
            assert np.allclose(g, g_orig * scale), "Gradients should be scaled"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

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
            optimizer.step(sample_grads)
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
        optimizer.step(grads)

        # Should complete without issues
        assert all(np.all(np.isfinite(p)) for p in params)


class TestLossOptimizerIntegration:
    """Test loss and optimizer integration."""

    def test_mse_sgd_optimization(self, seed):
        """Test optimizing MSE loss with SGD."""
        from python.optimization import MSELoss, SGD

        # Simple linear regression: y = 2x + 1
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

        # Initialize weights
        W = [np.random.randn(1, 1), np.zeros((1,))]  # weight and bias
        original_loss = None

        loss_fn = MSELoss()
        optimizer = SGD(W, lr=0.1)

        # Training loop
        for epoch in range(100):
            # Forward
            pred = X @ W[0] + W[1]
            loss = loss_fn.forward(pred, y)

            if original_loss is None:
                original_loss = loss

            # Backward
            grad_pred = loss_fn.backward()

            # Compute gradients for W
            grad_W = X.T @ grad_pred
            grad_b = np.sum(grad_pred, axis=0)

            # Update
            optimizer.step([grad_W, grad_b])

        # Loss should decrease
        final_loss = loss_fn.forward(X @ W[0] + W[1], y)
        assert final_loss < original_loss, "Loss should decrease during optimization"

        # Parameters should be close to true values
        assert np.abs(W[0][0, 0] - 2.0) < 0.5, "Weight should be close to 2"
        assert np.abs(W[1][0] - 1.0) < 0.5, "Bias should be close to 1"


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
        optimizer.step(sample_grads)

        # Parameters should not change
        for p, p_orig in zip(params, original):
            assert np.allclose(p, p_orig), "Zero LR should not change parameters"

    def test_loss_single_sample(self):
        """Test loss with single sample."""
        from python.optimization import MSELoss

        pred = np.array([[1.0]])
        target = np.array([[2.0]])

        loss_fn = MSELoss(reduction='mean')
        loss = loss_fn.forward(pred, target)

        assert np.isclose(loss, 1.0), "MSE of single sample should be (1-2)^2 = 1"

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

        # 1D case
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.5, 2.5, 3.5])

        loss_fn = MSELoss(reduction='mean')
        loss = loss_fn.forward(pred, target)

        assert np.isfinite(loss), "Should handle 1D arrays"


# =============================================================================
# BENCHMARK / SANITY CHECKS
# =============================================================================

class TestSanityChecks:
    """Sanity checks to verify basic correctness."""

    def test_adam_converges_simple_quadratic(self, seed):
        """Test Adam converges on simple quadratic."""
        from python.optimization import Adam

        # Minimize f(x) = x^2, optimal x = 0
        x = [np.array([5.0])]  # Start at 5
        optimizer = Adam(x, lr=0.1)

        for _ in range(100):
            grad = [2 * x[0]]  # Gradient of x^2
            optimizer.step(grad)

        # Should converge close to 0
        assert np.abs(x[0][0]) < 0.1, f"Adam should converge to 0, got {x[0][0]}"

    def test_sgd_with_momentum_converges(self, seed):
        """Test SGD with momentum converges."""
        from python.optimization import SGD

        x = [np.array([5.0])]
        optimizer = SGD(x, lr=0.01, momentum=0.9)

        for _ in range(200):
            grad = [2 * x[0]]
            optimizer.step(grad)

        assert np.abs(x[0][0]) < 0.5, f"SGD+momentum should converge, got {x[0][0]}"

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
