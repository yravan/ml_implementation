"""
Optimizers
==========

Gradient-based optimization algorithms for training neural networks.

This module provides comprehensive implementations of modern optimizers:

1. **First-Order Methods (Gradient-based)**:
   - SGD: The foundation, with momentum and Nesterov variants
   - RMSprop: Adaptive learning rates via running average of squared gradients
   - Adagrad: Adaptive learning with accumulated squared gradients
   - Adadelta: RMSprop improvement without learning rate parameter

2. **Adaptive Methods (Adam Family)**:
   - Adam: Adaptive moments (momentum + RMSprop)
   - AdamW: Adam with decoupled weight decay (preferred for Transformers)
   - NAdam: Adam with Nesterov momentum
   - RAdam: Rectified Adam with variance rectification
   - Adafactor: Memory-efficient Adam alternative
   - AMSGrad: Adam with max tracking for convergence guarantees

3. **Large-Scale Training**:
   - LAMB: Layer-wise Adaptive Moments for large batch training
   - LARS: Layer-wise Adaptive Rate Scaling

4. **Novel Optimizers**:
   - Lion: Evolved Sign Momentum (Google 2023)
   - Muon: Momentum Orthogonalized Update (novel momentum approach)

Theory
------
All optimizers solve: θ* = argmin_θ L(θ)

The general update is: θ_{t+1} = θ_t - η_t * d_t

Where d_t is the "update direction" which can be:
- Raw gradient: d = ∇L(θ) (vanilla SGD)
- Momentum-smoothed: d = momentum(∇L) (SGD+momentum)
- Preconditioned: d = P^{-1} * ∇L (natural gradient, Adam)
- Signed: d = sign(momentum(∇L)) (Lion)

Key concepts:
1. **Learning Rate (η)**: Step size. Too large → divergence, too small → slow
2. **Momentum**: Exponential moving average of gradients
3. **Adaptive LR**: Different learning rates for different parameters
4. **Weight Decay**: Regularization by shrinking weights toward zero

References
----------
- "An Overview of Gradient Descent Optimization Algorithms" - Ruder (2016)
  https://ruder.io/optimizing-gradient-descent/
- "Adam: A Method for Stochastic Optimization" - Kingma & Ba (2014)
  https://arxiv.org/abs/1412.6980
- "Decoupled Weight Decay Regularization" - Loshchilov & Hutter (2017)
  https://arxiv.org/abs/1711.05101
- "Large Batch Optimization for Deep Learning" (LAMB) - You et al. (2019)
  https://arxiv.org/abs/1904.00962
- "Symbolic Discovery of Optimization Algorithms" (Lion) - Chen et al. (2023)
  https://arxiv.org/abs/2302.06675
- "Muon: An optimizer for hidden layers in neural networks" (2024)
  https://kellerjordan.github.io/posts/muon/

Implementation Notes
--------------------
- All optimizers follow a consistent API: __init__, step, zero_grad, get_state, set_state
- State includes all running statistics (momentum buffers, etc.)
- Use float64 for optimizer states to avoid numerical issues
- Weight decay should be handled explicitly, not through L2 loss
"""
from abc import abstractmethod

# Implementation Status: NOT STARTED
# Complexity: Easy to Medium (varies by optimizer)
# Prerequisites: None

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Iterator, Union

from httplib2.auth import params
from sympy.physics.units import momentum

from python.foundations import Tensor


# =============================================================================
# Base Optimizer Class
# =============================================================================

class Optimizer:
    """
    Base class for all optimizers.

    Defines the common interface that all optimizers should implement.

    Example:
        >>> class MyOptimizer(Optimizer):
        ...     def step(self, gradients):
        ...         # Update parameters using gradients
        ...         pass
    """

    def __init__(self, params: Union[List[Tensor], List[Dict[str, Any]]], **defaults) -> None:
        """
        Initialize base optimizer.

        Args:
            params: List of parameter arrays to optimize or list of param dictionaries
            defaults: Dictionary of default hyperparameters
        """
        self.per_group_hyperparams = None
        self.defaults = defaults
        self.state: Dict[int, Dict[str, Any]] = {}  # Per-parameter state
        self._step_count = 0
        if isinstance(params[0], Tensor):
            for param in params:
                assert isinstance(param, Tensor)
            self.param_groups = [{"params": params}]
        else:
            self.param_groups = params
            for group in self.param_groups:
                assert "params" in group
                assert isinstance(group["params"], list)
                for param in group["params"]:
                    assert isinstance(param, Tensor)

    @abstractmethod
    def step(self) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError("Subclasses must implement step()")

    def zero_grad(self) -> None:
        """Reset optimizer state. Override in subclasses if needed."""
        for group in self.param_groups:
            for param in group["params"]:
                param.zero_grad()

    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        raise NotImplementedError("Subclasses must implement get_state()")

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state from checkpoint."""
        raise NotImplementedError("Subclasses must implement set_state()")

    def set_lr(self, lr: float) -> None:
        """Update learning rate."""
        self.defaults['lr'] = lr


# =============================================================================
# SGD and Variants
# =============================================================================

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and Nesterov acceleration.

    The fundamental optimization algorithm: θ = θ - lr * ∇L

    Variants:
    - Vanilla SGD: θ = θ - lr * g
    - Momentum: v = μv + g; θ = θ - lr * v
    - Nesterov: Evaluate gradient at lookahead position θ - μv
    - Weight Decay: Add λθ to gradient (L2 regularization)

    Math:
        # Vanilla SGD:
        θ_{t+1} = θ_t - η * g_t

        # Momentum:
        v_{t+1} = μ * v_t + g_t
        θ_{t+1} = θ_t - η * v_{t+1}

        # Nesterov (as implemented):
        v_{t+1} = μ * v_t + g_t
        θ_{t+1} = θ_t - η * (g_t + μ * v_{t+1})

    References:
        - Sutskever et al. "On the importance of initialization and momentum" (2013)
          https://proceedings.mlr.press/v28/sutskever13.html
        - Nesterov "A method for solving a convex programming problem" (1983)

    Example:
        >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
        >>> for batch in data:
        ...     grads = compute_gradients(model, batch)
        ...     optimizer.step(grads)
    """

    def __init__(self, params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 0.01,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 dampening: float = 0.0,
                 nesterov: bool = False) -> None:
        """
        Initialize SGD optimizer.

        Args:
            params: List of parameter arrays to optimize
            lr: Learning rate (step size)
            momentum: Momentum factor (0 = vanilla SGD, 0.9 typical)
            weight_decay: L2 regularization coefficient
            dampening: Dampening for momentum (usually 0)
            nesterov: Enable Nesterov accelerated gradient
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov)
        super().__init__(params, **defaults)
        self._initialize_velocities()

    def step(self) -> None:
        """
        Perform one SGD optimization step.
        """
        for group in self.param_groups:
            params = group['params']
            for p in params:
                # compute the gradient with weight decay
                weight_decay = group.get('weight_decay', self.defaults['weight_decay'])
                if weight_decay > 1e-8:
                    grad = p.grad + weight_decay * p.data
                else:
                    grad = p.grad

                # update the velocity in place
                momentum = group.get('momentum', self.defaults['momentum'])
                dampening = group.get('dampening', self.defaults['dampening'])
                if momentum > 1e-8:
                    velocity = self.velocities[p]
                    velocity *= momentum
                    if self._step_count > 0:
                        velocity += grad * (1 - dampening)
                    else:
                        velocity += grad
                else:
                    velocity = grad

                # perform gradient descent in place
                nesterov = group.get('nesterov', self.defaults['nesterov'])
                if nesterov:
                    descent_vector = velocity * momentum + p.grad
                else:
                    descent_vector = velocity
                lr = group.get('lr', self.defaults['lr'])
                p.data -= descent_vector * lr
        self._step_count += 1

    def _initialize_velocities(self) -> None:
        self.velocities = {}
        for group in self.param_groups:
            momentum = group.get('momentum', self.defaults['momentum'])
            if momentum > 1e-8:
                for p in group['params']:
                    self.velocities[p] = np.zeros_like(p.data)

class SGDW(SGD):
    """
    SGD with Decoupled Weight Decay.

    Regular SGD: θ = θ - lr * (g + λθ)  (weight decay in gradient)
    SGDW: θ = θ * (1 - lr*λ) - lr * g   (weight decay applied directly)

    For vanilla SGD these are mathematically equivalent, but decoupled
    weight decay is conceptually cleaner and matches AdamW formulation.

    Reference:
        Loshchilov & Hutter "Decoupled Weight Decay Regularization" (2017)
        https://arxiv.org/abs/1711.05101
    """

    def step(self) -> None:
        """
        Perform one SGD optimization step.
        """
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                # compute the gradient with weight decay
                weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
                lr = group.get("lr", self.defaults["lr"])
                if weight_decay > 1e-8:
                    p.data *= 1 - lr * weight_decay

                grad = p.grad

                # update the velocity in place
                momentum = group.get("momentum", self.defaults["momentum"])
                dampening = group.get("dampening", self.defaults["dampening"])
                if momentum > 1e-8:
                    velocity = self.velocities[p]
                    velocity *= momentum
                    if self._step_count > 0:
                        velocity += grad * (1 - dampening)
                    else:
                        velocity += grad
                else:
                    velocity = grad

                # perform gradient descent in place
                nesterov = group.get("nesterov", self.defaults["nesterov"])
                if nesterov:
                    descent_vector = velocity * momentum + p.grad
                else:
                    descent_vector = velocity
                p.data -= descent_vector * lr
        self._step_count += 1


# =============================================================================
# RMSprop and Adagrad Family
# =============================================================================

class RMSprop(Optimizer):
    """
    RMSprop: Root Mean Square Propagation.

    Adapts learning rate using running average of squared gradients.
    Parameters with larger gradients get smaller effective learning rates.

    Math:
        v_t = α * v_{t-1} + (1-α) * g_t²        # EMA of squared gradients
        θ_{t+1} = θ_t - lr * g_t / (√v_t + ε)

    RMSprop was proposed in Geoff Hinton's Coursera lectures (unpublished).
    It's the precursor to Adam (which adds momentum on top of RMSprop).

    When to use:
    - RNNs (original motivation)
    - When Adam doesn't work well
    - Reinforcement learning (often used in DQN)

    References:
        - Hinton's Coursera Lecture 6e
        - Tieleman & Hinton "Divide the gradient by a running average" (2012)

    Example:
        >>> optimizer = RMSprop(params, lr=0.001, alpha=0.99)
        >>> optimizer.step(gradients)
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 0.01,
                 alpha: float = 0.99,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 momentum: float = 0.0,
                 centered: bool = False):
        """
        Initialize RMSprop optimizer.

        Args:
            params: List of parameter arrays
            lr: Learning rate
            alpha: Smoothing constant (decay rate for squared gradient EMA)
            eps: Term added to denominator for numerical stability
            weight_decay: L2 penalty
            momentum: Momentum factor (optional, creates RMSprop with momentum)
            centered: If True, compute centered RMSprop (normalizes by variance)
        """
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        super().__init__(params, **defaults)
        self._initialize_velocities()
        self._initialize_grad_avg()
        self._initialize_buffer()

    def step(self) -> None:
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                momentum = group.get("momentum", self.defaults["momentum"])
                centered = group.get("centered", self.defaults["centered"])
                alpha = group.get("alpha", self.defaults["alpha"])
                lr = group.get("lr", self.defaults["lr"])
                eps = group.get("eps", self.defaults["eps"])
                weight_decay = group.get("weight_decay", self.defaults["weight_decay"])

                # compute the gradient with weight decay
                grad = p.grad
                if weight_decay > 1e-8:
                    grad = p.grad + weight_decay * p.data
                else:
                    grad = p.grad

                # update the velocity in place
                velocity = self.velocities[p]
                velocity *= alpha
                velocity += (1 - alpha) * grad ** 2

                # recenter
                if centered:
                    grad_avg = self.grad_avg[p]
                    grad_avg *= alpha; grad_avg += grad * (1 - alpha)
                    velocity = velocity - grad_avg ** 2

                # momentum
                if momentum > 1e-8:
                    descent_vector = self.buffer[p]
                    descent_vector *= momentum; descent_vector += grad / (np.sqrt(velocity) + eps)
                else:
                    descent_vector = p.grad / (np.sqrt(velocity) + eps)

                # perform gradient descent in place
                p.data -= descent_vector * lr
        self._step_count += 1

    def _initialize_velocities(self) -> None:
        self.velocities = {}
        for group in self.param_groups:
            alpha = group.get('alpha', self.defaults['alpha'])
            if alpha > 1e-8:
                for p in group['params']:
                    self.velocities[p] = np.zeros_like(p.data)

    def _initialize_grad_avg(self) -> None:
        self.grad_avg = {}
        for group in self.param_groups:
            centered = group.get('centered', self.defaults['centered'])
            if centered:
                for p in group['params']:
                    self.grad_avg[p] = np.zeros_like(p.data)

    def _initialize_buffer(self) -> None:
        self.buffer = {}
        for group in self.param_groups:
            momentum = group.get('momentum', self.defaults['momentum'])
            if momentum:
                for p in group['params']:
                    self.buffer[p] = np.zeros_like(p.data)


class Adagrad(Optimizer):
    """
    Adagrad: Adaptive Gradient Algorithm.

    Adapts learning rate for each parameter based on historical gradient information.
    Parameters that receive large gradients get smaller learning rates over time.

    Math:
        G_t = G_{t-1} + g_t²                    # Accumulated squared gradients
        θ_{t+1} = θ_t - lr * g_t / (√G_t + ε)

    Properties:
    - Learning rate decreases over time (can be problematic for long training)
    - Good for sparse features (e.g., NLP embeddings)
    - No hyperparameter tuning for learning rate decay

    Limitation: Accumulated G grows forever, so effective LR → 0.
    This is fixed by Adadelta and RMSprop (use EMA instead of sum).

    References:
        - Duchi et al. "Adaptive Subgradient Methods" (2011)
          https://jmlr.org/papers/v12/duchi11a.html

    Example:
        >>> optimizer = Adagrad(params, lr=0.01)
        >>> optimizer.step(gradients)
    """

    def __init__(self,
                 params: Union[List[Tensor], Dict[str, Any]],
                 lr: float = 0.01,
                 lr_decay: float = 0.0,
                 eps: float = 1e-10,
                 weight_decay: float = 0.0,
                 initial_accumulator_value: float = 0.0):
        """
        Initialize Adagrad optimizer.

        Args:
            params: List of parameter arrays
            lr: Learning rate
            lr_decay: Learning rate decay (applied every step)
            eps: Term for numerical stability
            weight_decay: L2 penalty
            initial_accumulator_value: Starting value for squared gradient sum
        """
        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)
        super().__init__(params, **defaults)
        self.state_sum = {}
        for group in self.param_groups:
            initial_accumulator = group.get('initial_accumulator_value', self.defaults['initial_accumulator_value'])
            for p in group['params']:
                self.state_sum[p] = np.zeros(p.shape).astype(np.float32)
                self.state_sum[p] += float(initial_accumulator)


    def step(self) -> None:
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                lr = group.get("lr", self.defaults["lr"])
                lr_decay = group.get("lr_decay", self.defaults["lr_decay"])
                eps = group.get("eps", self.defaults["eps"])
                weight_decay = group.get("weight_decay", self.defaults["weight_decay"])

                # compute the gradient with weight decay
                grad = p.grad
                if weight_decay > 1e-8:
                    grad = p.grad + weight_decay * p.data
                else:
                    grad = p.grad

                # update learning rate
                lr = lr * 1 / (1 + self._step_count * lr_decay)

                # update grad square in place
                G = self.state_sum[p]
                G += grad ** 2

                # perform gradient descent in place
                descent_vector = grad / (np.sqrt(G) + eps)
                p.data -= descent_vector * lr
        self._step_count += 1

class Adadelta(Optimizer):
    """
    Adadelta: An Adaptive Learning Rate Method.

    Extension of Adagrad that addresses its radically diminishing learning rates.
    Uses running averages of both gradients and parameter updates.

    Unique property: No learning rate hyperparameter needed!

    Math:
        g_t = gradient
        E[g²]_t = ρ * E[g²]_{t-1} + (1-ρ) * g_t²           # EMA of squared gradients
        Δθ_t = -√(E[Δθ²]_{t-1} + ε) / √(E[g²]_t + ε) * g_t # Update (note: uses old Δθ EMA)
        E[Δθ²]_t = ρ * E[Δθ²]_{t-1} + (1-ρ) * Δθ_t²       # EMA of squared updates
        θ_{t+1} = θ_t + Δθ_t

    The key insight: ratio of RMS(Δθ)/RMS(g) provides automatic learning rate.

    References:
        - Zeiler "ADADELTA: An Adaptive Learning Rate Method" (2012)
          https://arxiv.org/abs/1212.5701
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 1.0,
                 rho: float = 0.9,
                 eps: float = 1e-6,
                 weight_decay: float = 0.0):
        """
        Initialize Adadelta optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate multiplier (typically 1.0)
            rho: Decay rate for running averages (similar to momentum)
            eps: Small constant for numerical stability
            weight_decay: L2 penalty
        """
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize square_avg and acc_delta buffers for each parameter."""
        self.square_avg = {}
        self.acc_delta = {}
        for group in self.param_groups:
            for p in group['params']:
                self.square_avg[p] = np.zeros_like(p.data)
                self.acc_delta[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one Adadelta optimization step."""
        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            rho = group.get("rho", self.defaults["rho"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            for p in group['params']:
                grad = p.grad
                if weight_decay > 1e-8:
                    grad = grad + weight_decay * p.data

                # update expected value of grad squared
                square_avg = self.square_avg[p]
                square_avg *= rho ; square_avg += grad ** 2 * (1 - rho)

                # update expected value of delta
                acc_delta = self.acc_delta[p]
                descent =  np.sqrt(acc_delta + eps) / np.sqrt(square_avg + eps) * grad
                acc_delta *= rho ; acc_delta += descent ** 2 * (1 - rho)

                # update parameter
                p.data -= descent * lr
        self._step_count += 1


# =============================================================================
# Adam Family
# =============================================================================

class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation.

    The most widely used optimizer in deep learning. Combines:
    - Momentum (first moment): smoothed gradient direction
    - RMSprop (second moment): adaptive per-parameter learning rate

    Math:
        g_t = ∇L(θ_t)                               # Gradient
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t         # First moment (mean)
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²        # Second moment (variance)
        m̂_t = m_t / (1 - β₁^t)                      # Bias-corrected first moment
        v̂_t = v_t / (1 - β₂^t)                      # Bias-corrected second moment
        θ_{t+1} = θ_t - lr * m̂_t / (√v̂_t + ε)

    Bias correction is crucial early in training when m and v are initialized to 0.

    References:
        - Kingma & Ba "Adam: A Method for Stochastic Optimization" (2014)
          https://arxiv.org/abs/1412.6980

    Example:
        >>> optimizer = Adam(params, lr=0.001)
        >>> for batch in data:
        ...     grads = compute_gradients(model, batch)
        ...     optimizer.step(grads)
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False):
        """
        Initialize Adam optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate
            betas: (β₁, β₂) coefficients for moment estimates
            eps: Term for numerical stability
            weight_decay: L2 penalty (NOT recommended, use AdamW instead)
            amsgrad: Use AMSGrad variant (maintains max of past v_t)
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize first and second moment buffers for each parameter."""
        self.exp_avg = {}
        self.exp_avg_sq = {}
        self.max_exp_avg_sq = {}
        for group in self.param_groups:
            amsgrad = group.get('amsgrad', self.defaults['amsgrad'])
            for p in group['params']:
                self.exp_avg[p] = np.zeros_like(p.data)
                self.exp_avg_sq[p] = np.zeros_like(p.data)
                if amsgrad:
                    self.max_exp_avg_sq[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one Adam optimization step."""
        for group in self.param_groups:
            beta_1, beta_2 = group.get('betas', self.defaults['betas'])
            lr = group.get('lr', self.defaults['lr'])
            eps = group.get('eps', self.defaults['eps'])
            weight_decay = group.get('weight_decay', self.defaults['weight_decay'])
            amsgrad = group.get('amsgrad', self.defaults['amsgrad'])
            for p in group['params']:
                grad = p.grad
                if weight_decay > 1e-8:
                    grad = grad + weight_decay * p.data

                # update first moment
                exp_avg = self.exp_avg[p]
                exp_avg *= beta_1 ; exp_avg += grad * (1 - beta_1)
                corrected_exp_avg = exp_avg / (1 - beta_1 ** (self._step_count + 1))

                # update second moment
                exp_avg_sq = self.exp_avg_sq[p]
                exp_avg_sq *= beta_2 ; exp_avg_sq += grad**2 * (1 - beta_2)
                if amsgrad:
                    max_exp_avg_sq = self.max_exp_avg_sq[p]
                    np.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    corrected_exp_avg_sq = max_exp_avg_sq / (1 - beta_2**(self._step_count + 1))
                else:
                    corrected_exp_avg_sq = exp_avg_sq / (1 - beta_2 ** (self._step_count + 1))

                # update parameter
                descent = corrected_exp_avg / (np.sqrt(corrected_exp_avg_sq) + eps)
                p.data -= descent * lr
        self._step_count += 1

class AdamW(Optimizer):
    """
    AdamW: Adam with Decoupled Weight Decay.

    The key insight: L2 regularization and weight decay are NOT equivalent for Adam!

    - Adam + L2: gradient includes λθ, then adaptive scaling applies
    - AdamW: adaptive update, THEN weight decay (no scaling)

    AdamW is the preferred optimizer for Transformers and most modern architectures.

    Math:
        # Standard Adam update (without weight decay in gradient)
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        θ_{t+1} = θ_t - lr * (m̂_t / (√v̂_t + ε) + λ * θ_t)  # Decoupled!

    The weight decay λθ is NOT divided by √v, giving consistent regularization.

    References:
        - Loshchilov & Hutter "Decoupled Weight Decay Regularization" (2017)
          https://arxiv.org/abs/1711.05101
        - Used in BERT, GPT, and virtually all modern Transformers
    """

    def __init__(
        self,
        params: Union[List[Tensor], List[Dict[str, Any]]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        """
        Initialize Adam optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate
            betas: (β₁, β₂) coefficients for moment estimates
            eps: Term for numerical stability
            weight_decay: L2 penalty (NOT recommended, use AdamW instead)
            amsgrad: Use AMSGrad variant (maintains max of past v_t)
        """
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize first and second moment buffers for each parameter."""
        self.exp_avg = {}
        self.exp_avg_sq = {}
        self.max_exp_avg_sq = {}
        for group in self.param_groups:
            amsgrad = group.get("amsgrad", self.defaults["amsgrad"])
            for p in group["params"]:
                self.exp_avg[p] = np.zeros_like(p.data)
                self.exp_avg_sq[p] = np.zeros_like(p.data)
                if amsgrad:
                    self.max_exp_avg_sq[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one AdamW optimization step."""
        for group in self.param_groups:
            beta_1, beta_2 = group.get("betas", self.defaults["betas"])
            lr = group.get("lr", self.defaults["lr"])
            eps = group.get("eps", self.defaults["eps"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            amsgrad = group.get("amsgrad", self.defaults["amsgrad"])
            for p in group["params"]:
                grad = p.grad
                if weight_decay > 1e-8:
                    p.data *= (1 - weight_decay * lr)

                # update first moment
                exp_avg = self.exp_avg[p]
                exp_avg *= beta_1
                exp_avg += grad * (1 - beta_1)
                corrected_exp_avg = exp_avg / (1 - beta_1 ** (self._step_count + 1))

                # update second moment
                exp_avg_sq = self.exp_avg_sq[p]
                exp_avg_sq *= beta_2
                exp_avg_sq += grad**2 * (1 - beta_2)
                if amsgrad:
                    max_exp_avg_sq = self.max_exp_avg_sq[p]
                    np.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    corrected_exp_avg_sq = max_exp_avg_sq / (
                        1 - beta_2 ** (self._step_count + 1)
                    )
                else:
                    corrected_exp_avg_sq = exp_avg_sq / (
                        1 - beta_2 ** (self._step_count + 1)
                    )

                # update parameter
                descent = corrected_exp_avg / (np.sqrt(corrected_exp_avg_sq) + eps)
                p.data -= descent * lr
        self._step_count += 1


class NAdam(Optimizer):
    """
    NAdam: Nesterov-accelerated Adaptive Moment Estimation.

    Combines Adam with Nesterov momentum for potentially faster convergence.
    Instead of using current momentum, uses "lookahead" momentum.

    Math:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)

        # Nesterov modification: use lookahead momentum
        m̂_nesterov = β₁ * m̂_t + (1 - β₁) * g_t / (1 - β₁^t)
        θ_{t+1} = θ_t - lr * m̂_nesterov / (√v̂_t + ε)

    References:
        - Dozat "Incorporating Nesterov Momentum into Adam" (2016)
          https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 0.002,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 momentum_decay: float = 0.004):
        """
        Initialize NAdam optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate
            betas: (β₁, β₂) for moment estimates
            eps: Numerical stability term
            weight_decay: L2 penalty
            momentum_decay: Decay for momentum schedule
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize first and second moment buffers for each parameter."""
        self.exp_avg = {}
        self.exp_avg_sq = {}
        for group in self.param_groups:
            for p in group['params']:
                self.exp_avg[p] = np.zeros_like(p.data)
                self.exp_avg_sq[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one NAdam optimization step."""
        raise NotImplementedError("TODO: Implement NAdam step")


class RAdam(Optimizer):
    """
    RAdam: Rectified Adam.

    Addresses Adam's convergence issues in early training when variance
    estimate is unreliable. Automatically adjusts for variance bias.

    Key insight: Early in training, v_t has high variance because few
    samples have been seen. RAdam computes the "length" of the approximated
    SMA (simple moving average) and uses it to rectify the variance term.

    When variance estimate is unreliable (early training), RAdam falls back
    to SGD with momentum. As training progresses, it transitions to Adam.

    Math:
        ρ_∞ = 2/(1-β₂) - 1                      # Maximum SMA length
        ρ_t = ρ_∞ - 2*t*β₂^t/(1-β₂^t)          # Current SMA length

        if ρ_t > 5:  # Variance is tractable
            r_t = √((ρ_t-4)(ρ_t-2)ρ_∞ / ((ρ_∞-4)(ρ_∞-2)ρ_t))  # Rectification term
            θ = θ - lr * r_t * m̂ / √v̂
        else:  # Fall back to SGD with momentum
            θ = θ - lr * m̂

    References:
        - Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond" (2019)
          https://arxiv.org/abs/1908.03265
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0):
        """
        Initialize RAdam optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate
            betas: (β₁, β₂) for moment estimates
            eps: Numerical stability term
            weight_decay: L2 penalty
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize first and second moment buffers for each parameter."""
        self.exp_avg = {}
        self.exp_avg_sq = {}
        for group in self.param_groups:
            for p in group['params']:
                self.exp_avg[p] = np.zeros_like(p.data)
                self.exp_avg_sq[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one RAdam optimization step."""
        raise NotImplementedError("TODO: Implement RAdam step")


class Adafactor(Optimizer):
    """
    Adafactor: Memory-Efficient Adaptive Optimization.

    Reduces memory overhead of Adam by factorizing the second moment matrix.
    Instead of storing full v (same size as parameters), stores row and column
    running averages, reducing memory from O(nm) to O(n+m).

    Key ideas:
    1. Factorized second moment: v ≈ R * C (outer product of row/col means)
    2. No first moment by default (can be enabled)
    3. Relative step sizing (step bounded by parameter scale)

    Memory: Adam uses 2x parameters for m, v. Adafactor uses ~0.5x.

    Commonly used for training large language models (T5, etc.)

    References:
        - Shazeer & Stern "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" (2018)
          https://arxiv.org/abs/1804.04235
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: Optional[float] = None,
                 eps: Tuple[float, float] = (1e-30, 1e-3),
                 clip_threshold: float = 1.0,
                 decay_rate: float = -0.8,
                 beta1: Optional[float] = None,
                 weight_decay: float = 0.0,
                 scale_parameter: bool = True,
                 relative_step: bool = True,
                 warmup_init: bool = False):
        """
        Initialize Adafactor.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate (None for relative step sizing)
            eps: (eps1, eps2) regularization constants
            clip_threshold: Threshold for gradient clipping
            decay_rate: Coefficient for second moment decay
            beta1: First moment decay (None to disable)
            weight_decay: L2 penalty
            scale_parameter: Scale LR by root-mean-square of parameter
            relative_step: Use relative step size
            warmup_init: Use warmup initialization
        """
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate,
                        beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize factorized second moment buffers for each parameter."""
        self.exp_avg = {}
        self.exp_avg_sq_row = {}
        self.exp_avg_sq_col = {}
        self.exp_avg_sq = {}
        for group in self.param_groups:
            beta1 = group.get('beta1', self.defaults['beta1'])
            for p in group['params']:
                if beta1 is not None:
                    self.exp_avg[p] = np.zeros_like(p.data)
                if p.data.ndim >= 2:
                    self.exp_avg_sq_row[p] = np.zeros(p.data.shape[:-1])
                    self.exp_avg_sq_col[p] = np.zeros(p.data.shape[-1])
                else:
                    self.exp_avg_sq[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one Adafactor optimization step."""
        raise NotImplementedError("TODO: Implement Adafactor step")


# =============================================================================
# Large-Scale Training Optimizers
# =============================================================================

class LAMB(Optimizer):
    """
    LAMB: Layer-wise Adaptive Moments optimizer for Batch training.

    Enables training with very large batch sizes (e.g., 32K) by applying
    layer-wise learning rate scaling. Each layer gets its learning rate
    scaled by the ratio of parameter norm to update norm.

    This addresses the instability that occurs when scaling batch size:
    larger batches → smaller gradient variance → can use larger LR,
    but different layers may need different LR scaling.

    Math:
        # Compute Adam-style update
        m_t = β₁ * m_{t-1} + (1 - β₁) * g
        v_t = β₂ * v_{t-1} + (1 - β₂) * g²
        update = m̂_t / (√v̂_t + ε) + λ * θ  # AdamW-style weight decay

        # Layer-wise LR scaling (the "LAMB" part)
        r = ||θ|| / ||update||  # Trust ratio
        θ_{t+1} = θ_t - lr * r * update

    The trust ratio r ensures that updates are scaled appropriately
    for each layer based on its parameter magnitude.

    References:
        - You et al. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" (2019)
          https://arxiv.org/abs/1904.00962

    Example:
        >>> # Training BERT with large batch
        >>> optimizer = LAMB(params, lr=0.00176, weight_decay=0.01)
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0.0,
                 adam: bool = False):
        """
        Initialize LAMB optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate
            betas: (β₁, β₂) for moment estimates
            eps: Numerical stability term
            weight_decay: Decoupled weight decay
            adam: If True, disable layer-wise scaling (becomes AdamW)
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam=adam)
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize first and second moment buffers for each parameter."""
        self.exp_avg = {}
        self.exp_avg_sq = {}
        for group in self.param_groups:
            for p in group['params']:
                self.exp_avg[p] = np.zeros_like(p.data)
                self.exp_avg_sq[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one LAMB optimization step."""
        raise NotImplementedError("TODO: Implement LAMB step")


class LARS(Optimizer):
    """
    LARS: Layer-wise Adaptive Rate Scaling.

    Similar to LAMB but for SGD instead of Adam. Applies layer-wise
    learning rate scaling to enable large batch training with SGD.

    Math:
        # Local LR for each layer
        local_lr = trust_coefficient * ||θ|| / (||g|| + weight_decay * ||θ||)

        # Momentum update with local LR
        v = momentum * v + local_lr * (g + weight_decay * θ)
        θ = θ - global_lr * v

    References:
        - You et al. "Large Batch Training of Convolutional Networks" (2017)
          https://arxiv.org/abs/1708.03888

    Example:
        >>> # Large batch training with SGD
        >>> optimizer = LARS(params, lr=0.1, momentum=0.9)
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0,
                 trust_coefficient: float = 0.001,
                 eps: float = 1e-8):
        """
        Initialize LARS optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Global learning rate
            momentum: Momentum factor
            weight_decay: L2 penalty
            trust_coefficient: Trust coefficient for layer-wise scaling
            eps: Numerical stability term
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, **defaults)
        self._initialize_velocities()

    def _initialize_velocities(self) -> None:
        """Initialize momentum buffers for each parameter."""
        self.velocities = {}
        for group in self.param_groups:
            for p in group['params']:
                self.velocities[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one LARS optimization step."""
        raise NotImplementedError("TODO: Implement LARS step")


# =============================================================================
# Novel Optimizers
# =============================================================================

class Lion(Optimizer):
    """
    Lion: EvoLved Sign Momentum optimizer.

    Discovered through program search (AutoML), Lion uses only the SIGN
    of momentum, not its magnitude. This makes it more memory-efficient
    (no need to store second moment) and often faster.

    Surprisingly simple algorithm:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           # Momentum
        update = sign(β₂ * m_{t-1} + (1 - β₂) * g_t)  # Sign of interpolation
        θ_{t+1} = θ_t - lr * (update + λ * θ_t)       # Update with weight decay

    Properties:
    - Memory: Only stores momentum (like SGD), not second moment
    - Often achieves better results than AdamW on vision/language tasks
    - Requires smaller learning rate than AdamW (typically 10x smaller)
    - More sensitive to hyperparameters but faster per step

    References:
        - Chen et al. "Symbolic Discovery of Optimization Algorithms" (2023)
          https://arxiv.org/abs/2302.06675

    Example:
        >>> # Note: LR should be ~10x smaller than AdamW
        >>> optimizer = Lion(params, lr=1e-4, weight_decay=0.1)
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 1e-4,
                 betas: Tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        """
        Initialize Lion optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate (use ~10x smaller than AdamW)
            betas: (β₁, β₂) for momentum update and sign computation
            weight_decay: Decoupled weight decay (use larger than AdamW, e.g., 0.1)
        """
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize momentum buffer for each parameter (no second moment needed)."""
        self.exp_avg = {}
        for group in self.param_groups:
            for p in group['params']:
                self.exp_avg[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one Lion optimization step."""
        raise NotImplementedError("TODO: Implement Lion step")


class Muon(Optimizer):
    """
    Muon: Momentum Orthogonalized Update Normalization.

    A novel optimizer that orthogonalizes the update direction with respect
    to the momentum. This helps with optimization geometry, particularly
    in deep networks where gradient directions can be correlated.

    Key idea: Project out the component of the gradient that's parallel
    to the momentum, then normalize. This encourages exploration of new
    directions rather than reinforcing existing ones.

    Math:
        m_t = β * m_{t-1} + (1-β) * g_t             # Momentum
        g_orth = g_t - (g_t · m_t / ||m_t||²) * m_t # Orthogonalize gradient
        update = g_orth / ||g_orth||               # Normalize
        θ_{t+1} = θ_t - lr * update

    Note: This is a simplified description. The full Muon algorithm
    has additional components for stability and efficiency.

    References:
        - Jordan "Muon: An optimizer for hidden layers" (2024)
          https://kellerjordan.github.io/posts/muon/
        - Focuses on hidden layer optimization in deep networks

    Example:
        >>> optimizer = Muon(params, lr=0.02, momentum=0.95)
    """

    def __init__(self,
                 params: Union[List[Tensor], List[Dict[str, Any]]],
                 lr: float = 0.02,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 backend: str = 'newtonschulz',
                 backend_steps: int = 5):
        """
        Initialize Muon optimizer.

        Args:
            params: List of Tensor parameters or param groups
            lr: Learning rate
            momentum: Momentum factor
            nesterov: Use Nesterov momentum
            backend: Method for orthogonalization ('newtonschulz' or 'svd')
            backend_steps: Number of Newton-Schulz iterations
        """
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, **defaults)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize momentum buffer for each parameter."""
        self.momentum_buffer = {}
        for group in self.param_groups:
            for p in group['params']:
                self.momentum_buffer[p] = np.zeros_like(p.data)

    def step(self) -> None:
        """Perform one Muon optimization step."""
        raise NotImplementedError("TODO: Implement Muon step")

