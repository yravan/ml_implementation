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

# Implementation Status: NOT STARTED
# Complexity: Easy to Medium (varies by optimizer)
# Prerequisites: None

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Iterator, Union


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

    def __init__(self, params: List[np.ndarray], defaults: Dict[str, Any]):
        """
        Initialize base optimizer.

        Args:
            params: List of parameter arrays to optimize
            defaults: Dictionary of default hyperparameters
        """
        self.params = params
        self.defaults = defaults
        self.state: Dict[int, Dict[str, Any]] = {}  # Per-parameter state
        self._step_count = 0

    def step(self, gradients: List[np.ndarray]) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError("Subclasses must implement step()")

    def zero_grad(self) -> None:
        """Reset optimizer state. Override in subclasses if needed."""
        pass

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

    def __init__(self,
                 params: List[np.ndarray],
                 lr: float = 0.01,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 dampening: float = 0.0,
                 nesterov: bool = False):
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
        raise NotImplementedError(
            "TODO: Initialize SGD optimizer\n"
            "Hint:\n"
            "  defaults = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay,\n"
            "              'dampening': dampening, 'nesterov': nesterov}\n"
            "  super().__init__(params, defaults)\n"
            "  \n"
            "  # Initialize velocity buffer for each parameter\n"
            "  for i, p in enumerate(params):\n"
            "      self.state[i] = {'velocity': np.zeros_like(p)}"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        """
        Perform one SGD optimization step.

        Args:
            gradients: List of gradient arrays, same order as params
        """
        raise NotImplementedError(
            "TODO: Implement SGD step\n"
            "Hint:\n"
            "  lr = self.defaults['lr']\n"
            "  momentum = self.defaults['momentum']\n"
            "  weight_decay = self.defaults['weight_decay']\n"
            "  dampening = self.defaults['dampening']\n"
            "  nesterov = self.defaults['nesterov']\n"
            "  \n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      # Apply weight decay\n"
            "      if weight_decay != 0:\n"
            "          grad = grad + weight_decay * param\n"
            "      \n"
            "      # Apply momentum\n"
            "      if momentum != 0:\n"
            "          v = self.state[i]['velocity']\n"
            "          v = momentum * v + (1 - dampening) * grad\n"
            "          self.state[i]['velocity'] = v\n"
            "          \n"
            "          if nesterov:\n"
            "              grad = grad + momentum * v\n"
            "          else:\n"
            "              grad = v\n"
            "      \n"
            "      # Update parameter\n"
            "      param -= lr * grad"
        )

    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        raise NotImplementedError(
            "TODO: Return optimizer state\n"
            "Hint:\n"
            "  return {\n"
            "      'state': {i: {'velocity': s['velocity'].copy()} for i, s in self.state.items()},\n"
            "      'defaults': self.defaults.copy(),\n"
            "      'step_count': self._step_count\n"
            "  }"
        )

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state from checkpoint."""
        raise NotImplementedError(
            "TODO: Restore optimizer state\n"
            "Hint:\n"
            "  self.state = {i: {'velocity': s['velocity'].copy()} for i, s in state['state'].items()}\n"
            "  self.defaults = state['defaults'].copy()\n"
            "  self._step_count = state['step_count']"
        )


class SGDW(Optimizer):
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

    def __init__(self,
                 params: List[np.ndarray],
                 lr: float = 0.01,
                 momentum: float = 0.0,
                 weight_decay: float = 0.01,
                 nesterov: bool = False):
        raise NotImplementedError(
            "TODO: Initialize SGDW\n"
            "Hint: Same as SGD but weight_decay is applied differently in step()"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement SGDW step\n"
            "Hint:\n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      # Apply decoupled weight decay FIRST\n"
            "      param *= (1 - self.defaults['lr'] * self.defaults['weight_decay'])\n"
            "      \n"
            "      # Then apply regular SGD+momentum update (no weight decay in grad)\n"
            "      # ... momentum update as in SGD ..."
        )


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
                 params: List[np.ndarray],
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
        raise NotImplementedError(
            "TODO: Initialize RMSprop\n"
            "Hint:\n"
            "  defaults = {'lr': lr, 'alpha': alpha, 'eps': eps,\n"
            "              'weight_decay': weight_decay, 'momentum': momentum, 'centered': centered}\n"
            "  super().__init__(params, defaults)\n"
            "  \n"
            "  for i, p in enumerate(params):\n"
            "      self.state[i] = {\n"
            "          'square_avg': np.zeros_like(p),  # v_t (EMA of g²)\n"
            "          'momentum_buffer': np.zeros_like(p) if momentum > 0 else None,\n"
            "          'grad_avg': np.zeros_like(p) if centered else None  # For centered RMSprop\n"
            "      }"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement RMSprop step\n"
            "Hint:\n"
            "  alpha = self.defaults['alpha']\n"
            "  eps = self.defaults['eps']\n"
            "  \n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      state = self.state[i]\n"
            "      \n"
            "      # Update squared gradient EMA\n"
            "      state['square_avg'] = alpha * state['square_avg'] + (1 - alpha) * (grad ** 2)\n"
            "      \n"
            "      # Compute denominator\n"
            "      if self.defaults['centered']:\n"
            "          state['grad_avg'] = alpha * state['grad_avg'] + (1 - alpha) * grad\n"
            "          avg = state['square_avg'] - state['grad_avg'] ** 2  # Variance\n"
            "      else:\n"
            "          avg = state['square_avg']\n"
            "      \n"
            "      # Compute update\n"
            "      update = grad / (np.sqrt(avg) + eps)\n"
            "      \n"
            "      # Optional momentum\n"
            "      if self.defaults['momentum'] > 0:\n"
            "          buf = state['momentum_buffer']\n"
            "          buf = self.defaults['momentum'] * buf + update\n"
            "          state['momentum_buffer'] = buf\n"
            "          update = buf\n"
            "      \n"
            "      param -= self.defaults['lr'] * update"
        )


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
                 params: List[np.ndarray],
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
        raise NotImplementedError(
            "TODO: Initialize Adagrad\n"
            "Hint:\n"
            "  for i, p in enumerate(params):\n"
            "      self.state[i] = {\n"
            "          'sum': np.full_like(p, initial_accumulator_value)  # G_t\n"
            "      }"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement Adagrad step\n"
            "Hint:\n"
            "  self._step_count += 1\n"
            "  lr = self.defaults['lr'] / (1 + (self._step_count - 1) * self.defaults['lr_decay'])\n"
            "  \n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      # Weight decay\n"
            "      if self.defaults['weight_decay'] != 0:\n"
            "          grad = grad + self.defaults['weight_decay'] * param\n"
            "      \n"
            "      # Accumulate squared gradient\n"
            "      self.state[i]['sum'] += grad ** 2\n"
            "      \n"
            "      # Update\n"
            "      param -= lr * grad / (np.sqrt(self.state[i]['sum']) + self.defaults['eps'])"
        )


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
                 params: List[np.ndarray],
                 lr: float = 1.0,  # Often set to 1.0 (Adadelta derives its own LR)
                 rho: float = 0.9,
                 eps: float = 1e-6,
                 weight_decay: float = 0.0):
        """
        Initialize Adadelta optimizer.

        Args:
            params: List of parameter arrays
            lr: Learning rate multiplier (typically 1.0)
            rho: Decay rate for running averages (similar to momentum)
            eps: Small constant for numerical stability
            weight_decay: L2 penalty
        """
        raise NotImplementedError(
            "TODO: Initialize Adadelta\n"
            "Hint:\n"
            "  for i, p in enumerate(params):\n"
            "      self.state[i] = {\n"
            "          'square_avg': np.zeros_like(p),       # E[g²]\n"
            "          'acc_delta': np.zeros_like(p),        # E[Δθ²]\n"
            "      }"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement Adadelta step\n"
            "Hint:\n"
            "  rho = self.defaults['rho']\n"
            "  eps = self.defaults['eps']\n"
            "  \n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      state = self.state[i]\n"
            "      \n"
            "      # Update running average of squared gradients\n"
            "      state['square_avg'] = rho * state['square_avg'] + (1 - rho) * grad ** 2\n"
            "      \n"
            "      # Compute update using RMS ratio\n"
            "      std = np.sqrt(state['acc_delta'] + eps)\n"
            "      delta = std / np.sqrt(state['square_avg'] + eps) * grad\n"
            "      \n"
            "      # Update running average of squared updates\n"
            "      state['acc_delta'] = rho * state['acc_delta'] + (1 - rho) * delta ** 2\n"
            "      \n"
            "      # Apply update\n"
            "      param -= self.defaults['lr'] * delta"
        )


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
                 params: List[np.ndarray],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False):
        """
        Initialize Adam optimizer.

        Args:
            params: List of parameter arrays
            lr: Learning rate
            betas: (β₁, β₂) coefficients for moment estimates
            eps: Term for numerical stability
            weight_decay: L2 penalty (NOT recommended, use AdamW instead)
            amsgrad: Use AMSGrad variant (maintains max of past v_t)
        """
        raise NotImplementedError(
            "TODO: Initialize Adam\n"
            "Hint:\n"
            "  defaults = {'lr': lr, 'betas': betas, 'eps': eps,\n"
            "              'weight_decay': weight_decay, 'amsgrad': amsgrad}\n"
            "  super().__init__(params, defaults)\n"
            "  \n"
            "  for i, p in enumerate(params):\n"
            "      self.state[i] = {\n"
            "          'exp_avg': np.zeros_like(p),       # m_t (first moment)\n"
            "          'exp_avg_sq': np.zeros_like(p),    # v_t (second moment)\n"
            "          'max_exp_avg_sq': np.zeros_like(p) if amsgrad else None  # For AMSGrad\n"
            "      }"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement Adam step\n"
            "Hint:\n"
            "  self._step_count += 1\n"
            "  beta1, beta2 = self.defaults['betas']\n"
            "  \n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      # Weight decay (L2, but prefer AdamW)\n"
            "      if self.defaults['weight_decay'] != 0:\n"
            "          grad = grad + self.defaults['weight_decay'] * param\n"
            "      \n"
            "      state = self.state[i]\n"
            "      \n"
            "      # Update biased first moment estimate\n"
            "      state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad\n"
            "      \n"
            "      # Update biased second moment estimate\n"
            "      state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad ** 2\n"
            "      \n"
            "      # Bias correction\n"
            "      bias_correction1 = 1 - beta1 ** self._step_count\n"
            "      bias_correction2 = 1 - beta2 ** self._step_count\n"
            "      m_hat = state['exp_avg'] / bias_correction1\n"
            "      v_hat = state['exp_avg_sq'] / bias_correction2\n"
            "      \n"
            "      # AMSGrad: use max of past v_hat\n"
            "      if self.defaults['amsgrad']:\n"
            "          state['max_exp_avg_sq'] = np.maximum(state['max_exp_avg_sq'], v_hat)\n"
            "          v_hat = state['max_exp_avg_sq']\n"
            "      \n"
            "      # Update\n"
            "      param -= self.defaults['lr'] * m_hat / (np.sqrt(v_hat) + self.defaults['eps'])"
        )

    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError("TODO: Implement get_state")

    def set_state(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError("TODO: Implement set_state")


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

    def __init__(self,
                 params: List[np.ndarray],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):  # Note: default 0.01, not 0
        """
        Initialize AdamW optimizer.

        Args:
            params: List of parameter arrays
            lr: Learning rate
            betas: (β₁, β₂) for moment estimates
            eps: Numerical stability term
            weight_decay: Decoupled weight decay coefficient
        """
        raise NotImplementedError(
            "TODO: Initialize AdamW\n"
            "Hint: Same structure as Adam, but weight_decay handled differently in step()"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement AdamW step\n"
            "Hint:\n"
            "  # Same as Adam but:\n"
            "  # 1. Do NOT add weight_decay to gradient\n"
            "  # 2. After computing Adam update, apply weight decay separately:\n"
            "  # param -= lr * weight_decay * param"
        )


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
                 params: List[np.ndarray],
                 lr: float = 0.002,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 momentum_decay: float = 0.004):
        raise NotImplementedError("TODO: Initialize NAdam")

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement NAdam step\n"
            "Hint: Like Adam but with Nesterov-style lookahead in momentum"
        )


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
                 params: List[np.ndarray],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0):
        raise NotImplementedError("TODO: Initialize RAdam")

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement RAdam step\n"
            "Hint:\n"
            "  # Compute SMA length\n"
            "  rho_inf = 2.0 / (1 - beta2) - 1\n"
            "  rho_t = rho_inf - 2 * t * (beta2 ** t) / (1 - beta2 ** t)\n"
            "  \n"
            "  if rho_t > 5:\n"
            "      # Compute rectification term and use Adam update\n"
            "      rect = np.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf /\n"
            "                     ((rho_inf - 4) * (rho_inf - 2) * rho_t))\n"
            "      # Adam update with rectification\n"
            "  else:\n"
            "      # SGD with momentum (no adaptive LR)"
        )


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
                 params: List[np.ndarray],
                 lr: Optional[float] = None,  # None = use relative step size
                 eps: Tuple[float, float] = (1e-30, 1e-3),
                 clip_threshold: float = 1.0,
                 decay_rate: float = -0.8,
                 beta1: Optional[float] = None,  # None = no first moment
                 weight_decay: float = 0.0,
                 scale_parameter: bool = True,
                 relative_step: bool = True,
                 warmup_init: bool = False):
        """
        Initialize Adafactor.

        Args:
            params: Parameter arrays
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
        raise NotImplementedError(
            "TODO: Initialize Adafactor\n"
            "Hint:\n"
            "  # For each 2D+ parameter, store row and column factors\n"
            "  # For 1D parameters, store full second moment\n"
            "  for i, p in enumerate(params):\n"
            "      if p.ndim >= 2:\n"
            "          self.state[i] = {\n"
            "              'exp_avg_sq_row': np.zeros(p.shape[:-1]),\n"
            "              'exp_avg_sq_col': np.zeros(p.shape[:-2] + (p.shape[-1],)),\n"
            "          }\n"
            "      else:\n"
            "          self.state[i] = {'exp_avg_sq': np.zeros_like(p)}"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement Adafactor step\n"
            "Hint: See paper for factorized second moment computation"
        )


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
                 params: List[np.ndarray],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0.0,
                 adam: bool = False):  # If True, becomes ADAM (no layer-wise scaling)
        """
        Initialize LAMB optimizer.

        Args:
            params: Parameter arrays
            lr: Learning rate
            betas: (β₁, β₂) for moment estimates
            eps: Numerical stability term
            weight_decay: Decoupled weight decay
            adam: If True, disable layer-wise scaling (becomes AdamW)
        """
        raise NotImplementedError(
            "TODO: Initialize LAMB\n"
            "Hint: Similar to AdamW, but add trust ratio computation in step()"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement LAMB step\n"
            "Hint:\n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      # Compute Adam update (with decoupled weight decay)\n"
            "      # ... Adam moment updates ...\n"
            "      update = m_hat / (np.sqrt(v_hat) + eps) + weight_decay * param\n"
            "      \n"
            "      # Compute trust ratio (layer-wise scaling)\n"
            "      param_norm = np.linalg.norm(param)\n"
            "      update_norm = np.linalg.norm(update)\n"
            "      \n"
            "      if param_norm > 0 and update_norm > 0:\n"
            "          trust_ratio = param_norm / update_norm\n"
            "      else:\n"
            "          trust_ratio = 1.0\n"
            "      \n"
            "      # Apply scaled update\n"
            "      param -= lr * trust_ratio * update"
        )


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
                 params: List[np.ndarray],
                 lr: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0,
                 trust_coefficient: float = 0.001,
                 eps: float = 1e-8):
        """
        Initialize LARS optimizer.

        Args:
            params: Parameter arrays
            lr: Global learning rate
            momentum: Momentum factor
            weight_decay: L2 penalty
            trust_coefficient: Trust coefficient for layer-wise scaling
            eps: Numerical stability term
        """
        raise NotImplementedError("TODO: Initialize LARS")

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement LARS step\n"
            "Hint:\n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      # Compute local learning rate\n"
            "      param_norm = np.linalg.norm(param)\n"
            "      grad_norm = np.linalg.norm(grad)\n"
            "      \n"
            "      local_lr = trust_coefficient * param_norm / (grad_norm + weight_decay * param_norm + eps)\n"
            "      \n"
            "      # Momentum update with local LR\n"
            "      # ..."
        )


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
                 params: List[np.ndarray],
                 lr: float = 1e-4,
                 betas: Tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        """
        Initialize Lion optimizer.

        Args:
            params: Parameter arrays
            lr: Learning rate (use ~10x smaller than AdamW)
            betas: (β₁, β₂) for momentum update and sign computation
            weight_decay: Decoupled weight decay (use larger than AdamW, e.g., 0.1)
        """
        raise NotImplementedError(
            "TODO: Initialize Lion\n"
            "Hint:\n"
            "  for i, p in enumerate(params):\n"
            "      self.state[i] = {'exp_avg': np.zeros_like(p)}  # Only momentum, no v!"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement Lion step\n"
            "Hint:\n"
            "  beta1, beta2 = self.defaults['betas']\n"
            "  \n"
            "  for i, (param, grad) in enumerate(zip(self.params, gradients)):\n"
            "      m = self.state[i]['exp_avg']\n"
            "      \n"
            "      # Compute sign of interpolated momentum\n"
            "      update = np.sign(beta2 * m + (1 - beta2) * grad)\n"
            "      \n"
            "      # Apply update with weight decay\n"
            "      param -= self.defaults['lr'] * (update + self.defaults['weight_decay'] * param)\n"
            "      \n"
            "      # Update momentum (for next iteration)\n"
            "      self.state[i]['exp_avg'] = beta1 * m + (1 - beta1) * grad"
        )


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
                 params: List[np.ndarray],
                 lr: float = 0.02,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 backend: str = 'newtonschulz',  # or 'svd'
                 backend_steps: int = 5):
        """
        Initialize Muon optimizer.

        Args:
            params: Parameter arrays
            lr: Learning rate
            momentum: Momentum factor
            nesterov: Use Nesterov momentum
            backend: Method for orthogonalization ('newtonschulz' or 'svd')
            backend_steps: Number of Newton-Schulz iterations
        """
        raise NotImplementedError(
            "TODO: Initialize Muon\n"
            "Hint:\n"
            "  for i, p in enumerate(params):\n"
            "      self.state[i] = {'momentum_buffer': np.zeros_like(p)}"
        )

    def step(self, gradients: List[np.ndarray]) -> None:
        raise NotImplementedError(
            "TODO: Implement Muon step\n"
            "Hint:\n"
            "  # The key is orthogonalization\n"
            "  # For matrices (weight matrices), use Newton-Schulz iteration:\n"
            "  def newton_schulz(G, steps=5):\n"
            "      '''Compute G @ (G.T @ G)^{-1/2} via Newton-Schulz'''\n"
            "      a, b, c = (3.4445, -4.7750, 2.0315)\n"
            "      X = G / np.linalg.norm(G)\n"
            "      for _ in range(steps):\n"
            "          A = X @ X.T\n"
            "          B = b * A + c * A @ A\n"
            "          X = a * X + B @ X\n"
            "      return X\n"
            "  \n"
            "  # Apply orthogonalized update"
        )


# =============================================================================
# Functional Interfaces
# =============================================================================

def sgd_step(param: np.ndarray, grad: np.ndarray, velocity: np.ndarray,
             lr: float, momentum: float = 0.0, weight_decay: float = 0.0,
             dampening: float = 0.0, nesterov: bool = False
             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Functional SGD update for a single parameter.

    Args:
        param: Parameter array
        grad: Gradient array
        velocity: Momentum buffer
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: L2 penalty
        dampening: Dampening for momentum
        nesterov: Use Nesterov momentum

    Returns:
        Tuple of (updated_param, new_velocity)
    """
    raise NotImplementedError("TODO: Implement functional SGD")


def adam_step(param: np.ndarray, grad: np.ndarray,
              exp_avg: np.ndarray, exp_avg_sq: np.ndarray, step: int,
              lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
              eps: float = 1e-8, weight_decay: float = 0.0
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Functional Adam update for a single parameter.

    Returns:
        Tuple of (updated_param, new_exp_avg, new_exp_avg_sq)
    """
    raise NotImplementedError("TODO: Implement functional Adam")


def adamw_step(param: np.ndarray, grad: np.ndarray,
               exp_avg: np.ndarray, exp_avg_sq: np.ndarray, step: int,
               lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
               eps: float = 1e-8, weight_decay: float = 0.01
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Functional AdamW update for a single parameter.

    Returns:
        Tuple of (updated_param, new_exp_avg, new_exp_avg_sq)
    """
    raise NotImplementedError("TODO: Implement functional AdamW")
