"""
Utility Functions for Reinforcement Learning

This module provides common utility functions used across RL algorithms:
- Discount factor computations
- Polyak (exponential moving) averaging
- Variance and performance metrics
- Numerical stability helpers

IMPORTANCE:
These utilities may seem simple but are critical for:
- Training stability
- Numerical accuracy
- Algorithm convergence
- Fair comparisons between methods

REFERENCES:
    - Sutton & Barto (2018)
    - Deep RL best practices
    - OpenAI Baselines
    - Stable Baselines implementations
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from python.nn_core import Module


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted cumulative sum.

    Given array [x_0, x_1, x_2, ...], compute:
    [x_0 + γ*x_1 + γ²*x_2 + ...,
     x_1 + γ*x_2 + γ²*x_3 + ...,
     x_2 + γ*x_3 + ...,
     ...]

    MATH:
        out[t] = ∑_{i=0}^{n-t-1} γ^i * x_{t+i}

    APPLICATIONS:
    1. Computing returns from rewards:
       G_t = ∑_{i=0}^{T-1-t} γ^i * r_{t+i}

    2. Computing TD targets for GAE/n-step

    3. Any accumulated future quantity

    EFFICIENT ALGORITHM (SciPy-inspired):
        Uses backward pass: y[t] = x[t] + γ * y[t+1]

    Args:
        x: 1D array [T]
        gamma: Discount factor (0 to 1, typically 0.99)

    Returns:
        out: Discounted cumulative sum [T]

    Example:
        >>> rewards = np.array([1.0, 2.0, 3.0, 4.0])
        >>> gamma = 0.9
        >>> returns = discount_cumsum(rewards, gamma)
        >>> # returns[0] = 1 + 0.9*2 + 0.9^2*3 + 0.9^3*4
        >>> #            = 1 + 1.8 + 2.43 + 2.916 = 8.146
    """
    raise NotImplementedError(
        "Hint: Initialize output array of same shape as x. "
        "Iterate backward: out[t] = x[t] + gamma * out[t+1]. "
        "Or use scipy.signal.lfilter if available."
    )


def polyak_averaging(source: Module,
                    target: Module,
                    tau: float = 0.995) -> None:
    """
    Polyak (exponential moving) averaging of network parameters.

    Updates target network parameters toward source network:
    θ_target ← τ * θ_target + (1 - τ) * θ_source

    MOTIVATION:
    - Stabilizes learning by using slowly-changing targets
    - Used in DQN, DDPG, TD3, SAC
    - Smoother updates than periodic hard copies

    MATH:
        θ'_new = τ * θ'_old + (1 - τ) * θ

        Exponential moving average toward source network
        τ close to 1: slow updates (more stable)
        τ close to 0: fast updates (track source closely)

    TYPICAL VALUES:
        - DQN: hard copy every N steps (tau=1 for N steps, tau=0 otherwise)
        - DDPG/TD3/SAC: tau=0.995 or 0.999 (very smooth)

    PROPERTIES:
        - Moving average: ∑(1-τ)τ^i = 1 (normalized)
        - Exponential decay: weights decay exponentially into past
        - Effective averaging window ≈ 1/(1-tau) steps

    Args:
        source: Source network (recently trained)
        target: Target network (to be updated)
        tau: Polyak averaging coefficient (default: 0.995)
             tau=1.0: target unchanged
             tau=0.0: target = source (hard copy)

    Example:
        >>> model = PolicyNetwork(...)
        >>> target_model = PolicyNetwork(...)
        >>> # In training loop:
        >>> ... training ...
        >>> polyak_averaging(model, target_model, tau=0.995)

    ALGORITHM:
        Iterate through paired parameters and update target toward source
    """
    raise NotImplementedError(
        "Hint: Iterate through paired parameters of source and target networks. "
        "Update each target parameter with exponential moving average"
    )


def hard_copy(source: Module, target: Module) -> None:
    """
    Hard copy: make target network identical to source network.

    θ_target ← θ_source

    MOTIVATION:
    - Periodic network updates (e.g., DQN every N steps)
    - Initialize target networks
    - Create evaluation network copies

    Used in:
    - DQN (copy every N steps)
    - Double DQN (separate target network)
    - Prioritized Experience Replay

    Args:
        source: Source network
        target: Target network to update

    Example:
        >>> main_q = DQNNetwork(...)
        >>> target_q = DQNNetwork(...)
        >>> # Every N training steps:
        >>> hard_copy(main_q, target_q)
    """
    raise NotImplementedError(
        "Hint: Copy all parameters from source to target network"
    )


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance ratio for value function.

    DEFINITION:
        EV = 1 - Var(y_true - y_pred) / Var(y_true)

    INTERPRETATION:
        - EV = 1.0: Perfect prediction (y_pred = y_true everywhere)
        - EV = 0.0: No better than predicting mean of y_true
        - EV < 0.0: Worse than mean predictor (something is wrong!)

    MATH:
        Residual variance: σ²_residual = mean((y_true - y_pred)²) - mean(y_true - y_pred)²
        Target variance: σ²_target = mean(y_true²) - mean(y_true)²
        EV = 1 - σ²_residual / σ²_target

        Simplified: EV = 1 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8)

    USAGE IN RL:
        Monitor value function quality during training:
        >>> returns = ... # target values
        >>> values = value_net.predict(states)
        >>> ev = explained_variance(returns, values)
        >>> logger.log({"explained_variance": ev})

    INTERPRETATION GUIDE:
        - EV > 0.95: Excellent value function
        - EV > 0.90: Good fit
        - EV > 0.50: Reasonable fit
        - EV > 0.00: Better than nothing
        - EV < 0.00: Value function is broken!

    Args:
        y_pred: Predicted values [batch_size]
        y_true: True/target values [batch_size]

    Returns:
        ev: Explained variance in range (-∞, 1.0]
            1.0 = perfect prediction
            0.0 = predicting mean
            < 0 = worse than mean

    Example:
        >>> returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> values = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> ev = explained_variance(returns, values)
        >>> print(f"Explained variance: {ev:.3f}")
    """
    raise NotImplementedError(
        "Hint: Compute variance of residuals (y_true - y_pred), "
        "variance of y_true, return: 1 - (var_residuals / (var_true + eps))"
    )


def compute_gae_returns(rewards: np.ndarray,
                       values: np.ndarray,
                       next_values: np.ndarray,
                       gamma: float = 0.99,
                       lambda_coeff: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and returns efficiently.

    SHORTHAND for advantage.GeneralizedAdvantageEstimation.compute_advantages()

    Included here for convenience - computes both advantages and returns
    in a single efficient pass.

    Args:
        rewards: [T]
        values: [T]
        next_values: [T+1] (includes bootstrap value)
        gamma: Discount factor
        lambda_coeff: GAE λ parameter

    Returns:
        advantages: [T]
        returns: [T]
    """
    raise NotImplementedError(
        "Hint: This is identical to GAE computation. "
        "See advantage.py for detailed explanation."
    )


def normalize(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize array to zero mean and unit variance.

    FORMULA:
        x_norm = (x - mean) / (std + ε)

    MOTIVATION:
    - Stabilizes learning
    - Makes learning rate more robust
    - Helps with gradient flow

    Args:
        x: Array to normalize
        axis: Axis for mean/std (None = normalize entire array)

    Returns:
        x_normalized: Zero mean, unit variance
    """
    raise NotImplementedError(
        "Hint: return (x - x.mean(axis=axis, keepdims=True)) / "
        "(x.std(axis=axis, keepdims=True) + 1e-8)"
    )


def compute_td_error(rewards: np.ndarray,
                    values: np.ndarray,
                    next_values: np.ndarray,
                    dones: np.ndarray,
                    gamma: float = 0.99) -> np.ndarray:
    """
    Compute TD-error (temporal difference error).

    MATH:
        δ = r + γ * V(s') * (1 - done) - V(s)

    Used for:
    - Prioritized experience replay (priority = |δ|)
    - Value function training
    - Diagnostics

    Args:
        rewards: [batch_size, 1]
        values: [batch_size, 1] V(s)
        next_values: [batch_size, 1] V(s')
        dones: [batch_size, 1] episode termination flags
        gamma: Discount factor

    Returns:
        td_error: [batch_size, 1]

    Example:
        >>> td_error = compute_td_error(rewards, values, next_values, dones)
        >>> priority = np.abs(td_error)  # For PER
    """
    raise NotImplementedError(
        "Hint: return rewards + gamma * next_values * (1 - dones) - values"
    )


def clipped_value_loss(values: np.ndarray,
                      targets: np.ndarray,
                      old_values: np.ndarray,
                      clip_ratio: float = 0.2) -> float:
    """
    Clipped value loss for PPO-style training.

    Prevents large value function changes in a single update.

    MATH:
        V_clipped = old_V + clip(V - old_V, -ε, ε)
        L = mean(max((V - target)², (V_clipped - target)²))

    where ε = clip_ratio * |V_range|

    Args:
        values: Current value predictions [batch_size]
        targets: Target values [batch_size]
        old_values: Values from previous policy [batch_size]
        clip_ratio: Clipping range as fraction (default: 0.2 = ±20%)

    Returns:
        loss: Scalar clipped value loss
    """
    raise NotImplementedError(
        "Hint: Compute value difference, clip to [-clip_ratio, clip_ratio] * old_value. "
        "Compute MSE for both clipped and unclipped, take max."
    )


def compute_gae_batch(rewards: np.ndarray,
                     values: np.ndarray,
                     dones: np.ndarray,
                     gamma: float = 0.99,
                     lambda_coeff: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and returns for a batch.

    Numpy version for efficient batch computation.

    Args:
        rewards: [batch_size, T]
        values: [batch_size, T+1]
        dones: [batch_size, T]
        gamma: Discount factor
        lambda_coeff: GAE λ

    Returns:
        advantages: [batch_size, T]
        returns: [batch_size, T]
    """
    raise NotImplementedError(
        "Hint: Same as numpy GAE computation. "
        "Compute deltas, backward accumulate with (gamma * lambda_coeff)"
    )


def compute_entropy(probs: np.ndarray) -> float:
    """
    Compute entropy of probability distribution.

    MATH:
        H = -∑_a p(a) * log p(a)

    Used for:
    - Entropy regularization (encourage exploration)
    - Policy diagnostics
    - Early stopping

    Args:
        probs: Probability distribution [batch_size, num_actions] or [num_actions]

    Returns:
        entropy: Scalar entropy
    """
    raise NotImplementedError(
        "Hint: -np.sum(probs * np.log(probs + 1e-8))"
    )


def compute_returns_from_gae(advantages: np.ndarray,
                            values: np.ndarray) -> np.ndarray:
    """
    Recover returns (TD targets) from advantages and values.

    FORMULA:
        G_t = Â_t + V(s_t)

    This recovers the value targets used for value function training
    from pre-computed advantages and baseline values.

    Args:
        advantages: [T]
        values: [T] V(s_t)

    Returns:
        returns: [T] G_t = Â_t + V(s_t)
    """
    raise NotImplementedError(
        "Hint: Simply return advantages + values"
    )


def log_prob_from_distribution(distribution_params: np.ndarray,
                               actions: np.ndarray,
                               action_type: str = "continuous") -> np.ndarray:
    """
    Compute log probability from distribution parameters and actions.

    Args:
        distribution_params: [batch_size, action_dim or 2*action_dim]
                           For continuous: [mean; log_std]
        actions: [batch_size, action_dim]
        action_type: "discrete" or "continuous"

    Returns:
        log_probs: [batch_size] log π(a|s)
    """
    raise NotImplementedError(
        "Hint: For discrete, apply softmax to get probs, compute log. "
        "For continuous, extract mean/std, use Gaussian PDF formula."
    )


def action_from_distribution(distribution_params: np.ndarray,
                            action_type: str = "continuous",
                            deterministic: bool = False) -> np.ndarray:
    """
    Sample action from distribution parameters.

    Args:
        distribution_params: [batch_size, action_dim or 2*action_dim]
        action_type: "discrete" or "continuous"
        deterministic: If True, return mean/mode (no sampling)

    Returns:
        actions: [batch_size, action_dim]
    """
    raise NotImplementedError(
        "Hint: For discrete deterministic, use argmax. "
        "For discrete stochastic, use categorical sampling. "
        "For continuous deterministic, return mean. "
        "For continuous stochastic, sample from N(mean, std)."
    )
