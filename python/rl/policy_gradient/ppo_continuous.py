"""
PPO for Continuous Action Spaces

Theory:
    For continuous control tasks, actions are sampled from a Gaussian distribution
    parameterized by a mean (μ) and standard deviation (σ) output by the policy
    network. The same PPO clipped objective is used, but with continuous actions.

Mathematical Framework:
    Policy: π(a|s) = N(μ(s), σ(s))  or  N(μ(s), σ) with fixed/learned σ

    Log probability: log π(a|s) = -0.5 * [(a - μ)² / σ² + log(2πσ²)]

    Same clipped objective:
    L = -E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

    Action bounds: Often use tanh squashing or clipping for bounded actions.

Implementation Variants:
    1. Diagonal Gaussian: Independent σ for each action dimension
    2. State-dependent σ: σ(s) = network output
    3. Learned σ: σ as learnable parameter (not state-dependent)
    4. Fixed σ: σ as hyperparameter (simplest)

References:
    - Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
      https://arxiv.org/abs/1708.02747
    - Haarnoja et al. (2018): SAC uses similar Gaussian policies
      https://arxiv.org/abs/1801.01290
"""

# Implementation Status: NOT STARTED
# Complexity: Medium
# Prerequisites: ppo.py

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class PPOContinuous:
    """
    PPO agent for continuous action spaces using Gaussian policy.

    The policy outputs mean and log standard deviation for each action
    dimension. Actions are sampled from the resulting Gaussian.

    Example:
        >>> agent = PPOContinuous(
        ...     state_dim=17,  # e.g., HalfCheetah
        ...     action_dim=6,
        ...     action_low=-1.0,
        ...     action_high=1.0
        ... )
        >>> for episode in range(1000):
        ...     trajectories = collect_trajectories(env, agent)
        ...     metrics = agent.update(trajectories)

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        action_low: Lower bound for actions (scalar or array)
        action_high: Upper bound for actions (scalar or array)
        hidden_dims: Hidden layer sizes for networks
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        lam: GAE lambda parameter
        clip_epsilon: PPO clipping parameter
        epochs: Number of epochs per update
        batch_size: Minibatch size
        entropy_coef: Entropy bonus coefficient
        init_log_std: Initial log standard deviation
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        hidden_dims: List[int] = [64, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        init_log_std: float = 0.0
    ):
        """Initialize PPO agent for continuous actions."""
        raise NotImplementedError(
            "TODO: Initialize continuous PPO agent\n"
            "Hint:\n"
            "  self.state_dim = state_dim\n"
            "  self.action_dim = action_dim\n"
            "  self.action_low = action_low\n"
            "  self.action_high = action_high\n"
            "  self.gamma = gamma\n"
            "  self.lam = lam\n"
            "  self.clip_epsilon = clip_epsilon\n"
            "  self.epochs = epochs\n"
            "  self.batch_size = batch_size\n"
            "  self.entropy_coef = entropy_coef\n"
            "  \n"
            "  # Policy network outputs mean\n"
            "  self.policy_mean = build_mlp(state_dim, action_dim, hidden_dims, output_activation='tanh')\n"
            "  \n"
            "  # Log standard deviation (learnable parameter)\n"
            "  self.log_std = np.ones(action_dim) * init_log_std\n"
            "  \n"
            "  # Value network\n"
            "  self.value_fn = build_mlp(state_dim, 1, hidden_dims)"
        )

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select continuous action using Gaussian policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            action: Selected action (clipped to bounds)
            mean: Mean of action distribution
            log_prob: Log probability of action
        """
        raise NotImplementedError(
            "TODO: Select continuous action\n"
            "Hint:\n"
            "  # Get mean from policy network\n"
            "  mean = self.policy_mean.forward(state)\n"
            "  std = np.exp(self.log_std)\n"
            "  \n"
            "  if training:\n"
            "      # Sample from Gaussian\n"
            "      action = mean + std * np.random.randn(self.action_dim)\n"
            "  else:\n"
            "      action = mean\n"
            "  \n"
            "  # Clip to bounds\n"
            "  action = np.clip(action, self.action_low, self.action_high)\n"
            "  \n"
            "  # Compute log probability\n"
            "  log_prob = self.compute_log_prob(action, mean, std)\n"
            "  \n"
            "  return action, mean, log_prob"
        )

    def compute_log_prob(
        self,
        action: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray
    ) -> float:
        """
        Compute log probability of action under Gaussian.

        log π(a|s) = -0.5 * [Σ (a_i - μ_i)² / σ_i² + Σ log(2πσ_i²)]

        Args:
            action: Action vector
            mean: Mean of Gaussian
            std: Standard deviation of Gaussian

        Returns:
            log_prob: Log probability (scalar)
        """
        raise NotImplementedError(
            "TODO: Compute Gaussian log probability\n"
            "Hint:\n"
            "  var = std ** 2\n"
            "  log_prob = -0.5 * np.sum(\n"
            "      (action - mean) ** 2 / var +\n"
            "      np.log(2 * np.pi * var)\n"
            "  )\n"
            "  return log_prob"
        )

    def compute_entropy(self, std: np.ndarray) -> float:
        """
        Compute entropy of Gaussian policy.

        H(π) = 0.5 * Σ log(2πeσ_i²) = 0.5 * d * (1 + log(2π)) + Σ log(σ_i)

        Args:
            std: Standard deviation

        Returns:
            entropy: Entropy of distribution
        """
        raise NotImplementedError(
            "TODO: Compute Gaussian entropy\n"
            "Hint:\n"
            "  entropy = 0.5 * self.action_dim * (1 + np.log(2 * np.pi))\n"
            "  entropy += np.sum(np.log(std))\n"
            "  return entropy"
        )

    def compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: Rewards array
            values: Value estimates
            dones: Done flags

        Returns:
            advantages: GAE advantages
            returns: Target returns
        """
        raise NotImplementedError(
            "TODO: Implement GAE (same as discrete PPO)\n"
            "Hint: Same implementation as PPO.compute_advantages"
        )

    def compute_clipped_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute PPO clipped objective for continuous actions.

        Args:
            states: Batch of states
            actions: Batch of continuous actions
            old_log_probs: Log probs from old policy
            advantages: Computed advantages

        Returns:
            loss: Total loss
            info: Metrics dictionary
        """
        raise NotImplementedError(
            "TODO: Implement clipped loss for continuous actions\n"
            "Hint:\n"
            "  # Get means from policy\n"
            "  means = self.policy_mean.forward(states)\n"
            "  std = np.exp(self.log_std)\n"
            "  \n"
            "  # Compute new log probs\n"
            "  new_log_probs = np.array([\n"
            "      self.compute_log_prob(a, m, std)\n"
            "      for a, m in zip(actions, means)\n"
            "  ])\n"
            "  \n"
            "  # Ratio and clipped objective\n"
            "  ratio = np.exp(new_log_probs - old_log_probs)\n"
            "  surr1 = ratio * advantages\n"
            "  surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages\n"
            "  policy_loss = -np.mean(np.minimum(surr1, surr2))\n"
            "  \n"
            "  # Entropy bonus\n"
            "  entropy = self.compute_entropy(std)\n"
            "  \n"
            "  loss = policy_loss - self.entropy_coef * entropy\n"
            "  return loss, {'policy_loss': policy_loss, 'entropy': entropy}"
        )

    def update(self, trajectories: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update policy and value function.

        Args:
            trajectories: Dictionary with trajectory data

        Returns:
            info: Training metrics
        """
        raise NotImplementedError(
            "TODO: Implement PPO update for continuous actions\n"
            "Hint: Similar to discrete PPO but with continuous actions"
        )


# Alias for common naming
PPOContinuousAction = PPOContinuous
