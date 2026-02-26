"""
Proximal Policy Optimization (PPO)

Theory:
    PPO is a policy gradient algorithm that uses a clipped surrogate objective
    to constrain policy updates, achieving similar stability to TRPO but with
    simpler implementation. Instead of a hard KL constraint, PPO clips the
    probability ratio to prevent too large updates.

Mathematical Framework:
    L^{CLIP}(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

    Where:
    - r(θ) = π_θ(a|s) / π_θ_old(a|s)  (probability ratio)
    - A = advantage estimate
    - ε = clipping parameter (typically 0.1-0.2)

    The clipping removes incentive for moving r(θ) outside [1-ε, 1+ε].

Algorithm (PPO-Clip):
    1. Collect trajectories using current policy
    2. Compute advantages (usually GAE)
    3. For each epoch:
        a. Sample minibatches from collected data
        b. Compute clipped surrogate loss
        c. Update policy using gradient descent
    4. Update value function

Key Innovations:
    - Clipped objective: Simple alternative to TRPO's KL constraint
    - Multiple epochs: Reuse collected data for multiple updates
    - Minibatch updates: More efficient than full-batch TRPO

References:
    - Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
      https://arxiv.org/abs/1708.02747
    - OpenAI Spinning Up: PPO
      https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

# Implementation Status: NOT STARTED
# Complexity: Medium
# Prerequisites: reinforce.py, vpg.py

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class PPO:
    """
    Proximal Policy Optimization agent for discrete action spaces.

    PPO uses a clipped surrogate objective for stable policy updates
    without the computational complexity of TRPO.

    Example:
        >>> agent = PPO(
        ...     state_dim=4,
        ...     action_dim=2,
        ...     hidden_dims=[64, 64],
        ...     clip_epsilon=0.2
        ... )
        >>> for episode in range(1000):
        ...     trajectories = collect_trajectories(env, agent)
        ...     metrics = agent.update(trajectories)

    Args:
        state_dim: Dimension of state space
        action_dim: Number of discrete actions
        hidden_dims: Hidden layer sizes for networks
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        lam: GAE lambda parameter
        clip_epsilon: PPO clipping parameter
        epochs: Number of epochs per update
        batch_size: Minibatch size
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5
    ):
        """Initialize PPO agent."""
        raise NotImplementedError(
            "TODO: Initialize PPO agent\n"
            "Hint:\n"
            "  self.state_dim = state_dim\n"
            "  self.action_dim = action_dim\n"
            "  self.gamma = gamma\n"
            "  self.lam = lam\n"
            "  self.clip_epsilon = clip_epsilon\n"
            "  self.epochs = epochs\n"
            "  self.batch_size = batch_size\n"
            "  self.entropy_coef = entropy_coef\n"
            "  self.value_coef = value_coef\n"
            "  \n"
            "  # Policy network (actor)\n"
            "  self.policy = build_mlp(state_dim, action_dim, hidden_dims)\n"
            "  \n"
            "  # Value network (critic)\n"
            "  self.value_fn = build_mlp(state_dim, 1, hidden_dims)\n"
            "  \n"
            "  # Optimizers\n"
            "  self.policy_optimizer = Adam(lr=learning_rate)\n"
            "  self.value_optimizer = Adam(lr=learning_rate)"
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """
        Select action using current policy.

        Args:
            state: Current state
            training: Whether in training mode (use stochastic policy)

        Returns:
            action: Selected action
            log_prob: Log probability of action
        """
        raise NotImplementedError(
            "TODO: Select action from policy\n"
            "Hint:\n"
            "  logits = self.policy.forward(state)\n"
            "  probs = softmax(logits)\n"
            "  \n"
            "  if training:\n"
            "      action = np.random.choice(self.action_dim, p=probs)\n"
            "  else:\n"
            "      action = np.argmax(probs)\n"
            "  \n"
            "  log_prob = np.log(probs[action] + 1e-8)\n"
            "  return action, log_prob"
        )

    def get_action_probs(self, states: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for batch of states.

        Args:
            states: Batch of states (batch_size, state_dim)

        Returns:
            probs: Action probabilities (batch_size, action_dim)
        """
        raise NotImplementedError(
            "TODO: Get action probabilities\n"
            "Hint:\n"
            "  logits = self.policy.forward(states)\n"
            "  probs = softmax(logits, axis=-1)\n"
            "  return probs"
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
            values: Value estimates array
            dones: Done flags array

        Returns:
            advantages: GAE advantages
            returns: Returns (for value function training)
        """
        raise NotImplementedError(
            "TODO: Implement GAE computation\n"
            "Hint:\n"
            "  advantages = np.zeros_like(rewards)\n"
            "  gae = 0\n"
            "  \n"
            "  for t in reversed(range(len(rewards))):\n"
            "      if t == len(rewards) - 1:\n"
            "          next_value = 0\n"
            "      else:\n"
            "          next_value = values[t + 1]\n"
            "      \n"
            "      delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]\n"
            "      gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae\n"
            "      advantages[t] = gae\n"
            "  \n"
            "  returns = advantages + values\n"
            "  return advantages, returns"
        )

    def compute_clipped_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute PPO clipped surrogate loss.

        L = -E[min(r*A, clip(r, 1-ε, 1+ε)*A)]

        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Log probs from old policy
            advantages: Computed advantages

        Returns:
            loss: Clipped surrogate loss
            info: Dictionary with metrics
        """
        raise NotImplementedError(
            "TODO: Implement clipped surrogate loss\n"
            "Hint:\n"
            "  # Get new log probs\n"
            "  probs = self.get_action_probs(states)\n"
            "  new_log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)\n"
            "  \n"
            "  # Compute ratio\n"
            "  ratio = np.exp(new_log_probs - old_log_probs)\n"
            "  \n"
            "  # Clipped surrogate objective\n"
            "  surr1 = ratio * advantages\n"
            "  surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages\n"
            "  policy_loss = -np.mean(np.minimum(surr1, surr2))\n"
            "  \n"
            "  # Entropy bonus\n"
            "  entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1).mean()\n"
            "  \n"
            "  loss = policy_loss - self.entropy_coef * entropy\n"
            "  return loss, {'policy_loss': policy_loss, 'entropy': entropy}"
        )

    def compute_value_loss(
        self,
        states: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Compute value function loss (MSE).

        Args:
            states: Batch of states
            returns: Target returns

        Returns:
            loss: Value function loss
        """
        raise NotImplementedError(
            "TODO: Compute value loss\n"
            "Hint:\n"
            "  values = self.value_fn.forward(states).squeeze()\n"
            "  value_loss = np.mean((values - returns) ** 2)\n"
            "  return value_loss"
        )

    def update(self, trajectories: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update policy and value function using collected data.

        Args:
            trajectories: Dictionary with states, actions, rewards, etc.

        Returns:
            info: Dictionary with training metrics
        """
        raise NotImplementedError(
            "TODO: Implement PPO update\n"
            "Hint:\n"
            "  states = trajectories['states']\n"
            "  actions = trajectories['actions']\n"
            "  rewards = trajectories['rewards']\n"
            "  dones = trajectories['dones']\n"
            "  old_log_probs = trajectories['log_probs']\n"
            "  \n"
            "  # Compute values and advantages\n"
            "  values = self.value_fn.forward(states).squeeze()\n"
            "  advantages, returns = self.compute_advantages(rewards, values, dones)\n"
            "  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)\n"
            "  \n"
            "  # Multiple epochs of updates\n"
            "  total_metrics = defaultdict(list)\n"
            "  for epoch in range(self.epochs):\n"
            "      # Sample minibatches\n"
            "      indices = np.random.permutation(len(states))\n"
            "      for start in range(0, len(states), self.batch_size):\n"
            "          batch_idx = indices[start:start + self.batch_size]\n"
            "          \n"
            "          # Policy update\n"
            "          policy_loss, info = self.compute_clipped_loss(\n"
            "              states[batch_idx], actions[batch_idx],\n"
            "              old_log_probs[batch_idx], advantages[batch_idx]\n"
            "          )\n"
            "          self.policy.backward(policy_loss)\n"
            "          self.policy_optimizer.step(self.policy)\n"
            "          \n"
            "          # Value update\n"
            "          value_loss = self.compute_value_loss(states[batch_idx], returns[batch_idx])\n"
            "          self.value_fn.backward(value_loss)\n"
            "          self.value_optimizer.step(self.value_fn)\n"
            "  \n"
            "  return {k: np.mean(v) for k, v in total_metrics.items()}"
        )


# Alias for full name
ProximalPolicyOptimization = PPO
