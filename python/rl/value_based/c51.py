"""
Categorical DQN (C51) - Distributional Reinforcement Learning

Theory:
    Instead of learning the expected Q-value E[Z(s,a)], C51 learns the full
    distribution of returns Z(s,a). This provides more information for learning
    and leads to better performance, especially in stochastic environments.

Mathematical Framework:
    Traditional Q-learning: Q(s,a) = E[Z(s,a)]
    Distributional RL: Learn the distribution Z(s,a)

    C51 represents Z(s,a) as a categorical distribution over N atoms:
    - Atoms: z_i = V_min + i * Δz, where Δz = (V_max - V_min) / (N-1)
    - Distribution: P(Z(s,a) = z_i) = p_i(s,a)

    Distributional Bellman Equation:
    Z(s,a) =_D R + γZ(s', a')

    C51 Update:
    1. Compute target distribution: T_Z(s,a) = r + γZ(s', argmax_a' Q(s',a'))
    2. Project onto atoms: Project shifted distribution back to support
    3. Minimize KL divergence between projected target and current

References:
    - Bellemare et al. (2017): "A Distributional Perspective on RL"
      https://arxiv.org/abs/1707.06887
    - Dabney et al. (2018): "Implicit Quantile Networks"
      https://arxiv.org/abs/1806.06923
"""

# Implementation Status: NOT STARTED
# Complexity: Hard
# Prerequisites: dqn.py

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class DistributionBuffer:
    """
    Replay buffer that stores full return distributions.

    For C51, we need to store both states and the target distributions
    for computing the cross-entropy loss.

    Args:
        capacity: Maximum buffer size
        state_dim: Dimension of state space
        n_atoms: Number of atoms in distribution
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        n_atoms: int = 51
    ):
        """Initialize distribution buffer."""
        raise NotImplementedError(
            "TODO: Initialize distribution buffer\n"
            "Hint:\n"
            "  self.capacity = capacity\n"
            "  self.position = 0\n"
            "  self.size = 0\n"
            "  \n"
            "  self.states = np.zeros((capacity, state_dim))\n"
            "  self.actions = np.zeros(capacity, dtype=np.int32)\n"
            "  self.rewards = np.zeros(capacity)\n"
            "  self.next_states = np.zeros((capacity, state_dim))\n"
            "  self.dones = np.zeros(capacity, dtype=np.bool_)"
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add transition to buffer."""
        raise NotImplementedError(
            "TODO: Add transition to buffer\n"
            "Hint:\n"
            "  self.states[self.position] = state\n"
            "  self.actions[self.position] = action\n"
            "  self.rewards[self.position] = reward\n"
            "  self.next_states[self.position] = next_state\n"
            "  self.dones[self.position] = done\n"
            "  \n"
            "  self.position = (self.position + 1) % self.capacity\n"
            "  self.size = min(self.size + 1, self.capacity)"
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch of transitions."""
        raise NotImplementedError(
            "TODO: Sample random batch\n"
            "Hint:\n"
            "  indices = np.random.choice(self.size, batch_size, replace=False)\n"
            "  return (\n"
            "      self.states[indices],\n"
            "      self.actions[indices],\n"
            "      self.rewards[indices],\n"
            "      self.next_states[indices],\n"
            "      self.dones[indices]\n"
            "  )"
        )


class C51CategoricalDQN:
    """
    Categorical DQN (C51) for distributional reinforcement learning.

    Instead of learning scalar Q-values, C51 learns the full distribution
    of returns, represented as a categorical distribution over atoms.

    Example:
        >>> agent = C51CategoricalDQN(
        ...     state_dim=4,
        ...     action_dim=2,
        ...     n_atoms=51,
        ...     v_min=-10,
        ...     v_max=10
        ... )
        >>> for episode in range(1000):
        ...     state = env.reset()
        ...     while not done:
        ...         action = agent.select_action(state)
        ...         next_state, reward, done, _ = env.step(action)
        ...         agent.store_transition(state, action, reward, next_state, done)
        ...         agent.update()
        ...         state = next_state

    Args:
        state_dim: Dimension of state space
        action_dim: Number of actions
        n_atoms: Number of atoms in distribution (default: 51)
        v_min: Minimum value support
        v_max: Maximum value support
        hidden_dims: Network hidden dimensions
        learning_rate: Learning rate
        gamma: Discount factor
        buffer_size: Replay buffer size
        batch_size: Training batch size
        target_update_freq: Target network update frequency
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000
    ):
        """Initialize C51 agent."""
        raise NotImplementedError(
            "TODO: Initialize C51 agent\n"
            "Hint:\n"
            "  self.state_dim = state_dim\n"
            "  self.action_dim = action_dim\n"
            "  self.n_atoms = n_atoms\n"
            "  self.v_min = v_min\n"
            "  self.v_max = v_max\n"
            "  self.gamma = gamma\n"
            "  self.batch_size = batch_size\n"
            "  self.target_update_freq = target_update_freq\n"
            "  \n"
            "  # Compute atom spacing\n"
            "  self.delta_z = (v_max - v_min) / (n_atoms - 1)\n"
            "  self.atoms = np.linspace(v_min, v_max, n_atoms)  # Support\n"
            "  \n"
            "  # Network outputs: (action_dim * n_atoms) logits\n"
            "  # Reshaped to (action_dim, n_atoms) and softmaxed\n"
            "  self.network = build_mlp(state_dim, action_dim * n_atoms, hidden_dims)\n"
            "  self.target_network = build_mlp(state_dim, action_dim * n_atoms, hidden_dims)\n"
            "  \n"
            "  # Copy weights to target\n"
            "  self.target_network.load_params(self.network.get_params())\n"
            "  \n"
            "  # Replay buffer\n"
            "  self.buffer = DistributionBuffer(buffer_size, state_dim, n_atoms)\n"
            "  \n"
            "  self.update_count = 0"
        )

    def get_distribution(
        self,
        states: np.ndarray,
        use_target: bool = False
    ) -> np.ndarray:
        """
        Get return distributions for all actions.

        Args:
            states: Batch of states (batch_size, state_dim)
            use_target: Whether to use target network

        Returns:
            distributions: (batch_size, action_dim, n_atoms) probability distributions
        """
        raise NotImplementedError(
            "TODO: Get action distributions\n"
            "Hint:\n"
            "  network = self.target_network if use_target else self.network\n"
            "  logits = network.forward(states)  # (batch, action_dim * n_atoms)\n"
            "  logits = logits.reshape(-1, self.action_dim, self.n_atoms)\n"
            "  \n"
            "  # Softmax over atoms for each action\n"
            "  probs = softmax(logits, axis=-1)\n"
            "  return probs"
        )

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """
        Compute Q-values from distributions.

        Q(s,a) = Σ z_i * p_i(s,a)

        Args:
            states: Batch of states

        Returns:
            q_values: (batch_size, action_dim)
        """
        raise NotImplementedError(
            "TODO: Compute Q-values from distributions\n"
            "Hint:\n"
            "  distributions = self.get_distribution(states)  # (batch, action, atoms)\n"
            "  q_values = np.sum(distributions * self.atoms, axis=-1)\n"
            "  return q_values"
        )

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate

        Returns:
            action: Selected action
        """
        raise NotImplementedError(
            "TODO: Select action\n"
            "Hint:\n"
            "  if np.random.random() < epsilon:\n"
            "      return np.random.randint(self.action_dim)\n"
            "  \n"
            "  q_values = self.get_q_values(state[np.newaxis])[0]\n"
            "  return np.argmax(q_values)"
        )

    def project_distribution(
        self,
        rewards: np.ndarray,
        next_distributions: np.ndarray,
        dones: np.ndarray
    ) -> np.ndarray:
        """
        Project target distribution onto support.

        T_z_j = r + γ * z_j
        Clamp to [v_min, v_max] and distribute probability mass to nearest atoms.

        Args:
            rewards: Batch of rewards (batch_size,)
            next_distributions: Next state distributions (batch_size, n_atoms)
            dones: Done flags (batch_size,)

        Returns:
            projected: Projected distributions (batch_size, n_atoms)
        """
        raise NotImplementedError(
            "TODO: Implement distribution projection (key C51 algorithm)\n"
            "Hint:\n"
            "  batch_size = len(rewards)\n"
            "  projected = np.zeros((batch_size, self.n_atoms))\n"
            "  \n"
            "  for j in range(self.n_atoms):\n"
            "      # Compute Bellman update for each atom\n"
            "      Tz_j = rewards + (1 - dones) * self.gamma * self.atoms[j]\n"
            "      Tz_j = np.clip(Tz_j, self.v_min, self.v_max)\n"
            "      \n"
            "      # Find neighboring atoms\n"
            "      b = (Tz_j - self.v_min) / self.delta_z  # Atom index (float)\n"
            "      l = np.floor(b).astype(np.int32)  # Lower atom\n"
            "      u = np.ceil(b).astype(np.int32)   # Upper atom\n"
            "      \n"
            "      # Handle edge cases\n"
            "      l = np.clip(l, 0, self.n_atoms - 1)\n"
            "      u = np.clip(u, 0, self.n_atoms - 1)\n"
            "      \n"
            "      # Distribute probability mass\n"
            "      for i in range(batch_size):\n"
            "          if l[i] == u[i]:\n"
            "              projected[i, l[i]] += next_distributions[i, j]\n"
            "          else:\n"
            "              projected[i, l[i]] += next_distributions[i, j] * (u[i] - b[i])\n"
            "              projected[i, u[i]] += next_distributions[i, j] * (b[i] - l[i])\n"
            "  \n"
            "  return projected"
        )

    def compute_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> float:
        """
        Compute cross-entropy loss between current and projected distributions.

        L = -Σ m_i log p_i(s,a)
        where m is the projected target distribution.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            loss: Cross-entropy loss
        """
        raise NotImplementedError(
            "TODO: Compute C51 loss\n"
            "Hint:\n"
            "  batch_size = len(states)\n"
            "  \n"
            "  # Get current distributions for taken actions\n"
            "  current_dist = self.get_distribution(states)\n"
            "  current_dist = current_dist[np.arange(batch_size), actions]  # (batch, n_atoms)\n"
            "  \n"
            "  # Get target distributions\n"
            "  with no_grad():\n"
            "      next_q = self.get_q_values_target(next_states)\n"
            "      next_actions = np.argmax(next_q, axis=1)\n"
            "      next_dist = self.get_distribution(next_states, use_target=True)\n"
            "      next_dist = next_dist[np.arange(batch_size), next_actions]\n"
            "  \n"
            "  # Project target distribution\n"
            "  target_dist = self.project_distribution(rewards, next_dist, dones)\n"
            "  \n"
            "  # Cross-entropy loss\n"
            "  loss = -np.sum(target_dist * np.log(current_dist + 1e-8)) / batch_size\n"
            "  return loss"
        )

    def update(self) -> Optional[float]:
        """
        Update network using sampled batch.

        Returns:
            loss: Training loss, or None if buffer too small
        """
        raise NotImplementedError(
            "TODO: Implement C51 update\n"
            "Hint:\n"
            "  if self.buffer.size < self.batch_size:\n"
            "      return None\n"
            "  \n"
            "  # Sample batch\n"
            "  states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)\n"
            "  \n"
            "  # Compute and minimize loss\n"
            "  loss = self.compute_loss(states, actions, rewards, next_states, dones)\n"
            "  self.network.backward(loss)\n"
            "  self.optimizer.step()\n"
            "  \n"
            "  # Update target network\n"
            "  self.update_count += 1\n"
            "  if self.update_count % self.target_update_freq == 0:\n"
            "      self.target_network.load_params(self.network.get_params())\n"
            "  \n"
            "  return loss"
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
