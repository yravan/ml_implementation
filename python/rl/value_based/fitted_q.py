"""
Fitted Q-Iteration (FQI) - Batch Reinforcement Learning

Theory:
    Fitted Q-Iteration is an offline/batch RL algorithm that learns from a
    fixed dataset of transitions without requiring further environment interaction.
    It iteratively fits a function approximator to approximate Q-values using
    the Bellman equation as a fixed-point iteration.

Mathematical Framework:
    Given dataset D = {(s_i, a_i, r_i, s'_i)}, repeat:
    1. Build targets: y_i = r_i + γ * max_a' Q̂(s'_i, a')
    2. Fit Q̂ to minimize: Σ_i (Q̂(s_i, a_i) - y_i)²

    This is related to Q-learning but:
    - Uses full batch updates instead of online updates
    - Can use any supervised learning algorithm
    - Well-suited for offline RL settings

Algorithm Variants:
    - Neural FQI: Use neural network as function approximator
    - Extra-Trees FQI: Use ensemble of trees (often more stable)
    - Conservative Q-Learning: Add regularization for offline RL

Convergence:
    FQI converges to optimal Q* under certain conditions:
    - Finite state/action spaces (tabular case)
    - Sufficient data coverage
    - Appropriate function class

References:
    - Ernst et al. (2005): "Tree-Based Batch Mode RL"
      https://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
    - Riedmiller (2005): "Neural Fitted Q Iteration"
      https://ml.informatik.uni-freiburg.de/former/_media/publications/riedmiller05b.pdf
    - Kumar et al. (2020): "Conservative Q-Learning for Offline RL"
      https://arxiv.org/abs/2006.04779
"""

# Implementation Status: NOT STARTED
# Complexity: Medium
# Prerequisites: Basic regression, Q-learning concepts

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable


class BatchReplayBuffer:
    """
    Fixed batch buffer for offline RL.

    Unlike online replay buffers, this stores a fixed dataset
    that doesn't grow during training.

    Args:
        states: Array of states
        actions: Array of actions
        rewards: Array of rewards
        next_states: Array of next states
        dones: Array of done flags
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ):
        """Initialize batch buffer from fixed dataset."""
        raise NotImplementedError(
            "TODO: Initialize batch buffer\n"
            "Hint:\n"
            "  self.states = states\n"
            "  self.actions = actions\n"
            "  self.rewards = rewards\n"
            "  self.next_states = next_states\n"
            "  self.dones = dones\n"
            "  self.size = len(states)"
        )

    @classmethod
    def from_transitions(cls, transitions: List[Tuple]) -> 'BatchReplayBuffer':
        """
        Create buffer from list of transitions.

        Args:
            transitions: List of (s, a, r, s', done) tuples

        Returns:
            buffer: BatchReplayBuffer instance
        """
        raise NotImplementedError(
            "TODO: Create buffer from transitions list\n"
            "Hint:\n"
            "  states = np.array([t[0] for t in transitions])\n"
            "  actions = np.array([t[1] for t in transitions])\n"
            "  rewards = np.array([t[2] for t in transitions])\n"
            "  next_states = np.array([t[3] for t in transitions])\n"
            "  dones = np.array([t[4] for t in transitions])\n"
            "  return cls(states, actions, rewards, next_states, dones)"
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample random batch from buffer."""
        raise NotImplementedError(
            "TODO: Sample batch\n"
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

    def get_all(self) -> Tuple[np.ndarray, ...]:
        """Return all data (for full batch updates)."""
        return (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones
        )


class FittedQIteration:
    """
    Fitted Q-Iteration for batch/offline reinforcement learning.

    Learns Q-function from fixed dataset without environment interaction.
    Can use any regression model as the function approximator.

    Example:
        >>> # Collect dataset from behavior policy
        >>> dataset = collect_data(env, behavior_policy, n_episodes=100)
        >>> buffer = BatchReplayBuffer.from_transitions(dataset)
        >>>
        >>> # Train FQI
        >>> agent = FittedQIteration(
        ...     state_dim=4,
        ...     action_dim=2,
        ...     gamma=0.99
        ... )
        >>> agent.fit(buffer, n_iterations=100)
        >>>
        >>> # Evaluate
        >>> action = agent.select_action(state)

    Args:
        state_dim: Dimension of state space
        action_dim: Number of discrete actions
        hidden_dims: Hidden layer sizes for Q-network
        learning_rate: Learning rate for neural network
        gamma: Discount factor
        n_fitting_epochs: Epochs per FQI iteration
        batch_size: Batch size for training
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        n_fitting_epochs: int = 100,
        batch_size: int = 64
    ):
        """Initialize FQI agent."""
        raise NotImplementedError(
            "TODO: Initialize FQI agent\n"
            "Hint:\n"
            "  self.state_dim = state_dim\n"
            "  self.action_dim = action_dim\n"
            "  self.gamma = gamma\n"
            "  self.n_fitting_epochs = n_fitting_epochs\n"
            "  self.batch_size = batch_size\n"
            "  \n"
            "  # Q-network: state -> Q-values for all actions\n"
            "  self.q_network = build_mlp(state_dim, action_dim, hidden_dims)\n"
            "  self.optimizer = Adam(lr=learning_rate)"
        )

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions.

        Args:
            states: Batch of states (batch_size, state_dim)

        Returns:
            q_values: (batch_size, action_dim)
        """
        raise NotImplementedError(
            "TODO: Get Q-values\n"
            "Hint:\n"
            "  return self.q_network.forward(states)"
        )

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate (for evaluation)

        Returns:
            action: Selected action
        """
        raise NotImplementedError(
            "TODO: Select greedy action\n"
            "Hint:\n"
            "  if np.random.random() < epsilon:\n"
            "      return np.random.randint(self.action_dim)\n"
            "  q_values = self.get_q_values(state[np.newaxis])[0]\n"
            "  return np.argmax(q_values)"
        )

    def compute_targets(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> np.ndarray:
        """
        Compute Bellman targets.

        y = r + γ * max_a' Q(s', a') * (1 - done)

        Args:
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            targets: Bellman targets
        """
        raise NotImplementedError(
            "TODO: Compute Bellman targets\n"
            "Hint:\n"
            "  next_q_values = self.get_q_values(next_states)\n"
            "  max_next_q = np.max(next_q_values, axis=1)\n"
            "  targets = rewards + self.gamma * max_next_q * (1 - dones)\n"
            "  return targets"
        )

    def fit_iteration(
        self,
        buffer: BatchReplayBuffer
    ) -> float:
        """
        Perform one FQI iteration.

        1. Compute targets for all transitions
        2. Fit Q-network to minimize MSE to targets

        Args:
            buffer: Batch replay buffer

        Returns:
            loss: Final fitting loss
        """
        raise NotImplementedError(
            "TODO: Implement one FQI iteration\n"
            "Hint:\n"
            "  states, actions, rewards, next_states, dones = buffer.get_all()\n"
            "  \n"
            "  # Compute targets (no gradient through target computation)\n"
            "  targets = self.compute_targets(rewards, next_states, dones)\n"
            "  \n"
            "  # Fit Q-network\n"
            "  for epoch in range(self.n_fitting_epochs):\n"
            "      indices = np.random.permutation(buffer.size)\n"
            "      \n"
            "      for start in range(0, buffer.size, self.batch_size):\n"
            "          batch_idx = indices[start:start + self.batch_size]\n"
            "          \n"
            "          # Get Q-values for taken actions\n"
            "          q_values = self.get_q_values(states[batch_idx])\n"
            "          q_taken = q_values[np.arange(len(batch_idx)), actions[batch_idx]]\n"
            "          \n"
            "          # MSE loss\n"
            "          loss = np.mean((q_taken - targets[batch_idx]) ** 2)\n"
            "          \n"
            "          # Update\n"
            "          self.q_network.backward(loss)\n"
            "          self.optimizer.step(self.q_network)\n"
            "  \n"
            "  return loss"
        )

    def fit(
        self,
        buffer: BatchReplayBuffer,
        n_iterations: int = 100,
        verbose: bool = True
    ) -> List[float]:
        """
        Run FQI algorithm for multiple iterations.

        Args:
            buffer: Batch replay buffer with fixed dataset
            n_iterations: Number of FQI iterations
            verbose: Whether to print progress

        Returns:
            losses: List of losses per iteration
        """
        raise NotImplementedError(
            "TODO: Run full FQI algorithm\n"
            "Hint:\n"
            "  losses = []\n"
            "  for i in range(n_iterations):\n"
            "      loss = self.fit_iteration(buffer)\n"
            "      losses.append(loss)\n"
            "      if verbose and i % 10 == 0:\n"
            "          print(f'Iteration {i}, Loss: {loss:.4f}')\n"
            "  return losses"
        )

    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        epsilon: float = 0.0
    ) -> float:
        """
        Evaluate learned policy.

        Args:
            env: Environment to evaluate in
            n_episodes: Number of evaluation episodes
            epsilon: Exploration rate

        Returns:
            mean_return: Mean episode return
        """
        raise NotImplementedError(
            "TODO: Evaluate policy\n"
            "Hint:\n"
            "  returns = []\n"
            "  for _ in range(n_episodes):\n"
            "      state = env.reset()\n"
            "      episode_return = 0\n"
            "      done = False\n"
            "      while not done:\n"
            "          action = self.select_action(state, epsilon)\n"
            "          state, reward, done, _ = env.step(action)\n"
            "          episode_return += reward\n"
            "      returns.append(episode_return)\n"
            "  return np.mean(returns)"
        )


class ConservativeFQI(FittedQIteration):
    """
    Conservative Fitted Q-Iteration for offline RL.

    Adds a regularization term to prevent overestimation on
    out-of-distribution actions, which is a key challenge in offline RL.

    CQL Loss = FQI Loss + α * E[log Σ_a exp(Q(s,a)) - Q(s, a_data)]

    Args:
        alpha: Conservative regularization coefficient
        (other args same as FittedQIteration)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 1.0,
        **kwargs
    ):
        """Initialize Conservative FQI."""
        raise NotImplementedError(
            "TODO: Initialize Conservative FQI\n"
            "Hint:\n"
            "  super().__init__(state_dim, action_dim, **kwargs)\n"
            "  self.alpha = alpha"
        )

    def compute_conservative_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> float:
        """
        Compute CQL regularization term.

        L_CQL = E[log Σ_a exp(Q(s,a)) - Q(s, a_data)]

        This penalizes high Q-values on unseen actions.

        Args:
            states: Batch of states
            actions: Actions taken in data

        Returns:
            cql_loss: Conservative regularization loss
        """
        raise NotImplementedError(
            "TODO: Implement CQL loss\n"
            "Hint:\n"
            "  q_values = self.get_q_values(states)\n"
            "  \n"
            "  # Log-sum-exp over actions (pushes down all Q-values)\n"
            "  logsumexp = np.log(np.sum(np.exp(q_values), axis=1) + 1e-8)\n"
            "  \n"
            "  # Q-values for data actions (pushes up data Q-values)\n"
            "  q_data = q_values[np.arange(len(actions)), actions]\n"
            "  \n"
            "  cql_loss = np.mean(logsumexp - q_data)\n"
            "  return cql_loss"
        )
