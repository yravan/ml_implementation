"""
Experience Replay Buffers
=========================

Implementations of various replay buffers:
- Standard replay buffer
- Prioritized experience replay
- Trajectory/rollout buffer for on-policy algorithms
- Hindsight Experience Replay (HER) buffer
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, NamedTuple
from collections import deque
import random


class Transition(NamedTuple):
    """Single transition tuple."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Standard experience replay buffer for off-policy algorithms."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": torch.FloatTensor(self.states[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[indices]).to(self.device),
            "dones": torch.FloatTensor(self.dones[indices]).to(self.device),
        }

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples transitions with probability proportional to TD-error priority.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: str = "cpu"
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.ptr = 0
        self.size = 0

        # Data storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        # Priority tree for efficient sampling
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add transition with maximum priority."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = {
            "states": torch.FloatTensor(self.states[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[indices]).to(self.device),
            "dones": torch.FloatTensor(self.dones[indices]).to(self.device),
        }

        return batch, indices, torch.FloatTensor(weights).to(self.device)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        priorities = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO, A2C, REINFORCE).
    Stores complete trajectories and computes advantages.
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu"
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.ptr = 0

        # Pre-allocate memory
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Computed quantities
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float, last_done: bool):
        """
        Compute returns and GAE advantages.
        Must be called after collecting a full rollout.
        """
        # GAE computation
        last_gae = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data from buffer."""
        return {
            "states": torch.FloatTensor(self.states[:self.ptr]).to(self.device),
            "actions": torch.FloatTensor(self.actions[:self.ptr]).to(self.device),
            "log_probs": torch.FloatTensor(self.log_probs[:self.ptr]).to(self.device),
            "advantages": torch.FloatTensor(self.advantages[:self.ptr]).to(self.device),
            "returns": torch.FloatTensor(self.returns[:self.ptr]).to(self.device),
            "values": torch.FloatTensor(self.values[:self.ptr]).to(self.device),
        }

    def sample_minibatches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Generate random minibatches from the buffer."""
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)

        minibatches = []
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_indices = indices[start:end]

            minibatches.append({
                "states": torch.FloatTensor(self.states[batch_indices]).to(self.device),
                "actions": torch.FloatTensor(self.actions[batch_indices]).to(self.device),
                "log_probs": torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                "advantages": torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                "returns": torch.FloatTensor(self.returns[batch_indices]).to(self.device),
                "values": torch.FloatTensor(self.values[batch_indices]).to(self.device),
            })

        return minibatches

    def reset(self):
        """Reset the buffer."""
        self.ptr = 0


class HERBuffer:
    """
    Hindsight Experience Replay buffer for goal-conditioned RL.
    Implements the 'future' strategy for goal relabeling.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        n_sampled_goals: int = 4,
        goal_selection_strategy: str = "future",
        device: str = "cpu"
    ):
        self.capacity = capacity
        self.device = device
        self.n_sampled_goals = n_sampled_goals
        self.goal_selection_strategy = goal_selection_strategy
        self.ptr = 0
        self.size = 0

        # Episode buffer
        self.current_episode = []

        # Main buffer stores (state, action, reward, next_state, done, goal, achieved_goal)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        goal: np.ndarray,
        achieved_goal: np.ndarray
    ):
        """Add transition to current episode."""
        self.current_episode.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "goal": goal,
            "achieved_goal": achieved_goal
        })

    def end_episode(self, compute_reward_fn):
        """
        End current episode and add to buffer with HER goal relabeling.

        Args:
            compute_reward_fn: Function(achieved_goal, desired_goal) -> reward
        """
        episode = self.current_episode
        episode_length = len(episode)

        for t, transition in enumerate(episode):
            # Add original transition
            self._add_to_buffer(
                transition["state"],
                transition["action"],
                transition["reward"],
                transition["next_state"],
                transition["done"],
                transition["goal"]
            )

            # Sample additional goals using HER
            if self.goal_selection_strategy == "future":
                # Sample from future states in episode
                future_indices = np.random.randint(t, episode_length, size=self.n_sampled_goals)

                for future_idx in future_indices:
                    # Use achieved goal from future state as new goal
                    new_goal = episode[future_idx]["achieved_goal"]

                    # Recompute reward with new goal
                    new_reward = compute_reward_fn(
                        transition["achieved_goal"],
                        new_goal
                    )

                    self._add_to_buffer(
                        transition["state"],
                        transition["action"],
                        new_reward,
                        transition["next_state"],
                        transition["done"],
                        new_goal
                    )

        self.current_episode = []

    def _add_to_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        goal: np.ndarray
    ):
        """Add single transition to main buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.goals[self.ptr] = goal

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        # Concatenate state and goal for goal-conditioned policy
        states_with_goals = np.concatenate([self.states[indices], self.goals[indices]], axis=1)
        next_states_with_goals = np.concatenate([self.next_states[indices], self.goals[indices]], axis=1)

        return {
            "states": torch.FloatTensor(states_with_goals).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_states": torch.FloatTensor(next_states_with_goals).to(self.device),
            "dones": torch.FloatTensor(self.dones[indices]).to(self.device),
        }

    def __len__(self) -> int:
        return self.size


class TrajectoryBuffer:
    """
    Buffer for storing complete trajectories.
    Useful for REINFORCE, imitation learning, and trajectory optimization.
    """

    def __init__(self, max_trajectories: int = 1000):
        self.max_trajectories = max_trajectories
        self.trajectories = deque(maxlen=max_trajectories)
        self.current_trajectory = []

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[dict] = None
    ):
        """Add transition to current trajectory."""
        self.current_trajectory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "info": info or {}
        })

    def end_trajectory(self):
        """End current trajectory and add to buffer."""
        if len(self.current_trajectory) > 0:
            self.trajectories.append(self.current_trajectory.copy())
            self.current_trajectory = []

    def sample_trajectories(self, n_trajectories: int) -> List[List[dict]]:
        """Sample n complete trajectories."""
        n = min(n_trajectories, len(self.trajectories))
        return random.sample(list(self.trajectories), n)

    def get_all_trajectories(self) -> List[List[dict]]:
        """Get all stored trajectories."""
        return list(self.trajectories)

    def compute_returns(self, gamma: float = 0.99) -> List[np.ndarray]:
        """Compute discounted returns for all trajectories."""
        all_returns = []

        for trajectory in self.trajectories:
            rewards = [t["reward"] for t in trajectory]
            returns = np.zeros(len(rewards))

            running_return = 0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                returns[t] = running_return

            all_returns.append(returns)

        return all_returns

    def __len__(self) -> int:
        return len(self.trajectories)
