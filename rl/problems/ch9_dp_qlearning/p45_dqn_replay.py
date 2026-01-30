"""
Problem 45: DQN with Experience Replay
======================================

Implement Deep Q-Network with experience replay buffer.

Algorithm (from Lecture Notes Algorithm 15):
    1. Initialize replay buffer D
    2. Initialize Q-network with random weights θ
    3. For each episode:
        For each step:
            - Select action using ε-greedy from Q
            - Execute action, observe r, s'
            - Store (s, a, r, s', done) in D
            - Sample minibatch from D
            - Compute target: y = r + γ max_a' Q(s', a'; θ)
            - Update θ by minimizing (Q(s, a; θ) - y)²

Key Components:
    - Experience replay: breaks correlation in training data
    - Q-network: approximates Q(s, a)
    - ε-greedy exploration: balances exploration and exploitation

References:
    - Lecture notes Section 9.4
    - Mnih et al. (2015) "Human-level control through deep reinforcement learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
import sys
sys.path.insert(0, '../..')

from rl_framework.networks import DiscreteQNetwork
from rl_framework.buffers import ReplayBuffer
from rl_framework.utils import set_seed, get_device
from rl_framework.environments import make_env, get_env_info


class DQN:
    """Deep Q-Network with Experience Replay."""

    def __init__(
        self,
        env,
        hidden_dims: list = [64, 64],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        learning_starts: int = 1000,
        train_freq: int = 4,
        device: str = "auto"
    ):
        """
        Args:
            env: Gymnasium environment
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps for epsilon to decay
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            learning_starts: Steps before training starts
            train_freq: Update network every train_freq steps
            device: Device to use
        """
        self.env = env
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.device = get_device(device)

        # Get environment info
        env_info = get_env_info(env)
        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]

        # Initialize Q-network
        self.q_network = DiscreteQNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=self.state_dim,
            action_dim=1,  # Discrete action stored as single integer
            device=self.device
        )

        # Logging
        self.episode_returns = []
        self.losses = []
        self.total_steps = 0

    def get_epsilon(self) -> float:
        """
        Compute current epsilon using linear decay.

        Returns:
            Current epsilon value

        TODO: Implement linear epsilon decay
            epsilon = epsilon_end + (epsilon_start - epsilon_end) *
                      max(0, 1 - total_steps / epsilon_decay)
        """
        raise NotImplementedError("Implement epsilon decay")

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state
            evaluate: If True, use greedy policy (no exploration)

        Returns:
            Selected action

        TODO: Implement ε-greedy action selection
            1. With probability ε, return random action
            2. Otherwise, return argmax_a Q(s, a)
        """
        if not evaluate and random.random() < self.get_epsilon():
            # TODO: Return random action
            pass
        else:
            # TODO: Return greedy action
            # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # with torch.no_grad():
            #     q_values = self.q_network(state_tensor)
            # return q_values.argmax(dim=1).item()
            pass

        raise NotImplementedError("Implement ε-greedy action selection")

    def update(self) -> float:
        """
        Perform one step of gradient descent on the Q-network.

        Returns:
            Loss value

        TODO: Implement DQN update
            1. Sample minibatch from replay buffer
            2. Compute target: y = r + γ * max_a' Q(s', a') * (1 - done)
            3. Compute loss: MSE(Q(s, a), y)
            4. Backpropagate and update
        """
        # Sample from buffer
        batch = self.buffer.sample(self.batch_size)

        states = batch["states"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # TODO: Compute current Q values
        # current_q = self.q_network(states).gather(1, actions).squeeze()

        # TODO: Compute target Q values
        # with torch.no_grad():
        #     next_q = self.q_network(next_states).max(dim=1)[0]
        #     target_q = rewards.squeeze() + self.gamma * next_q * (1 - dones.squeeze())

        # TODO: Compute loss and update
        # loss = F.mse_loss(current_q, target_q)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return loss.item()

        raise NotImplementedError("Implement DQN update")

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 100
    ):
        """
        Train DQN agent.

        Args:
            total_timesteps: Total timesteps to train
            log_interval: How often to log progress
        """
        obs, _ = self.env.reset()
        episode_return = 0
        episode_length = 0

        for step in range(total_timesteps):
            self.total_steps = step

            # Select action
            action = self.select_action(obs)

            # Execute action
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.buffer.add(obs, np.array([action]), reward, next_obs, done)

            episode_return += reward
            episode_length += 1

            # Update network
            if step >= self.learning_starts and step % self.train_freq == 0:
                loss = self.update()
                self.losses.append(loss)

            if done:
                self.episode_returns.append(episode_return)

                if len(self.episode_returns) % log_interval == 0:
                    recent_returns = self.episode_returns[-log_interval:]
                    print(f"Step {step}, Episode {len(self.episode_returns)}")
                    print(f"  Mean return: {np.mean(recent_returns):.2f}")
                    print(f"  Epsilon: {self.get_epsilon():.3f}")
                    if len(self.losses) > 0:
                        print(f"  Loss: {np.mean(self.losses[-100:]):.4f}")

                obs, _ = self.env.reset()
                episode_return = 0
                episode_length = 0
            else:
                obs = next_obs

    def evaluate(self, n_episodes: int = 10) -> float:
        """Evaluate the current policy."""
        returns = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_return = 0
            done = False

            while not done:
                action = self.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_return += reward

            returns.append(episode_return)

        return np.mean(returns)

    def save(self, path: str):
        """Save Q-network."""
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str):
        """Load Q-network."""
        self.q_network.load_state_dict(torch.load(path))


def main():
    """Run DQN on CartPole-v1."""
    print("=" * 50)
    print("Problem 45: DQN with Experience Replay")
    print("=" * 50)

    # Set seed
    set_seed(42)

    # Create environment
    env = make_env("CartPole-v1", seed=42, normalize=False, monitor=False)

    # Create DQN agent
    agent = DQN(
        env=env,
        hidden_dims=[64, 64],
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=10000,
        buffer_size=50000,
        batch_size=64,
        learning_starts=1000,
        train_freq=4
    )

    # Train
    print("\nTraining DQN...")
    agent.train(total_timesteps=50000, log_interval=10)

    # Evaluate
    mean_return = agent.evaluate(n_episodes=10)
    print(f"\nFinal evaluation: {mean_return:.2f}")

    # Plot learning curve
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(agent.episode_returns)
    window = min(20, len(agent.episode_returns))
    if len(agent.episode_returns) >= window:
        smoothed = np.convolve(agent.episode_returns,
                              np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(agent.episode_returns)), smoothed,
                label='Smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('DQN Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(agent.losses)
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_learning_curve.png', dpi=150)
    plt.show()

    env.close()


if __name__ == "__main__":
    print("\nTODO: Implement the DQN class methods:")
    print("  1. get_epsilon(): Linear epsilon decay")
    print("  2. select_action(): ε-greedy action selection")
    print("  3. update(): DQN update step")
    print("\nRun this file after implementation to test.")
    print("=" * 50)

    # Uncomment after implementation:
    # main()
