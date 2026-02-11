#!/usr/bin/env python3
"""
Example: Train DQN Agent on CartPole

This script demonstrates training a Deep Q-Network on the
classic CartPole balancing task from Gymnasium.

Usage:
    python examples/rl_dqn_cartpole.py

Prerequisites to implement first:
    - rl/value_based/dqn.py (DQN class)
    - rl/core/replay_buffer.py (ReplayBuffer class)
    - nn_core/layers/linear.py (Linear class)
    - nn_core/activations/relu.py (ReLU class)

External dependencies:
    pip install gymnasium
"""

import numpy as np
import sys
sys.path.insert(0, '..')

try:
    import gymnasium as gym
except ImportError:
    print("Please install gymnasium: pip install gymnasium")
    sys.exit(1)

from python.rl.value_based.dqn import DQN
from python.rl.core.replay_buffer import ReplayBuffer


def train_dqn(
    env,
    agent,
    n_episodes: int = 500,
    max_steps: int = 500,
    batch_size: int = 64,
    train_freq: int = 4,
    target_update_freq: int = 100,
    warmup_steps: int = 1000,
    verbose: bool = True
):
    """
    Train DQN agent on environment.

    Args:
        env: Gymnasium environment
        agent: DQN agent
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        batch_size: Batch size for training
        train_freq: Train every N steps
        target_update_freq: Update target network every N steps
        warmup_steps: Steps before training starts
        verbose: Print progress

    Returns:
        episode_rewards: List of total rewards per episode
    """
    episode_rewards = []
    total_steps = 0
    best_reward = -float('inf')

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            total_steps += 1

            # Select action
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.buffer.add(state, action, reward, next_state, done)

            # Train
            if total_steps >= warmup_steps and total_steps % train_freq == 0:
                if len(agent.buffer) >= batch_size:
                    loss = agent.train_step(batch_size)

            # Update target network
            if total_steps % target_update_freq == 0:
                agent.update_target()

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        # Decay epsilon
        agent.decay_epsilon()

        # Track best
        if episode_reward > best_reward:
            best_reward = episode_reward

        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {episode_reward:.0f} | "
                  f"Avg(10): {avg_reward:.1f} | "
                  f"Best: {best_reward:.0f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.buffer)}")

        # Solved condition
        if len(episode_rewards) >= 100:
            if np.mean(episode_rewards[-100:]) >= 475:
                print(f"\nSolved in {episode + 1} episodes!")
                break

    return episode_rewards


def evaluate_policy(env, agent, n_episodes: int = 10, render: bool = False):
    """Evaluate learned policy."""
    total_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(500):
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def main():
    print("=" * 50)
    print("DQN on CartPole-v1")
    print("=" * 50)

    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    print(f"\nEnvironment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {n_actions}")

    # Create agent
    agent = DQN(
        state_dim=state_dim,
        action_dim=n_actions,
        hidden_dims=[128, 128],
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000
    )

    print(f"\nAgent: DQN")
    print(f"Hidden layers: [128, 128]")
    print(f"Learning rate: {agent.learning_rate}")

    # Train
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    rewards = train_dqn(
        env, agent,
        n_episodes=500,
        batch_size=64,
        verbose=True
    )

    # Evaluate
    print("\n" + "=" * 50)
    print("Evaluation")
    print("=" * 50)

    mean_reward, std_reward = evaluate_policy(env, agent)
    print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")

    env.close()

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(rewards, alpha=0.3, label='Raw')

        # Smoothed
        window = 20
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, label=f'Smoothed (window={window})')

        ax.axhline(y=475, color='g', linestyle='--', label='Solved threshold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('DQN on CartPole-v1')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('dqn_cartpole_results.png')
        print("\nPlot saved to dqn_cartpole_results.png")

    except ImportError:
        print("\nInstall matplotlib to see learning curves")


if __name__ == "__main__":
    main()
