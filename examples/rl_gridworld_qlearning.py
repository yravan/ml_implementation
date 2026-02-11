#!/usr/bin/env python3
"""
Example: Train Q-Learning Agent on GridWorld

This script demonstrates training a tabular Q-learning agent
on a simple gridworld environment. Run this after implementing
the core RL modules.

Usage:
    python examples/rl_gridworld_qlearning.py

Prerequisites to implement first:
    - envs/gridworld.py (GridWorld class)
    - rl/tabular/q_learning.py (QLearning class)
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from python.envs.gridworld import GridWorld
from python.rl.tabular.q_learning import QLearning


def train_qlearning(
    env,
    agent,
    n_episodes: int = 500,
    max_steps: int = 100,
    verbose: bool = True
):
    """
    Train Q-learning agent on environment.

    Args:
        env: GridWorld environment
        agent: QLearning agent
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Print progress

    Returns:
        episode_rewards: List of total rewards per episode
        episode_lengths: List of steps per episode
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Select action (epsilon-greedy)
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update Q-table
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

        # Decay epsilon
        agent.decay_epsilon()

        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return episode_rewards, episode_lengths


def evaluate_policy(env, agent, n_episodes: int = 10):
    """Evaluate learned policy (no exploration)."""
    total_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(100):
            action = agent.select_action(state, greedy=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def visualize_policy(env, agent):
    """Visualize learned policy as arrows."""
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    print("\nLearned Policy:")
    print("-" * (env.width * 3 + 1))

    for row in range(env.height):
        line = "|"
        for col in range(env.width):
            pos = (row, col)
            if pos == env.goal:
                line += " G "
            elif pos in env.obstacles:
                line += " # "
            else:
                state = env.pos_to_state(pos)
                action = np.argmax(agent.q_table[state])
                line += f" {arrows[action]} "
        print(line + "|")

    print("-" * (env.width * 3 + 1))


def main():
    print("=" * 50)
    print("Q-Learning on GridWorld")
    print("=" * 50)

    # Create environment
    env = GridWorld(
        grid_size=(5, 5),
        start=(0, 0),
        goal=(4, 4),
        obstacles=[(1, 1), (2, 2), (3, 1)],
        reward_goal=10.0,
        reward_step=-0.1
    )

    print(f"\nEnvironment: {env.height}x{env.width} grid")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Obstacles: {env.obstacles}")

    # Create agent
    agent = QLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    print(f"\nAgent: Q-Learning")
    print(f"Learning rate: {agent.learning_rate}")
    print(f"Discount factor: {agent.discount_factor}")

    # Train
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    rewards, lengths = train_qlearning(
        env, agent,
        n_episodes=500,
        verbose=True
    )

    # Evaluate
    print("\n" + "=" * 50)
    print("Evaluation (greedy policy)")
    print("=" * 50)

    mean_reward, std_reward = evaluate_policy(env, agent)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Visualize
    visualize_policy(env, agent)

    # Plot learning curve (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Smooth rewards
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

        ax1.plot(smoothed)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Learning Curve (Smoothed)')
        ax1.grid(True)

        ax2.plot(lengths)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('qlearning_results.png')
        print("\nPlot saved to qlearning_results.png")

    except ImportError:
        print("\nInstall matplotlib to see learning curves:")
        print("  pip install matplotlib")


if __name__ == "__main__":
    main()
