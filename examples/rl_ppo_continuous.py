#!/usr/bin/env python3
"""
Example: Train PPO Agent on Continuous Control

This script demonstrates training Proximal Policy Optimization
on a continuous control task (Pendulum or HalfCheetah).

Usage:
    python examples/rl_ppo_continuous.py

Prerequisites to implement first:
    - rl/policy_gradient/ppo.py (PPO class)
    - rl/core/policies.py (GaussianPolicy class)
    - rl/core/value_functions.py (ValueNetwork class)
    - rl/core/advantage.py (compute_gae function)

External dependencies:
    pip install gymnasium
    # For MuJoCo envs: pip install gymnasium[mujoco]
"""

import numpy as np
import sys
sys.path.insert(0, '..')

try:
    import gymnasium as gym
except ImportError:
    print("Please install gymnasium: pip install gymnasium")
    sys.exit(1)

from python.rl.policy_gradient.ppo import PPO


def collect_rollout(env, policy, n_steps: int = 2048):
    """
    Collect rollout data for PPO training.

    Args:
        env: Gymnasium environment
        policy: PPO policy
        n_steps: Number of steps to collect

    Returns:
        Dictionary with states, actions, rewards, dones, values, log_probs
    """
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []

    state, _ = env.reset()
    episode_reward = 0
    episode_rewards = []

    for _ in range(n_steps):
        # Get action, value, log_prob
        action, log_prob, value = policy.get_action_and_value(state)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        log_probs.append(log_prob)

        episode_reward += reward
        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            state, _ = env.reset()

    # Get final value for GAE
    _, _, last_value = policy.get_action_and_value(state)

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'values': np.array(values),
        'log_probs': np.array(log_probs),
        'last_value': last_value,
        'episode_rewards': episode_rewards
    }


def train_ppo(
    env,
    agent,
    total_timesteps: int = 1_000_000,
    n_steps: int = 2048,
    n_epochs: int = 10,
    batch_size: int = 64,
    verbose: bool = True
):
    """
    Train PPO agent.

    Args:
        env: Gymnasium environment
        agent: PPO agent
        total_timesteps: Total training timesteps
        n_steps: Steps per rollout
        n_epochs: Epochs per PPO update
        batch_size: Mini-batch size
        verbose: Print progress

    Returns:
        all_rewards: Episode rewards
    """
    all_rewards = []
    timesteps = 0
    iteration = 0

    while timesteps < total_timesteps:
        iteration += 1

        # Collect rollout
        rollout = collect_rollout(env, agent, n_steps)
        timesteps += n_steps

        # Compute advantages
        advantages, returns = agent.compute_advantages(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
            rollout['last_value']
        )

        # PPO update
        metrics = agent.update(
            states=rollout['states'],
            actions=rollout['actions'],
            old_log_probs=rollout['log_probs'],
            advantages=advantages,
            returns=returns,
            n_epochs=n_epochs,
            batch_size=batch_size
        )

        # Track rewards
        all_rewards.extend(rollout['episode_rewards'])

        if verbose and iteration % 5 == 0:
            if len(all_rewards) > 0:
                recent_rewards = all_rewards[-10:] if len(all_rewards) >= 10 else all_rewards
                avg_reward = np.mean(recent_rewards)
                print(f"Iter {iteration} | "
                      f"Timesteps: {timesteps:,} | "
                      f"Avg Reward: {avg_reward:.1f} | "
                      f"Policy Loss: {metrics['policy_loss']:.4f} | "
                      f"Value Loss: {metrics['value_loss']:.4f}")

    return all_rewards


def evaluate_policy(env, agent, n_episodes: int = 10):
    """Evaluate PPO policy."""
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(1000):
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    print("=" * 50)
    print("PPO on Continuous Control")
    print("=" * 50)

    # Create environment
    # Use Pendulum for quick testing, HalfCheetah for real benchmark
    env_name = 'Pendulum-v1'  # or 'HalfCheetah-v4'

    try:
        env = gym.make(env_name)
    except:
        print(f"Could not create {env_name}")
        print("Trying InvertedPendulum-v4...")
        env_name = 'InvertedPendulum-v4'
        env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]

    print(f"\nEnvironment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action range: [-{action_high}, {action_high}]")

    # Create PPO agent
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5
    )

    print(f"\nAgent: PPO")
    print(f"Hidden layers: [64, 64]")
    print(f"Clip epsilon: {agent.clip_epsilon}")

    # Train
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    rewards = train_ppo(
        env, agent,
        total_timesteps=100_000,  # Use 1M for real training
        n_steps=2048,
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

        ax.plot(rewards, alpha=0.3, label='Episodes')

        if len(rewards) >= 10:
            window = min(50, len(rewards) // 5)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, label=f'Smoothed')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(f'PPO on {env_name}')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('ppo_results.png')
        print("\nPlot saved to ppo_results.png")

    except ImportError:
        print("\nInstall matplotlib to see learning curves")


if __name__ == "__main__":
    main()
