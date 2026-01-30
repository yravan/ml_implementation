"""
Problem 20: Proximal Policy Optimization (PPO) - Clipped
=========================================================

Implement PPO with clipped surrogate objective.

Algorithm Overview:
    1. Collect trajectories using current policy π_θ
    2. Compute advantages using GAE
    3. For K epochs:
        - Sample minibatches from collected data
        - Update policy using clipped objective
        - Update value function

Clipped Objective:
    L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

where:
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
    ε = 0.2 (typical value)

References:
    - Lecture notes Section 4.5.3
    - Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../..')

from rl_framework.networks import ActorCritic
from rl_framework.buffers import RolloutBuffer
from rl_framework.utils import compute_gae, set_seed, get_device
from rl_framework.environments import make_env, get_env_info


class PPO:
    """Proximal Policy Optimization with clipped objective."""

    def __init__(
        self,
        env,
        hidden_dims: list = [64, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto"
    ):
        """
        Args:
            env: Gymnasium environment
            hidden_dims: Hidden layer dimensions for networks
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Minibatch size
            device: Device to use
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = get_device(device)

        # Get environment info
        env_info = get_env_info(env)
        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.is_discrete = env_info["is_discrete"]

        # Initialize actor-critic network
        self.policy = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            continuous=not self.is_discrete
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Logging
        self.episode_returns = []
        self.episode_lengths = []

    def collect_rollouts(self, n_steps: int) -> RolloutBuffer:
        """
        Collect trajectories using current policy.

        Args:
            n_steps: Number of steps to collect

        Returns:
            RolloutBuffer containing collected data

        TODO: Implement rollout collection
            1. Initialize buffer
            2. For each step:
                - Get action and value from policy
                - Execute action in environment
                - Store transition in buffer
            3. Compute advantages using GAE
        """
        buffer = RolloutBuffer(
            buffer_size=n_steps,
            state_dim=self.state_dim,
            action_dim=1 if self.is_discrete else self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device
        )

        obs, _ = self.env.reset()
        episode_return = 0
        episode_length = 0

        for step in range(n_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, log_prob, entropy, value = self.policy.get_action_and_value(obs_tensor)

                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

            # Execute action
            if self.is_discrete:
                action_env = int(action)
            else:
                action_env = action

            next_obs, reward, terminated, truncated, info = self.env.step(action_env)
            done = terminated or truncated

            # Store transition
            buffer.add(
                state=obs,
                action=np.array([action]) if self.is_discrete else action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )

            episode_return += reward
            episode_length += 1

            if done:
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(episode_length)
                obs, _ = self.env.reset()
                episode_return = 0
                episode_length = 0
            else:
                obs = next_obs

        # Compute advantages
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, _, _, last_value = self.policy.get_action_and_value(obs_tensor)
            last_value = last_value.cpu().numpy()

        buffer.compute_returns_and_advantages(last_value, done)

        return buffer

    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Update policy using PPO clipped objective.

        Args:
            buffer: RolloutBuffer with collected data

        Returns:
            Dictionary of training metrics

        TODO: Implement PPO update
            1. Get all data from buffer
            2. For each epoch:
                - Shuffle and create minibatches
                - For each minibatch:
                    - Compute new log probs and values
                    - Compute ratio: r_t = exp(log_prob_new - log_prob_old)
                    - Compute clipped objective
                    - Compute value loss
                    - Compute entropy bonus
                    - Backprop combined loss
        """
        # Get data from buffer
        data = buffer.get()
        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            # Get minibatches
            minibatches = buffer.sample_minibatches(self.batch_size)

            for batch in minibatches:
                # TODO: Implement PPO update step

                # 1. Get new log probs, entropy, and values
                # new_log_probs, entropy, values = ...

                # 2. Compute ratio
                # ratio = torch.exp(new_log_probs - batch["log_probs"])

                # 3. Compute clipped objective
                # surr1 = ratio * batch["advantages"]
                # surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch["advantages"]
                # policy_loss = -torch.min(surr1, surr2).mean()

                # 4. Compute value loss
                # value_loss = F.mse_loss(values, batch["returns"])

                # 5. Compute entropy bonus
                # entropy_loss = -entropy.mean()

                # 6. Combined loss
                # loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 7. Backprop
                # self.optimizer.zero_grad()
                # loss.backward()
                # nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # self.optimizer.step()

                raise NotImplementedError("Implement PPO update step")

        return {
            "policy_loss": total_policy_loss / n_updates if n_updates > 0 else 0,
            "value_loss": total_value_loss / n_updates if n_updates > 0 else 0,
            "entropy": total_entropy / n_updates if n_updates > 0 else 0,
        }

    def train(
        self,
        total_timesteps: int,
        rollout_steps: int = 2048,
        log_interval: int = 1
    ):
        """
        Train PPO agent.

        Args:
            total_timesteps: Total timesteps to train
            rollout_steps: Steps per rollout
            log_interval: How often to log progress
        """
        n_updates = total_timesteps // rollout_steps

        for update in range(n_updates):
            # Collect rollouts
            buffer = self.collect_rollouts(rollout_steps)

            # Update policy
            metrics = self.update(buffer)
            buffer.reset()

            # Log progress
            if (update + 1) % log_interval == 0 and len(self.episode_returns) > 0:
                recent_returns = self.episode_returns[-10:]
                print(f"Update {update + 1}/{n_updates}")
                print(f"  Mean return: {np.mean(recent_returns):.2f}")
                print(f"  Policy loss: {metrics['policy_loss']:.4f}")
                print(f"  Value loss: {metrics['value_loss']:.4f}")

    def save(self, path: str):
        """Save policy."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        """Load policy."""
        self.policy.load_state_dict(torch.load(path))


def main():
    """Run PPO on CartPole-v1."""
    print("=" * 50)
    print("Problem 20: PPO with Clipped Objective")
    print("=" * 50)

    # Set seed for reproducibility
    set_seed(42)

    # Create environment
    env = make_env("CartPole-v1", seed=42, normalize=False)

    # Create PPO agent
    agent = PPO(
        env=env,
        hidden_dims=[64, 64],
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=10,
        batch_size=64
    )

    # Train
    print("\nTraining PPO...")
    agent.train(total_timesteps=100000, rollout_steps=2048)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(agent.episode_returns)

    # Smoothed curve
    window = min(20, len(agent.episode_returns))
    if len(agent.episode_returns) >= window:
        smoothed = np.convolve(agent.episode_returns,
                              np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(agent.episode_returns)), smoothed,
                label=f'Smoothed (window={window})')

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('PPO on CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ppo_learning_curve.png', dpi=150)
    plt.show()

    env.close()


if __name__ == "__main__":
    print("\nTODO: Implement the PPO class methods:")
    print("  1. update(): PPO clipped objective update")
    print("\nRun this file after implementation to test.")
    print("=" * 50)

    # Uncomment after implementation:
    # main()
