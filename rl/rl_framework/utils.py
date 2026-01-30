"""
Utilities for Training, Logging, and Visualization
===================================================

Provides:
- Training utilities
- Logging and metrics
- Visualization tools
- Evaluation utilities
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable, Any
import time
from collections import deque
import json
import os


# ========================
# Training Utilities
# ========================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Soft update target network parameters: θ_target = τ*θ_source + (1-τ)*θ_target"""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module, source_net: nn.Module):
    """Hard update target network parameters: θ_target = θ_source"""
    target_net.load_state_dict(source_net.state_dict())


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Array of rewards
        values: Array of value estimates
        dones: Array of done flags
        next_value: Value estimate for the last state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps)
    last_gae = 0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


def compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute discounted returns.

    Args:
        rewards: Array of rewards
        dones: Array of done flags
        gamma: Discount factor
        normalize: Whether to normalize returns

    Returns:
        returns: Discounted returns
    """
    n_steps = len(rewards)
    returns = np.zeros(n_steps)
    running_return = 0

    for t in reversed(range(n_steps)):
        if dones[t]:
            running_return = 0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


# ========================
# Logging and Metrics
# ========================

class Logger:
    """Simple logger for training metrics."""

    def __init__(self, log_dir: Optional[str] = None):
        self.metrics: Dict[str, List[float]] = {}
        self.log_dir = log_dir
        self.step = 0

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for the current step."""
        if step is not None:
            self.step = step
        else:
            self.step += 1

        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_recent(self, key: str, n: int = 100) -> List[float]:
        """Get the most recent n values for a metric."""
        if key in self.metrics:
            return self.metrics[key][-n:]
        return []

    def get_mean(self, key: str, n: int = 100) -> float:
        """Get the mean of the most recent n values."""
        values = self.get_recent(key, n)
        if len(values) > 0:
            return np.mean(values)
        return 0.0

    def save(self, filename: str = "metrics.json"):
        """Save metrics to file."""
        if self.log_dir is not None:
            filepath = os.path.join(self.log_dir, filename)
            with open(filepath, "w") as f:
                json.dump(self.metrics, f)

    def load(self, filename: str = "metrics.json"):
        """Load metrics from file."""
        if self.log_dir is not None:
            filepath = os.path.join(self.log_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    self.metrics = json.load(f)


class RunningMeanStd:
    """Tracks running mean and standard deviation."""

    def __init__(self, shape=(), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count


# ========================
# Visualization
# ========================

def plot_learning_curves(
    metrics: Dict[str, List[float]],
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (12, 8),
    smoothing: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot learning curves for multiple metrics.

    Args:
        metrics: Dictionary mapping metric names to lists of values
        title: Plot title
        figsize: Figure size
        smoothing: Window size for smoothing
        save_path: Path to save the figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        x = np.arange(len(values))

        # Raw values
        ax.plot(x, values, alpha=0.3, label="Raw")

        # Smoothed values
        if len(values) >= smoothing:
            smoothed = np.convolve(values, np.ones(smoothing) / smoothing, mode="valid")
            ax.plot(np.arange(len(smoothed)) + smoothing // 2, smoothed, label="Smoothed")

        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_episode_returns(
    returns: List[float],
    title: str = "Episode Returns",
    figsize: Tuple[int, int] = (10, 6),
    smoothing: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot episode returns with confidence intervals.

    Args:
        returns: List of episode returns
        title: Plot title
        figsize: Figure size
        smoothing: Window size for moving average
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    episodes = np.arange(len(returns))

    # Raw returns
    ax.plot(episodes, returns, alpha=0.3, color="blue", label="Episode Return")

    # Moving average
    if len(returns) >= smoothing:
        ma = np.convolve(returns, np.ones(smoothing) / smoothing, mode="valid")
        ma_x = episodes[smoothing - 1:]
        ax.plot(ma_x, ma, color="blue", linewidth=2, label=f"Moving Avg ({smoothing} eps)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_regret(
    regrets: List[float],
    title: str = "Cumulative Regret",
    figsize: Tuple[int, int] = (10, 6),
    theoretical_bound: Optional[Callable] = None,
    save_path: Optional[str] = None
):
    """
    Plot cumulative regret for bandit algorithms.

    Args:
        regrets: List of cumulative regrets at each step
        title: Plot title
        figsize: Figure size
        theoretical_bound: Optional function T -> theoretical regret bound
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    T = np.arange(1, len(regrets) + 1)
    ax.plot(T, regrets, label="Cumulative Regret", linewidth=2)

    if theoretical_bound is not None:
        bounds = [theoretical_bound(t) for t in T]
        ax.plot(T, bounds, "--", label="Theoretical Bound", linewidth=2)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_value_function(
    value_fn: Callable,
    state_bounds: Tuple[np.ndarray, np.ndarray],
    resolution: int = 50,
    title: str = "Value Function",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot 2D value function heatmap.

    Args:
        value_fn: Function that takes states and returns values
        state_bounds: Tuple of (low, high) bounds for 2D state space
        resolution: Grid resolution
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    low, high = state_bounds

    x = np.linspace(low[0], high[0], resolution)
    y = np.linspace(low[1], high[1], resolution)
    X, Y = np.meshgrid(x, y)

    states = np.stack([X.flatten(), Y.flatten()], axis=1)
    with torch.no_grad():
        values = value_fn(torch.FloatTensor(states)).numpy()
    Z = values.reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Value")

    ax.set_xlabel("State Dim 0")
    ax.set_ylabel("State Dim 1")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_policy(
    policy_fn: Callable,
    state_bounds: Tuple[np.ndarray, np.ndarray],
    resolution: int = 20,
    title: str = "Policy",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot 2D policy as quiver plot (for continuous 2D actions).

    Args:
        policy_fn: Function that takes states and returns actions
        state_bounds: Tuple of (low, high) bounds for 2D state space
        resolution: Grid resolution
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    low, high = state_bounds

    x = np.linspace(low[0], high[0], resolution)
    y = np.linspace(low[1], high[1], resolution)
    X, Y = np.meshgrid(x, y)

    states = np.stack([X.flatten(), Y.flatten()], axis=1)
    with torch.no_grad():
        actions = policy_fn(torch.FloatTensor(states)).numpy()

    U = actions[:, 0].reshape(resolution, resolution)
    V = actions[:, 1].reshape(resolution, resolution) if actions.shape[1] > 1 else np.zeros_like(U)

    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(X, Y, U, V)

    ax.set_xlabel("State Dim 0")
    ax.set_ylabel("State Dim 1")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_trajectory(
    trajectory: List[Dict],
    title: str = "Trajectory",
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None
):
    """
    Visualize a trajectory showing states, actions, and rewards.

    Args:
        trajectory: List of transition dictionaries
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    steps = np.arange(len(trajectory))

    # States
    states = np.array([t["state"] for t in trajectory])
    for i in range(min(states.shape[1], 4)):
        axes[0].plot(steps, states[:, i], label=f"Dim {i}")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("State")
    axes[0].set_title("State Trajectory")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Actions
    actions = np.array([t["action"] for t in trajectory])
    if len(actions.shape) == 1:
        axes[1].plot(steps, actions)
    else:
        for i in range(min(actions.shape[1], 4)):
            axes[1].plot(steps, actions[:, i], label=f"Dim {i}")
        axes[1].legend()
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Action")
    axes[1].set_title("Action Trajectory")
    axes[1].grid(True, alpha=0.3)

    # Rewards
    rewards = np.array([t["reward"] for t in trajectory])
    axes[2].plot(steps, rewards)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Reward")
    axes[2].set_title(f"Rewards (Total: {rewards.sum():.2f})")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ========================
# Evaluation Utilities
# ========================

def evaluate_policy(
    env,
    policy: Callable,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    max_steps: int = 1000,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate a policy on an environment.

    Args:
        env: Gymnasium environment
        policy: Policy function or network
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
        max_steps: Maximum steps per episode
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    returns = []
    lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0

        for step in range(max_steps):
            if render:
                env.render()

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                if hasattr(policy, "get_action"):
                    action, _ = policy.get_action(obs_tensor, deterministic=deterministic)
                    action = action.cpu().numpy()[0]
                else:
                    action = policy(obs_tensor).cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            if done:
                break

        returns.append(episode_return)
        lengths.append(episode_length)

    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "min_return": np.min(returns),
        "max_return": np.max(returns),
        "mean_length": np.mean(lengths),
    }


def record_video(
    env,
    policy: Callable,
    video_path: str,
    n_episodes: int = 1,
    deterministic: bool = True,
    max_steps: int = 1000,
    fps: int = 30,
    device: str = "cpu"
):
    """
    Record video of policy execution.

    Args:
        env: Gymnasium environment (should be render_mode="rgb_array")
        policy: Policy function or network
        video_path: Path to save video
        n_episodes: Number of episodes to record
        deterministic: Whether to use deterministic actions
        max_steps: Maximum steps per episode
        fps: Frames per second
        device: Device to run on
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio[ffmpeg]")
        return

    frames = []

    for _ in range(n_episodes):
        obs, _ = env.reset()

        for step in range(max_steps):
            frame = env.render()
            frames.append(frame)

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                if hasattr(policy, "get_action"):
                    action, _ = policy.get_action(obs_tensor, deterministic=deterministic)
                    action = action.cpu().numpy()[0]
                else:
                    action = policy(obs_tensor).cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Video saved to {video_path}")


# ========================
# Timer Utility
# ========================

class Timer:
    """Simple timer for profiling."""

    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, name: str):
        self.start_times[name] = time.time()

    def stop(self, name: str) -> float:
        elapsed = time.time() - self.start_times[name]
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
        return elapsed

    def get_mean(self, name: str) -> float:
        if name in self.times and len(self.times[name]) > 0:
            return np.mean(self.times[name])
        return 0.0

    def summary(self) -> Dict[str, float]:
        return {name: np.mean(times) for name, times in self.times.items()}
