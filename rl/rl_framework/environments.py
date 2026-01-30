"""
Environment Wrappers and Utilities
===================================

Provides:
- Gymnasium environment wrappers
- Custom bandit environments
- Goal-conditioned environment wrappers
- Environment vectorization
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Callable
from collections import deque


class NormalizedEnv(gym.Wrapper):
    """
    Normalizes observations and rewards using running statistics.
    Essential for stable training of policy gradient algorithms.
    """

    def __init__(
        self,
        env: gym.Env,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8
    ):
        super().__init__(env)
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        # Running statistics for observations
        obs_shape = env.observation_space.shape
        self.obs_mean = np.zeros(obs_shape)
        self.obs_var = np.ones(obs_shape)
        self.obs_count = epsilon

        # Running statistics for rewards
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = epsilon
        self.returns = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Update return statistics
        self.returns = self.returns * self.gamma + reward

        if self.normalize_reward:
            self._update_reward_stats(self.returns)
            reward = self._normalize_reward(reward)

        if done:
            self.returns = 0.0

        if self.normalize_obs:
            self._update_obs_stats(obs)
            obs = self._normalize_obs(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.returns = 0.0

        if self.normalize_obs:
            self._update_obs_stats(obs)
            obs = self._normalize_obs(obs)

        return obs, info

    def _update_obs_stats(self, obs: np.ndarray):
        """Update running mean and variance for observations."""
        batch_mean = obs
        batch_var = np.zeros_like(obs)
        batch_count = 1

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count

        self.obs_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.obs_count * batch_count / total_count
        self.obs_var = M2 / total_count
        self.obs_count = total_count

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        obs_normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(obs_normalized, -self.clip_obs, self.clip_obs)

    def _update_reward_stats(self, ret: float):
        """Update running mean and variance for returns."""
        delta = ret - self.ret_mean
        self.ret_count += 1
        self.ret_mean += delta / self.ret_count
        delta2 = ret - self.ret_mean
        self.ret_var += (delta * delta2 - self.ret_var) / self.ret_count

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using return variance."""
        reward_normalized = reward / np.sqrt(self.ret_var + self.epsilon)
        return np.clip(reward_normalized, -self.clip_reward, self.clip_reward)


class FrameStack(gym.Wrapper):
    """Stack the last n frames for temporal information."""

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Update observation space
        obs_shape = env.observation_space.shape
        low = np.repeat(env.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = gym.spaces.Box(low=low.flatten(), high=high.flatten(), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)


class EpisodeMonitor(gym.Wrapper):
    """Monitor episode statistics (returns, lengths)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []
        self.current_return = 0.0
        self.current_length = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.current_return += reward
        self.current_length += 1

        if done:
            info["episode"] = {
                "r": self.current_return,
                "l": self.current_length
            }
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            self.current_return = 0.0
            self.current_length = 0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_return = 0.0
        self.current_length = 0
        return self.env.reset(**kwargs)

    def get_episode_stats(self) -> Dict[str, List[float]]:
        """Get episode statistics."""
        return {
            "returns": self.episode_returns.copy(),
            "lengths": self.episode_lengths.copy()
        }


# ========================
# Bandit Environments
# ========================

class MultiArmedBandit:
    """
    Multi-Armed Bandit environment.
    Each arm has a fixed (unknown) reward distribution.
    """

    def __init__(
        self,
        n_arms: int = 10,
        reward_type: str = "bernoulli",
        true_means: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        self.n_arms = n_arms
        self.reward_type = reward_type
        self.rng = np.random.default_rng(seed)

        if true_means is not None:
            self.true_means = true_means
        else:
            self.true_means = self.rng.uniform(0, 1, size=n_arms)

        self.optimal_arm = np.argmax(self.true_means)
        self.optimal_reward = self.true_means[self.optimal_arm]
        self.t = 0
        self.total_regret = 0.0

    def pull(self, arm: int) -> float:
        """Pull an arm and receive a reward."""
        mean = self.true_means[arm]

        if self.reward_type == "bernoulli":
            reward = float(self.rng.random() < mean)
        elif self.reward_type == "gaussian":
            reward = self.rng.normal(mean, 1.0)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        # Track regret
        self.t += 1
        self.total_regret += self.optimal_reward - mean

        return reward

    def get_regret(self) -> float:
        """Get cumulative regret."""
        return self.total_regret

    def reset(self):
        """Reset the bandit."""
        self.t = 0
        self.total_regret = 0.0


class ContextualBandit:
    """
    Contextual Bandit environment with linear reward structure.
    Reward = theta_arm^T @ context + noise
    """

    def __init__(
        self,
        n_arms: int = 5,
        context_dim: int = 10,
        noise_std: float = 0.1,
        seed: Optional[int] = None
    ):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # True parameters (unknown to agent)
        self.theta = self.rng.randn(n_arms, context_dim)
        # Normalize
        self.theta = self.theta / np.linalg.norm(self.theta, axis=1, keepdims=True)

        self.t = 0
        self.total_regret = 0.0
        self.current_context = None

    def get_context(self) -> np.ndarray:
        """Get a new context vector."""
        self.current_context = self.rng.randn(self.context_dim)
        self.current_context = self.current_context / np.linalg.norm(self.current_context)
        return self.current_context

    def pull(self, arm: int) -> float:
        """Pull an arm given the current context."""
        if self.current_context is None:
            raise ValueError("Must call get_context() first")

        # Compute expected rewards for all arms
        expected_rewards = self.theta @ self.current_context
        optimal_arm = np.argmax(expected_rewards)
        optimal_reward = expected_rewards[optimal_arm]

        # Get actual reward with noise
        reward = expected_rewards[arm] + self.rng.normal(0, self.noise_std)

        # Track regret
        self.t += 1
        self.total_regret += optimal_reward - expected_rewards[arm]

        self.current_context = None  # Reset context
        return reward

    def get_regret(self) -> float:
        return self.total_regret

    def reset(self):
        self.t = 0
        self.total_regret = 0.0
        self.current_context = None


# ========================
# Goal-Conditioned Environments
# ========================

class GoalConditionedWrapper(gym.Wrapper):
    """
    Wrapper to convert standard environments to goal-conditioned.
    Appends goal to observation and provides sparse rewards.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_fn: Callable,
        achieved_goal_fn: Callable,
        reward_fn: Callable,
        goal_dim: int
    ):
        super().__init__(env)
        self.goal_fn = goal_fn  # Function to sample goals
        self.achieved_goal_fn = achieved_goal_fn  # Function to extract achieved goal from state
        self.reward_fn = reward_fn  # Function(achieved_goal, desired_goal) -> reward

        # Update observation space to include goal
        obs_dim = env.observation_space.shape[0]
        new_dim = obs_dim + goal_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )

        self.goal = None
        self.goal_dim = goal_dim

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.goal = self.goal_fn()

        info["desired_goal"] = self.goal
        info["achieved_goal"] = self.achieved_goal_fn(obs)

        return np.concatenate([obs, self.goal]), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        achieved_goal = self.achieved_goal_fn(obs)
        reward = self.reward_fn(achieved_goal, self.goal)

        info["desired_goal"] = self.goal
        info["achieved_goal"] = achieved_goal

        return np.concatenate([obs, self.goal]), reward, terminated, truncated, info


# ========================
# Environment Creation Utilities
# ========================

def make_env(
    env_id: str,
    seed: int = 0,
    normalize: bool = True,
    monitor: bool = True,
    **kwargs
) -> gym.Env:
    """
    Create and wrap a Gymnasium environment.

    Args:
        env_id: Gymnasium environment ID (e.g., "CartPole-v1", "HalfCheetah-v4")
        seed: Random seed
        normalize: Whether to normalize observations and rewards
        monitor: Whether to add episode monitoring

    Returns:
        Wrapped environment
    """
    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)

    if monitor:
        env = EpisodeMonitor(env)

    if normalize:
        env = NormalizedEnv(env)

    return env


def make_vec_envs(
    env_id: str,
    n_envs: int = 4,
    seed: int = 0,
    normalize: bool = True,
    **kwargs
) -> gym.vector.VectorEnv:
    """
    Create vectorized environments for parallel data collection.

    Args:
        env_id: Gymnasium environment ID
        n_envs: Number of parallel environments
        seed: Base random seed
        normalize: Whether to normalize observations and rewards

    Returns:
        Vectorized environment
    """
    def make_env_fn(env_seed):
        def _init():
            env = gym.make(env_id, **kwargs)
            env.reset(seed=env_seed)
            if normalize:
                env = NormalizedEnv(env)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv([make_env_fn(seed + i) for i in range(n_envs)])
    return envs


# ========================
# Common Environment Configurations
# ========================

CLASSIC_CONTROL_ENVS = [
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "Acrobot-v1",
    "LunarLander-v2",
    "LunarLanderContinuous-v2",
]

MUJOCO_ENVS = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
    "Swimmer-v4",
    "Reacher-v4",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
]

ATARI_ENVS = [
    "ALE/Breakout-v5",
    "ALE/Pong-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Asteroids-v5",
]


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """Get environment information."""
    obs_space = env.observation_space
    act_space = env.action_space

    info = {
        "observation_space": obs_space,
        "action_space": act_space,
        "is_discrete": isinstance(act_space, gym.spaces.Discrete),
    }

    if isinstance(obs_space, gym.spaces.Box):
        info["state_dim"] = obs_space.shape[0]
    elif isinstance(obs_space, gym.spaces.Discrete):
        info["state_dim"] = obs_space.n

    if isinstance(act_space, gym.spaces.Box):
        info["action_dim"] = act_space.shape[0]
        info["action_low"] = act_space.low
        info["action_high"] = act_space.high
    elif isinstance(act_space, gym.spaces.Discrete):
        info["action_dim"] = act_space.n

    return info
