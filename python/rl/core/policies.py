"""
Policy Representations for Reinforcement Learning

This module defines policy classes that map states to actions in various RL algorithms.
Policies are central to RL - they represent the agent's decision-making strategy.

THEORY:
    A policy π is a mapping from states to actions (or action distributions):
    π: S → A (deterministic) or π: S → P(A|S) (stochastic)

    The objective is typically to find the optimal policy π* that maximizes:
    J(π) = E_π[∑_t γ^t r_t]  (expected discounted return)

    Different policy classes suit different algorithms:
    - Discrete action spaces: Epsilon-greedy, softmax, categorical
    - Continuous action spaces: Gaussian, squashed gaussian, deterministic
    - Special cases: Deterministic for deterministic PG algorithms

MATH:
    1. Epsilon-Greedy (discrete):
       π(a|s) = 1 - ε + ε/|A|  if a = argmax_a Q(s,a)
       π(a|s) = ε/|A|           otherwise

    2. Softmax/Categorical (discrete):
       π(a|s) = exp(θ_a) / ∑_a' exp(θ_a')  (Boltzmann distribution)

    3. Gaussian (continuous):
       π(a|s) = N(μ(s), σ(s)^2)  where μ, σ are neural networks

    4. Squashed Gaussian (continuous, bounded [-1,1]):
       a = tanh(μ + σ * ε)  where ε ~ N(0,1)
       With log-det-jacobian correction for density

    5. Deterministic (continuous):
       π(a|s) = δ(a - μ(s))  deterministic policy for DDPG/TD3

REFERENCES:
    - Sutton & Barto (2018), Chapter 2 & 13
    - Spinning Up guide: https://spinningup.openai.com/
    - Gaussian policies: Chua et al., SAC (2018)
    - Squashed Gaussian: Haarnoja et al., SAC (2018)
"""

import numpy as np
from typing import Tuple, Optional, Union, Any
from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """
    Abstract base class for all policies.

    A policy determines the agent's behavior by mapping states to actions.
    """

    @abstractmethod
    def sample(self, state: Union[np.ndarray]) -> np.ndarray:
        """
        Sample an action from the policy given a state.

        Args:
            state: Current state (single sample, not batch)

        Returns:
            action: Sampled action
        """
        raise NotImplementedError("Subclasses must implement sample()")

    @abstractmethod
    def get_action_dist(self, states: np.ndarray) -> Any:
        """
        Get the action distribution over a batch of states.

        For discrete policies, returns logits or probabilities.
        For continuous policies, returns distribution parameters (mean, std).

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            Distribution or distribution parameters
        """
        raise NotImplementedError("Subclasses must implement get_action_dist()")


class EpsilonGreedyPolicy(BasePolicy):
    """
    Epsilon-Greedy Policy for discrete action spaces.

    With probability ε, take a random action (exploration).
    With probability 1-ε, take the greedy action (exploitation).

    Commonly used in:
    - Q-learning
    - DQN
    - Double Q-learning

    MATH:
        π(a|s) = { 1 - ε + ε/|A|,  if a = argmax_a Q(s,a)
                 { ε/|A|,           otherwise
    """

    def __init__(self,
                 q_values: np.ndarray,
                 epsilon: float = 0.1,
                 num_actions: Optional[int] = None):
        """
        Initialize epsilon-greedy policy.

        Args:
            q_values: Q(s,a) values of shape [num_actions] for single state
            epsilon: Exploration probability (default: 0.1)
            num_actions: Number of actions (inferred from q_values if None)

        Example:
            >>> q_values = np.array([1.5, 2.0, 0.5, 1.2])  # Q-values for 4 actions
            >>> policy = EpsilonGreedyPolicy(q_values, epsilon=0.1)
            >>> action = policy.sample(q_values)
        """
        self.q_values = q_values
        self.epsilon = epsilon
        self.num_actions = num_actions or len(q_values)

    def sample(self, state: Union[np.ndarray]) -> int:
        """
        Sample action using epsilon-greedy strategy.

        Args:
            state: Current state (can be ignored if q_values passed to __init__)

        Returns:
            action: Integer action index
        """
        raise NotImplementedError(
            "Hint: With probability epsilon, return random action from "
            "np.random.choice(self.num_actions). Otherwise return greedy action "
            "using np.argmax(self.q_values)"
        )

    def get_action_dist(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get epsilon-greedy action distribution (probabilities).

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            logits: Log probabilities [batch_size, num_actions]
            probs: Action probabilities [batch_size, num_actions]
        """
        raise NotImplementedError(
            "Hint: Create probability distribution where greedy action "
            "has probability (1 - eps + eps/|A|) and non-greedy "
            "actions have probability eps/|A|. Use np.log() for logits."
        )


class SoftmaxPolicy(BasePolicy):
    """
    Softmax/Categorical Policy for discrete action spaces.

    Also known as Boltzmann exploration. Action probabilities follow
    a softmax distribution over action values or logits.

    Commonly used in:
    - Actor-critic methods
    - Policy gradient methods
    - Maximum entropy RL (SAC discrete)

    MATH:
        π(a|s) = exp(h(s,a)) / ∑_a' exp(h(s,a'))

        where h(s,a) are logits (typically output of neural network)
        or h(s,a) = Q(s,a)/T where T is temperature
    """

    def __init__(self,
                 logits: np.ndarray,
                 temperature: float = 1.0):
        """
        Initialize softmax policy.

        Args:
            logits: Raw action logits/preferences [num_actions]
            temperature: Softmax temperature (higher = more uniform)
                        T → 0: deterministic, T → ∞: uniform

        Example:
            >>> logits = np.array([2.0, 0.5, 1.0])
            >>> policy = SoftmaxPolicy(logits, temperature=0.5)
            >>> action = policy.sample(None)
        """
        self.logits = logits
        self.temperature = temperature

    def sample(self, state: Optional[Union[np.ndarray]] = None) -> int:
        """
        Sample action from softmax distribution.

        Args:
            state: Not used (logits from init)

        Returns:
            action: Integer action index
        """
        raise NotImplementedError(
            "Hint: Scale logits by temperature, apply softmax normalization, "
            "then sample action using np.random.choice() with softmax probabilities"
        )

    def get_action_dist(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get softmax action distribution.

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            logits: Raw logits [batch_size, num_actions]
            probs: Softmax probabilities [batch_size, num_actions]
        """
        raise NotImplementedError(
            "Hint: Divide logits by temperature, use softmax normalization "
            "along action dimension to get probabilities"
        )


class GaussianPolicy(BasePolicy):
    """
    Gaussian Policy for continuous action spaces.

    Actions are sampled from a Gaussian distribution with mean and
    standard deviation predicted by neural networks.

    Commonly used in:
    - A3C (continuous actions)
    - PPO (continuous control)
    - TRPO
    - SAC

    MATH:
        a ~ π(·|s) = N(μ(s), σ(s)^2)

        where μ(s) and σ(s) are outputs of neural networks

        log π(a|s) = -0.5 * log(2π) - 0.5 * log(σ^2) - (a - μ)^2 / (2σ^2)
                   = -0.5 * log(2π σ^2) - (a - μ)^2 / (2σ^2)
    """

    def __init__(self,
                 mean: Union[np.ndarray],
                 std: Union[np.ndarray],
                 learn_std: bool = True):
        """
        Initialize Gaussian policy.

        Args:
            mean: Mean of Gaussian [action_dim] or [batch_size, action_dim]
            std: Standard deviation [action_dim] or [batch_size, action_dim]
                 Usually constrained to positive range (e.g., exp(logstd))
            learn_std: Whether std is learnable (typically True)

        Example:
            >>> mean = np.random.randn(1, 6)  # mean for batch_size=1, action_dim=6
            >>> std = np.ones(1, 6)
            >>> policy = GaussianPolicy(mean, std)
            >>> action, logprob = policy.sample_with_logprob(None)
        """
        self.mean = mean if isinstance(mean, np.ndarray) else np.array(mean)
        self.std = std if isinstance(std, np.ndarray) else np.array(std)
        self.learn_std = learn_std

    def sample(self, state: Optional[Union[np.ndarray]] = None) -> np.ndarray:
        """
        Sample action from Gaussian distribution.

        Args:
            state: Not used (mean/std from init)

        Returns:
            action: Sampled action [action_dim]
        """
        raise NotImplementedError(
            "Hint: Sample epsilon ~ N(0, 1), return action = mean + std * epsilon"
        )

    def sample_with_logprob(self,
                           state: Optional[Union[np.ndarray]] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample action and compute log probability.

        Returns:
            action: Sampled action [batch_size, action_dim]
            logprob: Log probability [batch_size]

        IMPORTANT: This is crucial for policy gradient methods!
        """
        raise NotImplementedError(
            "Hint: Sample action, then compute log probability using "
            "Gaussian PDF formula: -0.5*log(2π) - log(std) - (a-mean)^2/(2*std^2)"
        )

    def get_action_dist(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Gaussian distribution parameters.

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            mean: Mean [batch_size, action_dim]
            std: Std [batch_size, action_dim]
        """
        return self.mean, self.std


class SquashedGaussianPolicy(BasePolicy):
    """
    Squashed Gaussian Policy (with tanh squashing) for bounded continuous actions.

    Ensures actions are bounded to [-1, 1] using tanh squashing.
    Includes log-det-jacobian correction for proper probability density.

    Commonly used in:
    - SAC (Soft Actor-Critic)
    - Bounded continuous control problems

    MATH:
        a = tanh(μ + σ * ε)  where ε ~ N(0,1)

        log π(a|s) = log π_u(u|s) - ∑_i log(1 - tanh(u_i)^2)

        where u = μ + σ * ε is the unbounded action (before tanh)

        The log-det-jacobian term corrects for the tanh squashing:
        log|∂a/∂u| = -∑_i log(1 - tanh(u_i)^2)
    """

    def __init__(self,
                 mean: Union[np.ndarray],
                 std: Union[np.ndarray],
                 action_scale: float = 1.0):
        """
        Initialize squashed Gaussian policy.

        Args:
            mean: Unbounded mean [action_dim] or [batch_size, action_dim]
            std: Unbounded std [action_dim] or [batch_size, action_dim]
            action_scale: Scale factor for action space (default: 1.0 for [-1,1])
                         If action space is [-scale, scale], use this parameter

        Example:
            >>> mean = np.random.randn(1, 6)
            >>> std = np.ones(1, 6)
            >>> policy = SquashedGaussianPolicy(mean, std, action_scale=1.0)
            >>> action, logprob = policy.sample_with_logprob(None)
            >>> # action will be in [-1, 1]
        """
        self.mean = mean if isinstance(mean, np.ndarray) else np.array(mean)
        self.std = std if isinstance(std, np.ndarray) else np.array(std)
        self.action_scale = action_scale

    def sample(self, state: Optional[Union[np.ndarray]] = None) -> np.ndarray:
        """
        Sample bounded action using tanh squashing.

        Returns:
            action: Sampled action in [-action_scale, action_scale]
        """
        raise NotImplementedError(
            "Hint: Sample unbounded action u = mean + std * eps where eps ~ N(0,1), "
            "then apply tanh squashing: a = tanh(u), return a * action_scale"
        )

    def sample_with_logprob(self,
                           state: Optional[Union[np.ndarray]] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample bounded action and compute log probability with jacobian correction.

        CRITICAL: Must include log-det-jacobian for correct density!

        Returns:
            action: Squashed action [batch_size, action_dim]
            logprob: Log probability with jacobian correction [batch_size]
        """
        raise NotImplementedError(
            "Hint: Sample u = mean + std * eps. Compute unbounded log prob from "
            "Gaussian. Apply tanh squashing: a = tanh(u). Subtract jacobian term: "
            "log(1 - tanh(u)^2) for each action dimension. Sum jacobian terms "
            "and subtract from log prob. Scale by action_scale."
        )

    def get_action_dist(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get squashed Gaussian parameters.

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            mean: Unbounded mean [batch_size, action_dim]
            std: Unbounded std [batch_size, action_dim]
        """
        return self.mean, self.std


class DeterministicPolicy(BasePolicy):
    """
    Deterministic Policy for continuous action spaces.

    Maps states directly to actions without stochasticity.
    Used in deterministic policy gradient algorithms (DPG, DDPG, TD3).

    Commonly used in:
    - DDPG (Deep Deterministic Policy Gradient)
    - TD3 (Twin Delayed DDPG)
    - Deterministic PG

    MATH:
        a = μ(s)  (deterministic)

        ∇_θ J(θ) = E[∇_a Q(s,a)|_{a=μ(s)} * ∇_θ μ(s)]

        (Policy gradient flows through Q-function)
    """

    def __init__(self, mean: Union[np.ndarray]):
        """
        Initialize deterministic policy.

        Args:
            mean: Deterministic action μ(s) [action_dim] or [batch_size, action_dim]

        Example:
            >>> mean = np.random.randn(1, 6)  # deterministic action
            >>> policy = DeterministicPolicy(mean)
            >>> action = policy.sample(None)
        """
        self.mean = mean if isinstance(mean, np.ndarray) else np.array(mean)

    def sample(self, state: Optional[Union[np.ndarray]] = None) -> np.ndarray:
        """
        Return deterministic action (no sampling).

        Returns:
            action: Deterministic action [action_dim]
        """
        raise NotImplementedError(
            "Hint: Simply return self.mean as numpy array"
        )

    def get_action_dist(self, states: np.ndarray) -> np.ndarray:
        """
        Get deterministic action (no distribution).

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            action: Deterministic actions [batch_size, action_dim]
        """
        return self.mean


# Utility function for policy evaluation
def compute_entropy(logprobs: np.ndarray) -> np.ndarray:
    """
    Compute policy entropy from log probabilities.

    Higher entropy = more exploration, lower entropy = more exploitation

    MATH:
        H(π) = -E[log π(a|s)] = -∑_a π(a|s) log π(a|s)

    Args:
        logprobs: Log probabilities [batch_size, num_actions] or [batch_size]

    Returns:
        entropy: Policy entropy [batch_size] or scalar

    Example:
        >>> logprobs = np.array([[-1.0, -2.0], [-0.5, -1.5]])
        >>> entropy = compute_entropy(logprobs)
    """
    raise NotImplementedError(
        "Hint: Convert logprobs to probs using np.exp(), then compute "
        "-sum(probs * logprobs, axis=-1)"
    )
