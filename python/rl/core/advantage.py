"""
Advantage Function Estimation for Reinforcement Learning

This module implements various methods to estimate advantages A(s,a) = Q(s,a) - V(s),
which measure how much better an action is compared to the baseline.

CRITICAL FOR POLICY GRADIENTS:
    Policy gradient theorem:
    ∇_θ J(θ) = E[∇_θ log π(a|s) * A(s,a)]

    The advantage function significantly affects:
    1. Convergence speed (better estimates = faster learning)
    2. Variance (biased estimates = lower variance, unbiased = higher variance)
    3. Sample efficiency (n-step methods trade off bias/variance)

ADVANTAGE DECOMPOSITION:
    A(s,a) = Q(s,a) - V(s) = Q(s,a) - E_a[Q(s,a)]

THEORY:
    The choice of advantage estimator affects the bias-variance tradeoff:

    1. MONTE CARLO (λ=1):
       Very low bias, very high variance
       Use entire episode return as advantage estimate

    2. TD(0) (λ=0):
       High bias, very low variance
       Use single-step TD error as advantage

    3. N-STEP:
       Intermediate bias/variance
       Look ahead n steps for balance

    4. TD(λ) & GAE:
       Exponential moving average of TD errors
       Flexible bias/variance control via λ

REFERENCES:
    - Sutton & Barto (2018), Chapters 9-13
    - Policy Gradient Methods: Sutton et al. (2000)
    - Advantage Estimation: Mnih et al. (2016) - A3C
    - GAE: Schulman et al. (2015) - "High-Dimensional Continuous Control Using GAE"
    - N-step Importance Sampling: Mahmood et al. (2014)
    - Spinning Up on policy gradients
"""

import numpy as np
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod


class BaseAdvantageEstimator(ABC):
    """
    Abstract base class for advantage function estimators.

    All advantage estimators convert raw trajectories into advantage estimates
    for policy gradient computation.
    """

    @abstractmethod
    def compute_advantages(self,
                          states: np.ndarray,
                          rewards: np.ndarray,
                          dones: np.ndarray,
                          next_states: np.ndarray,
                          values: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages and returns for a trajectory.

        Args:
            states: State sequence [T, state_dim]
            rewards: Reward sequence [T]
            dones: Episode termination flags [T]
            next_states: Next state sequence [T, state_dim]
            values: State values V(s) [T+1] (including bootstrap value)

        Returns:
            advantages: A(s_t, a_t) [T] for policy gradient
            returns: G_t target returns [T] for value function
        """
        raise NotImplementedError()


class MonteCarloAdvantageEstimator(BaseAdvantageEstimator):
    """
    Monte Carlo Advantage Estimation.

    Uses full episode returns G_t = ∑_{i=t}^{T-1} γ^i r_{t+i} as advantage.

    PROS:
    - Completely unbiased advantage estimate
    - No function approximation errors (uses actual returns)

    CONS:
    - Very high variance (especially in long episodes)
    - Requires complete episodes
    - Slow convergence

    MATH:
        G_t = ∑_{i=0}^{T-1-t} γ^i r_{t+i}
        A(s_t, a_t) = G_t - V(s_t)  (with value baseline for variance reduction)

    Used in:
    - REINFORCE
    - Vanilla policy gradient
    - Early RL implementations

    NOTE: Usually combined with value baseline V(s) to reduce variance
    """

    def __init__(self, gamma: float = 0.99):
        """
        Initialize Monte Carlo advantage estimator.

        Args:
            gamma: Discount factor (typically 0.99)

        Example:
            >>> estimator = MonteCarloAdvantageEstimator(gamma=0.99)
            >>> adv, ret = estimator.compute_advantages(states, rewards, dones, next_states, values)
        """
        self.gamma = gamma

    def compute_advantages(self,
                          states: np.ndarray,
                          rewards: np.ndarray,
                          dones: np.ndarray,
                          next_states: np.ndarray,
                          values: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Monte Carlo returns and advantages.

        ALGORITHM:
            1. Initialize G_T = 0 (or V(s_T) for bootstrap)
            2. Backwards through trajectory:
               - If done[t]: G_t = r_t (episode ended, no bootstrap)
               - Else: G_t = r_t + γ * G_{t+1}
            3. Advantages: A_t = G_t - V(s_t)

        Args:
            states: [T, state_dim]
            rewards: [T]
            dones: [T]
            next_states: [T, state_dim]
            values: [T+1] V(s) including V(s_T) for bootstrap

        Returns:
            advantages: [T]
            returns: [T]

        COMPUTATIONAL COMPLEXITY:
            O(T) where T is trajectory length
        """
        raise NotImplementedError(
            "Hint: Start from end of trajectory. For each timestep t from T-1 to 0:\n"
            "  if dones[t]: G_t = rewards[t]\n"
            "  else: G_t = rewards[t] + gamma * G_{t+1}\n"
            "Advantages = returns - values[:-1]"
        )


class TDAdvantageEstimator(BaseAdvantageEstimator):
    """
    Temporal Difference (TD) Advantage Estimation.

    Uses single-step TD error as advantage: A_t = r_t + γ V(s_{t+1}) - V(s_t)

    PROS:
    - Low variance (bootstraps to value function)
    - Works with incomplete episodes
    - Fast convergence (less bias = faster updates)

    CONS:
    - Biased by value function approximation error
    - High bias if V is inaccurate

    MATH:
        TD error (δ_t) = r_t + γ V(s_{t+1}) - V(s_t)
        A(s_t, a_t) = δ_t

        This is the generalized advantage estimator with λ=0:
        A_t = ∑_{l=0}^{0} γ^l δ_{t+l} = δ_t

    Used in:
    - SARSA
    - Actor-Critic methods
    - A2C/A3C (with λ < 1)

    BIAS-VARIANCE TRADEOFF:
        Compared to MC:
        - Variance: Much lower (single step)
        - Bias: Higher (depends on V approximation quality)
    """

    def __init__(self, gamma: float = 0.99):
        """
        Initialize TD advantage estimator.

        Args:
            gamma: Discount factor

        Example:
            >>> estimator = TDAdvantageEstimator(gamma=0.99)
            >>> adv, ret = estimator.compute_advantages(states, rewards, dones, next_states, values)
        """
        self.gamma = gamma

    def compute_advantages(self,
                          states: np.ndarray,
                          rewards: np.ndarray,
                          dones: np.ndarray,
                          next_states: np.ndarray,
                          values: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute single-step TD advantages.

        ALGORITHM:
            For each timestep t:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t
            G_t = r_t + γ * V(s_{t+1}) * (1 - done_t)  (TD return)

        Args:
            states: [T, state_dim]
            rewards: [T]
            dones: [T]
            next_states: [T, state_dim]
            values: [T+1]

        Returns:
            advantages: [T] TD errors
            returns: [T] TD targets (for value function training)

        NOTE: Returns are 1-step TD bootstrapped values, not MC returns
        """
        raise NotImplementedError(
            "Hint: Compute TD targets: r_t + gamma * V(s_{t+1}) * (1 - done_t)\n"
            "Advantages = targets - V(s_t)\n"
            "Returns = targets (for value function training)"
        )


class NStepAdvantageEstimator(BaseAdvantageEstimator):
    """
    N-Step Advantage Estimation.

    Intermediate between TD (1-step) and MC (∞-step).
    Look ahead n steps before bootstrapping to value function.

    PROS:
    - Flexible bias-variance control
    - Lower variance than MC, lower bias than TD
    - Works well in practice

    CONS:
    - Still biased (depends on final V estimate)
    - Requires n steps of lookahead

    MATH:
        A_t = G_t^(n) - V(s_t)
        G_t^(n) = ∑_{i=0}^{n-1} γ^i r_{t+i} + γ^n V(s_{t+n})

        where G_t^(n) is n-step return

    SPECIAL CASES:
        n=1: TD advantage (δ_t)
        n=∞: MC advantage (entire episode return)

    Used in:
    - A3C (n=5 or more steps)
    - PPO (can use n-step)
    - Various distributed algorithms

    REFERENCE:
        Peng & Williams (1996)
    """

    def __init__(self, gamma: float = 0.99, n_steps: int = 3):
        """
        Initialize n-step advantage estimator.

        Args:
            gamma: Discount factor
            n_steps: Number of steps to lookahead (default: 3)

        Example:
            >>> estimator = NStepAdvantageEstimator(gamma=0.99, n_steps=5)
            >>> adv, ret = estimator.compute_advantages(states, rewards, dones, next_states, values)
        """
        self.gamma = gamma
        self.n_steps = n_steps

    def compute_advantages(self,
                          states: np.ndarray,
                          rewards: np.ndarray,
                          dones: np.ndarray,
                          next_states: np.ndarray,
                          values: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute n-step advantages.

        ALGORITHM:
            For each timestep t:
            G_t^(n) = ∑_{i=0}^{min(n-1, T-1-t)} γ^i r_{t+i} + γ^{n_steps} V(s_{t+n_steps})
                     (unless episode ends before n steps)
            A_t = G_t^(n) - V(s_t)

        Args:
            states: [T, state_dim]
            rewards: [T]
            dones: [T]
            next_states: [T, state_dim]
            values: [T+1]

        Returns:
            advantages: [T] n-step TD errors
            returns: [T] n-step returns for value training
        """
        raise NotImplementedError(
            "Hint: For each timestep t, accumulate n-step return:\n"
            "  n_step_return = 0\n"
            "  for i in range(min(n_steps, T-t)):\n"
            "    n_step_return += gamma^i * rewards[t+i] * (1 - dones[t+i])\n"
            "    if dones[t+i]: break\n"
            "  n_step_return += gamma^(steps_taken) * V(s_{t+steps_taken})\n"
            "Advantages = n_step_return - V(s_t)"
        )


class TDLambdaAdvantageEstimator(BaseAdvantageEstimator):
    """
    TD(λ) Advantage Estimation using eligibility traces.

    Exponential moving average of n-step returns (n=1,2,3,...).
    Provides smooth interpolation between TD(0) and MC.

    PROS:
    - Flexible λ parameter controls bias-variance
    - Smooth transition: λ→0 (low variance), λ→1 (low bias)
    - Theoretically elegant (backward view via traces)

    CONS:
    - More complex to implement
    - Requires trace variables in on-policy settings
    - GAE is often preferred (more practical)

    MATH - FORWARD VIEW:
        G_t^(λ) = (1-λ) ∑_{n=1}^{∞} λ^{n-1} G_t^(n)

        Weighted average of all n-step returns
        λ weights: n=1 gets (1-λ), n=2 gets (1-λ)λ, n=3 gets (1-λ)λ², etc.

    SPECIAL CASES:
        λ=0: Pure TD (G_t^(1) only)
        λ=1: Monte Carlo (all G_t^(n) weighted equally)
        λ=0.9 or 0.95: Common in practice (good tradeoff)

    Used in:
    - Actor-Critic methods
    - Policy gradient algorithms
    - Historical RL (eligibility traces)

    REFERENCE:
        Sutton (1988): "Learning to Predict by the Methods of Temporal Differences"
        Sutton & Barto (2018): Chapter 12
    """

    def __init__(self, gamma: float = 0.99, lambda_coeff: float = 0.95):
        """
        Initialize TD(λ) advantage estimator.

        Args:
            gamma: Discount factor
            lambda_coeff: TD(λ) parameter (0 to 1)
                         0: pure TD, 1: pure MC

        Example:
            >>> estimator = TDLambdaAdvantageEstimator(gamma=0.99, lambda_coeff=0.95)
            >>> adv, ret = estimator.compute_advantages(states, rewards, dones, next_states, values)
        """
        self.gamma = gamma
        self.lambda_coeff = lambda_coeff

    def compute_advantages(self,
                          states: np.ndarray,
                          rewards: np.ndarray,
                          dones: np.ndarray,
                          next_states: np.ndarray,
                          values: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute TD(λ) advantages.

        ALGORITHM (Efficient Backward View):
            Initialize advantages[T] = 0
            For t from T-1 down to 0:
              δ_t = r_t + γ V(s_{t+1}) - V(s_t)
              advantages[t] = δ_t + γ λ advantages[t+1]

            This efficiently computes: G_t^(λ) = ∑_{l=0}^{T-1-t} (γλ)^l δ_{t+l}

        Args:
            states: [T, state_dim]
            rewards: [T]
            dones: [T]
            next_states: [T, state_dim]
            values: [T+1]

        Returns:
            advantages: [T] TD(λ) advantages
            returns: [T] TD(λ) returns for value training
        """
        raise NotImplementedError(
            "Hint: First compute TD errors (deltas):\n"
            "  delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)\n"
            "Then backwards accumulate with eligibility traces:\n"
            "  adv[T-1] = delta[T-1]\n"
            "  for t in range(T-2, -1, -1):\n"
            "    adv[t] = delta[t] + gamma * lambda_coeff * adv[t+1]\n"
            "Returns = advantages + values[:-1]"
        )


class GeneralizedAdvantageEstimation(BaseAdvantageEstimator):
    r"""
    Generalized Advantage Estimation (GAE) - THE MOST IMPORTANT METHOD FOR MODERN RL

    This is the primary advantage estimator for:
    - PPO (Proximal Policy Optimization)
    - TRPO (Trust Region Policy Optimization)
    - A2C/A3C variants
    - Most modern policy gradient algorithms

    GAE provides a practical solution to the bias-variance tradeoff through
    exponential smoothing of advantage estimates.

    CRITICAL FOR ROBOTICS AND PPO:
    ==========================
    GAE is the foundation of modern practical RL for robotics.
    Understanding and implementing GAE correctly is ESSENTIAL.

    PROS (Why GAE dominates modern RL):
    - Smooth bias-variance tradeoff via λ parameter
    - Efficient computation (single backward pass)
    - Theoretically grounded (exponential trace weighting)
    - Practical performance (better than alternatives)
    - Works well with neural networks
    - Reduced variance compared to MC
    - Reduced bias compared to pure TD

    MATH - DETAILED DERIVATION:
    ==========================

    1. BASIC ADVANTAGE DEFINITION:
       A(s_t, a_t) = Q(s_t, a_t) - V(s_t)

    2. GAE USES TD RESIDUALS (δ_t):
       δ_t^V = r_t + γ V(s_{t+1}) - V(s_t)  (TD error for value function)

       The key insight: Advantage can be expressed as sum of TD residuals
       A(s_t, a_t) = ∑_{l=0}^{∞} (γλ)^l δ_{t+l}^V

       This is because:
       Q(s_t, a_t) - V(s_t) = [r_t + γ Q(s_{t+1}, a_{t+1}) - V(s_t)]
                             = [r_t + γ V(s_{t+1}) - V(s_t)] + γ[Q(s_{t+1}, a_{t+1}) - V(s_{t+1})]
                             = δ_t^V + γ A(s_{t+1}, a_{t+1})

    3. FINITE HORIZON GAE (Practical):
       Â_t = ∑_{l=0}^{T-1-t} (γλ)^l δ_{t+l}^V
       = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...

       where T is trajectory length

    4. EFFICIENT COMPUTATION (Backward Recursion):
       Starting from end of trajectory:
       Â_T = 0  (no future advantages beyond episode end)
       Â_t = δ_t + (γλ) Â_{t+1}  for t = T-1, ..., 0

       This provides:
       - O(T) computation complexity (single pass)
       - Numerically stable (avoids long exponential sums)
       - Works naturally with episode boundaries

    5. VALUE TARGET (Generalized Return):
       G_t = Â_t + V(s_t)

       This is the target for value function training.
       The generalized return combines advantage + baseline.

    PARAMETER TUNING (λ vs γ):
    ===========================

    λ (lambda) - GAE parameter:
    - λ = 0: Uses only 1-step TD error (low variance, high bias)
             Â_t = δ_t (pure TD(0))
    - λ = 1: Uses full trajectory (MC estimation, high variance, low bias)
             Â_t = ∑_l δ_{t+l} (cumulative TD errors ≈ MC return)
    - λ ∈ (0, 1): Smooth tradeoff (RECOMMENDED in practice)
    - Common values: λ = 0.95 or 0.98

    γ (gamma) - Discount factor:
    - Controls how much future matters
    - Separate from λ (both needed!)
    - Typical: γ = 0.99 for continuing tasks, γ = 0.98 for episodic

    INTERPRETATION:
    - Large λ (0.95-1.0): Trust rewards, collect more data, higher variance
    - Small λ (0.0-0.5): Trust value function, faster learning, lower variance
    - For robotics: λ ∈ [0.9, 0.99] works well

    COMPARISON WITH OTHER METHODS:
    ==============================

    1. Monte Carlo (λ=1):
       Â_t = G_t - V(s_t)  where G_t = ∑_{i=t}^{T} γ^i r_i
       - Unbiased but very high variance
       - Slow convergence

    2. TD(0) (λ=0):
       Â_t = r_t + γ V(s_{t+1}) - V(s_t)
       - Low variance but biased
       - Depends heavily on V accuracy

    3. N-step:
       Â_t = [∑_{i=0}^{n-1} γ^i r_{t+i} + γ^n V(s_{t+n})] - V(s_t)
       - Fixed compromise between MC and TD
       - Less flexible than GAE

    4. GAE:
       Â_t = ∑_{l=0}^{T-1-t} (γλ)^l δ_{t+l}
       - Smooth bias-variance control
       - Exponential weighting (natural decay)
       - BEST for policy gradients

    EPISODE BOUNDARIES (CRITICAL):
    ==============================

    At episode end (done=True), V(s_T) = 0 (no bootstrap).
    When continuing: V(s_T) = predicted value (bootstrap)

    Implementation:
    - If done[t] = True: Set next_value to 0 (episode ended)
    - Otherwise: Use V(s_{t+1}) prediction

    REFERENCES:
    ===========
    - Schulman et al. (2015): "High-Dimensional Continuous Control Using GAE"
      https://arxiv.org/abs/1506.02438
    - OpenAI Spinning Up
    - PPO paper: Schulman et al. (2017)
    - Implementation examples: OpenAI baselines, Stable Baselines
    """

    def __init__(self, gamma: float = 0.99, lambda_coeff: float = 0.95):
        """
        Initialize Generalized Advantage Estimation.

        Args:
            gamma: Discount factor (default: 0.99)
                  Controls temporal scope of returns
            lambda_coeff: GAE λ parameter (default: 0.95)
                         Controls bias-variance tradeoff
                         0 = low variance/high bias (TD)
                         1 = high variance/low bias (MC)

        TYPICAL VALUES FOR DIFFERENT TASKS:
        - Robotics continuous control: γ=0.99, λ=0.95
        - Atari games: γ=0.99, λ=0.98
        - MuJoCo locomotion: γ=0.99, λ=0.95
        - Short-horizon tasks: γ=0.99, λ=0.90

        Example:
            >>> # PPO-style GAE
            >>> gae = GeneralizedAdvantageEstimation(gamma=0.99, lambda_coeff=0.95)
            >>> advantages, returns = gae.compute_advantages(
            ...     states, rewards, dones, next_states, values
            ... )
            >>> # Use for policy gradient: ∇_θ J = E[∇ log π * A]
            >>> # Use for value training: minimize (V - G_t)²
        """
        self.gamma = gamma
        self.lambda_coeff = lambda_coeff

    def compute_advantages(self,
                          states: np.ndarray,
                          rewards: np.ndarray,
                          dones: np.ndarray,
                          next_states: np.ndarray,
                          values: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute GAE advantages and returns.

        INPUTS:
        -------
        states: [T, state_dim]
            State sequence of trajectory
        rewards: [T]
            Reward sequence
        dones: [T]
            Episode termination flags (True if episode ended at step t)
        next_states: [T, state_dim]
            Next state sequence (optional - used for bootstrap)
        values: [T+1]
            State value estimates V(s_t) for all timesteps
            IMPORTANT: values[T] is bootstrap value for next state (or 0 if done)

        ALGORITHM:
        ----------
        Step 1: Compute TD residuals (δ)
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)

            Handle episode boundaries:
            - If done_t = True: δ_t = r_t - V(s_t) (no bootstrap)
            - Otherwise: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

        Step 2: Compute advantages backward
            Â_T = 0  (no future advantages)
            Â_t = δ_t + γ * λ * Â_{t+1}  for t = T-1, ..., 0

            This implements:
            Â_t = ∑_{l=0}^{T-1-t} (γλ)^l * δ_{t+l}

        Step 3: Compute returns (targets for value function)
            G_t = Â_t + V(s_t)

            This is the generalized return combining advantage + baseline.

        OUTPUTS:
        --------
        advantages: [T]
            Generalized advantage estimates (for policy gradient)
        returns: [T]
            Generalized returns (for value function training)

        IMPLEMENTATION NOTES:
        - Process values as: [V(s_0), V(s_1), ..., V(s_T)]
        - values[T] might be 0 if done, or final state value if continuing
        - Advantages should be normalized before use in training
        - Returns are raw, not normalized

        NUMERICAL STABILITY:
        - Use (1 - dones) to mask out bootstraps at episode ends
        - Accumulate backward to avoid numerical instability
        - Consider clipping extreme advantages in some algorithms

        RETURN SHAPES:
            advantages: numpy array of shape [T]
            returns: numpy array of shape [T]
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        raise NotImplementedError(
            "CRITICAL IMPLEMENTATION:\n"
            "1. Compute TD residuals:\n"
            "   deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]\n\n"
            "2. Compute advantages backward:\n"
            "   advantages[T-1] = deltas[T-1]\n"
            "   for t in range(T-2, -1, -1):\n"
            "     advantages[t] = deltas[t] + (gamma * lambda_coeff) * advantages[t+1]\n\n"
            "3. Compute returns:\n"
            "   returns = advantages + values[:-1]\n\n"
            "NOTE: This single backward pass gives both advantages and returns efficiently!"
        )


def normalize_advantages(advantages: np.ndarray,
                        epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize advantages for stable training.

    MOTIVATION:
        Large advantage magnitudes can destabilize learning.
        Normalization helps with:
        - Numerical stability (prevents divergence)
        - Learning rate independence
        - Consistent gradient magnitudes

    MATH:
        A_normalized = (A - mean(A)) / (std(A) + ε)

        where ε prevents division by zero

    Args:
        advantages: Advantage estimates [batch_size or trajectory_length]
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        normalized_advantages: Zero-mean, unit-variance advantages

    IMPORTANT:
        - Normalize within batch/trajectory, not globally
        - Usually done before computing policy loss
        - Not typically done for value function targets
        - Can improve convergence significantly
    """
    raise NotImplementedError(
        "Hint: Compute mean and std of advantages, "
        "return (advantages - mean) / (std + epsilon)"
    )


def compute_returns(rewards: np.ndarray,
                   gamma: float = 0.99) -> np.ndarray:
    """
    Compute discounted cumulative returns (Monte Carlo targets).

    MATH:
        G_t = ∑_{i=0}^{T-1-t} γ^i * r_{t+i}

    Efficient backward computation:
        G_T = 0
        G_t = r_t + γ * G_{t+1}

    Args:
        rewards: [T]
        gamma: Discount factor

    Returns:
        returns: [T] discounted cumulative rewards
    """
    raise NotImplementedError(
        "Hint: Compute backward: G[T-1] = rewards[T-1], "
        "G[t] = rewards[t] + gamma * G[t+1]"
    )


def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
                dones: np.ndarray, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation.

    Convenience function wrapping GeneralizedAdvantageEstimation class.

    Args:
        rewards: Array of rewards
        values: Array of value estimates V(s)
        next_values: Array of next state values V(s')
        dones: Array of done flags
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        Array of advantage estimates
    """
    raise NotImplementedError(
        "TODO: Implement GAE computation\\n"
        "Hint: Use GeneralizedAdvantageEstimation class or implement directly"
    )

