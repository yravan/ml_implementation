"""
Epsilon-Greedy Exploration Strategy for Multi-Armed Bandits

Implementation Status: Stub with comprehensive documentation
Complexity: Beginner to Intermediate
Prerequisites: NumPy, basic probability theory

The epsilon-greedy algorithm is the simplest and most intuitive exploration strategy
for multi-armed bandits. It serves as a baseline for understanding exploration-exploitation
tradeoffs and is widely used in practice due to its simplicity and effectiveness.
"""

from typing import Tuple, List, Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseBanditAlgorithm(ABC):
    """
    Abstract base class for all bandit algorithms.
    
    Defines the common interface that all bandit algorithms must implement
    for maintaining consistency across different exploration strategies.
    """
    
    def __init__(self, n_arms: int, seed: Optional[int] = None):
        """
        Initialize the base bandit algorithm.
        
        Args:
            n_arms: Number of arms in the bandit problem
            seed: Random seed for reproducibility
        """
        raise NotImplementedError("Subclasses must implement __init__")
    
    @abstractmethod
    def select_arm(self) -> int:
        """
        Select an arm to pull based on the current exploration strategy.
        
        Returns:
            Index of the selected arm (0 to n_arms-1)
        """
        raise NotImplementedError("Subclasses must implement select_arm")
    
    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """
        Update the algorithm's beliefs based on the observed reward.
        
        Args:
            arm: Index of the arm that was pulled
            reward: Observed reward from pulling that arm
        """
        raise NotImplementedError("Subclasses must implement update")
    
    @abstractmethod
    def get_best_arm(self) -> int:
        """
        Return the arm with the highest estimated value.
        
        Returns:
            Index of the best arm according to current estimates
        """
        raise NotImplementedError("Subclasses must implement get_best_arm")


class EpsilonGreedy(BaseBanditAlgorithm):
    """
    Epsilon-Greedy Exploration Strategy
    
    Theory:
        The epsilon-greedy algorithm is the simplest approach to the exploration-exploitation
        dilemma. At each time step, with probability epsilon (ε), it selects a random action
        to explore; with probability (1-ε), it selects the action with the highest estimated
        value to exploit. This simple strategy provides a balance between exploration and
        exploitation. The algorithm maintains empirical estimates of each arm's value using
        the sample average of observed rewards. Despite its simplicity, epsilon-greedy can
        achieve O(log T) expected regret bounds with appropriately chosen epsilon values.
    
    Math:
        Arm selection rule:
            a_t = argmax_a Q_t(a)        with probability 1-ε
            a_t = random from {1,...,K}  with probability ε
        
        Value estimation (incremental sample average):
            N(a) ← N(a) + 1
            Q(a) ← Q(a) + (1/N(a)) * (r - Q(a))
        
        Expected regret bound:
            E[R(T)] = O(ε*T*K + log(T)/ε)  where K is number of arms
        
        Optimal epsilon scheduling:
            ε_t = c / (t^p)  for p ∈ (0,1), typically p ≈ 1/2 or 1/3
    
    Attributes:
        n_arms: Number of arms in the bandit
        epsilon: Exploration probability (0 to 1)
        q_estimates: Array of estimated values for each arm
        arm_counts: Number of times each arm has been selected
        rng: NumPy random number generator
    
    References:
        - Sutton & Barto Chapter 2: https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
        - Auer et al. "Finite-time Analysis of the Multiarmed Bandit Problem": https://arxiv.org/abs/1402.6028
        - "Introduction to Online Learning": https://www.mit.edu/~rakhlin/6.883/lectures/lecture1.pdf
    
    Examples:
        >>> # Create an epsilon-greedy bandit with 5 arms
        >>> bandit = EpsilonGreedy(n_arms=5, epsilon=0.1)
        >>>
        >>> # Simulate 1000 interactions
        >>> rewards = []
        >>> for t in range(1000):
        ...     arm = bandit.select_arm()
        ...     # Assume reward is 1 for arm 2, 0 otherwise
        ...     reward = 1.0 if arm == 2 else 0.0
        ...     bandit.update(arm, reward)
        ...     rewards.append(reward)
        >>>
        >>> # Analyze results
        >>> print(f"Best arm found: {bandit.get_best_arm()}")
        >>> print(f"Estimated values: {bandit.get_value_estimates()}")
        >>> print(f"Total reward: {sum(rewards)}")
    """
    
    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
        initial_estimate: float = 0.0
    ) -> None:
        """
        Initialize the Epsilon-Greedy algorithm.
        
        Args:
            n_arms: Number of arms in the bandit problem
            epsilon: Exploration probability (must be in [0, 1])
            seed: Random seed for reproducibility
            initial_estimate: Initial value estimate for all arms
        
        Raises:
            ValueError: If epsilon is not in [0, 1] or n_arms <= 0
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Validate inputs (epsilon in [0,1], n_arms > 0)\n"
            "2. Initialize q_estimates with initial_estimate values\n"
            "3. Initialize arm_counts to zeros\n"
            "4. Set up random number generator with seed\n"
            "5. Store n_arms and epsilon as instance variables"
        )
    
    def select_arm(self) -> int:
        """
        Select an arm using epsilon-greedy strategy.
        
        Returns:
            Index of selected arm (0 to n_arms-1)
        
        Implementation Approach:
            1. Generate random number from [0, 1)
            2. If random < epsilon: return random arm index
            3. Else: return argmax of q_estimates
            4. Handle ties in argmax by selecting smallest index or random
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Generate u = rng.uniform(0, 1)\n"
            "2. If u < self.epsilon:\n"
            "       return rng.integers(0, self.n_arms)\n"
            "3. Else:\n"
            "       return np.argmax(self.q_estimates)\n"
            "4. Consider handling ties randomly for better exploration"
        )
    
    def update(self, arm: int, reward: float) -> None:
        """
        Update value estimates based on observed reward.
        
        Uses the incremental sample average formula to update estimates
        without storing individual samples.
        
        Args:
            arm: Index of the arm that was pulled (0 to n_arms-1)
            reward: Observed reward from the arm
        
        Raises:
            ValueError: If arm index is invalid
        
        Implementation Approach:
            1. Validate arm index
            2. Increment arm_counts[arm]
            3. Update q_estimates[arm] using incremental formula:
               Q(a) = Q(a) + (1/N(a)) * (r - Q(a))
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Validate: 0 <= arm < self.n_arms\n"
            "2. self.arm_counts[arm] += 1\n"
            "3. n = self.arm_counts[arm]\n"
            "4. self.q_estimates[arm] += (1/n) * (reward - self.q_estimates[arm])\n"
            "   or equivalently:\n"
            "   self.q_estimates[arm] = (old_sum + reward) / n"
        )
    
    def get_best_arm(self) -> int:
        """
        Return the arm with highest estimated value.
        
        Returns:
            Index of the best arm according to current estimates
        
        Note:
            In case of ties, returns the smallest index by default.
            Could be modified to return random arm among tied best arms.
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Return np.argmax(self.q_estimates)\n"
            "2. Or np.argmax(self.q_estimates[::-1]) for largest index tie-breaking\n"
            "3. Or randomized selection among tied arms"
        )
    
    def get_value_estimates(self) -> np.ndarray:
        """
        Get current value estimates for all arms.
        
        Returns:
            Array of estimated values, shape (n_arms,)
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Return a copy of self.q_estimates\n"
            "2. Use .copy() to prevent external modification"
        )
    
    def get_arm_counts(self) -> np.ndarray:
        """
        Get the number of times each arm has been pulled.
        
        Returns:
            Array of pull counts, shape (n_arms,)
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Return a copy of self.arm_counts\n"
            "2. Useful for analyzing exploration behavior"
        )
    
    def calculate_regret(self, optimal_arm_value: Optional[float] = None) -> float:
        """
        Calculate cumulative regret of the algorithm.
        
        Regret measures how much worse the algorithm performed compared to
        always pulling the best arm.
        
        Args:
            optimal_arm_value: True expected value of the best arm.
                               If None, uses empirical maximum.
        
        Returns:
            Total regret accumulated across all pulls
        
        Theory:
            Regret_T = sum_{t=1}^T (V* - V_{a_t})
            where V* is the optimal arm value and V_{a_t} is the value
            of the selected arm at time t.
        
        Implementation:
            Since we don't track individual arm pulls, regret can be
            approximated or exactly calculated depending on design choices.
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Total pulls: total_pulls = np.sum(self.arm_counts)\n"
            "2. If optimal_arm_value is None:\n"
            "       optimal = np.max(self.q_estimates)\n"
            "   Else:\n"
            "       optimal = optimal_arm_value\n"
            "3. Expected value from current distribution:\n"
            "       expected_reward = np.sum(self.q_estimates * self.arm_counts) / total_pulls\n"
            "4. regret = total_pulls * optimal - np.sum(self.q_estimates * self.arm_counts)"
        )


class LinearEpsilonGreedy(EpsilonGreedy):
    """
    Epsilon-Greedy with Linear Epsilon Decay
    
    Theory:
        This variant extends the basic epsilon-greedy algorithm with a linear
        epsilon decay schedule. Instead of using a fixed epsilon, the exploration
        probability decreases linearly over time from epsilon_start to epsilon_end.
        This allows for more exploration early on when uncertainties are high,
        then transitions to exploitation as the algorithm gathers more data.
        The linear schedule is simpler than other decay schemes but still effective.
    
    Math:
        Time-dependent epsilon:
            ε(t) = max(ε_end, ε_start - (ε_start - ε_end) * (t / T_decay))
        
        Where:
            t = current time step (number of pulls)
            T_decay = number of steps over which to decay
            ε_start = initial exploration probability
            ε_end = final minimum exploration probability
    
    Attributes:
        epsilon_start: Initial exploration probability
        epsilon_end: Final minimum exploration probability
        decay_steps: Number of steps to decay over
        t: Current time step
    
    References:
        - Decay schedules in bandits: https://arxiv.org/abs/1111.1797
        - Optimism in the Face of Uncertainty: https://arxiv.org/abs/0902.3900
    """
    
    def __init__(
        self,
        n_arms: int,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        decay_steps: int = 10000,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize Linear Epsilon-Greedy algorithm.
        
        Args:
            n_arms: Number of arms
            epsilon_start: Initial exploration probability
            epsilon_end: Final exploration probability
            decay_steps: Number of steps to decay over
            seed: Random seed
        
        Raises:
            ValueError: If epsilon values invalid or decay_steps <= 0
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Call parent __init__ with n_arms\n"
            "2. Store epsilon_start, epsilon_end, decay_steps\n"
            "3. Initialize time step counter t = 0\n"
            "4. Validate: epsilon_start > epsilon_end, decay_steps > 0"
        )
    
    def select_arm(self) -> int:
        """
        Select arm with time-decaying epsilon.
        
        Returns:
            Index of selected arm
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Calculate current epsilon:\n"
            "       epsilon_t = max(self.epsilon_end,\n"
            "                       self.epsilon_start -\n"
            "                       (self.epsilon_start - self.epsilon_end) *\n"
            "                       (self.t / self.decay_steps))\n"
            "2. Store epsilon_t in self.epsilon for base class\n"
            "3. Call parent select_arm method"
        )
    
    def update(self, arm: int, reward: float) -> None:
        """
        Update and increment time counter.
        
        Args:
            arm: Selected arm index
            reward: Observed reward
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Call parent update(arm, reward)\n"
            "2. self.t += 1"
        )
    
    def get_current_epsilon(self) -> float:
        """
        Get the current epsilon value based on time decay.
        
        Returns:
            Current exploration probability
        """
        raise NotImplementedError(
            "Implementation hint:\n"
            "1. Calculate epsilon using decay formula\n"
            "2. Return max(self.epsilon_end, calculated_epsilon)"
        )
