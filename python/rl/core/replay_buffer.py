"""
Experience Replay Buffers for Reinforcement Learning

This module implements different replay buffer strategies for storing and sampling
experience tuples (state, action, reward, next_state, done).

THEORY:
    Experience replay addresses several key issues in RL:

    1. CORRELATION: Sequential experience is highly correlated
       - Adjacent transitions are nearly identical
       - Training on correlated data biases learning

    2. STABILITY: Neural networks are sensitive to data distribution
       - Naive mini-batch learning causes catastrophic forgetting
       - Replay from past experience stabilizes training

    3. SAMPLE EFFICIENCY: Learn from transitions multiple times
       - Uniformly random replay: older transitions as valuable as recent
       - Prioritized replay: focus on high TD-error (more informative) transitions

BUFFER TYPES:
    1. Uniform Replay: Sample transitions uniformly at random (standard)
    2. Prioritized Experience Replay (PER): Weight by TD-error priority
    3. N-step Returns: Bootstrap further in future (n steps vs 1 step)

REFERENCES:
    - Sutton & Barto (2018), Section 11.8
    - DQN: Mnih et al. (2015)
    - PER: Schaul et al. (2015)
    - N-step: Peng & Williams (1996)
"""

import numpy as np
from typing import Tuple, Optional, NamedTuple, List
from collections import deque
import random


class Transition(NamedTuple):
    """
    Single transition tuple (s, a, r, s', done).

    COMPONENTS:
    - state: Current state s_t
    - action: Action taken a_t
    - reward: Immediate reward r_t
    - next_state: Resulting state s_{t+1}
    - done: Episode termination flag (boolean)
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class BaseReplayBuffer:
    """Abstract base class for replay buffers."""

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        raise NotImplementedError()

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sample a batch of transitions.

        Returns:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size, 1]
            next_states: [batch_size, state_dim]
            dones: [batch_size, 1]
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """Return buffer size."""
        raise NotImplementedError()


class UniformReplayBuffer(BaseReplayBuffer):
    """
    Standard Uniform Experience Replay Buffer.

    Stores transitions in a circular buffer and samples uniformly at random.
    Used in DQN and most standard deep RL algorithms.

    SAMPLING:
        Draw batch_size transitions uniformly at random (with replacement)
        Each transition equally likely regardless of recency or importance

    ADVANTAGES:
    - Simple and efficient
    - No bias from priority estimation
    - Stable, well-understood

    DISADVANTAGES:
    - Ignores transition importance
    - Doesn't learn efficiently from rare/important experiences
    """

    def __init__(self,
                 max_size: int = 100000,
                 state_dim: Optional[int] = None,
                 action_dim: Optional[int] = None):
        """
        Initialize uniform replay buffer.

        Args:
            max_size: Maximum capacity of buffer (default: 100k)
            state_dim: Dimension of state (optional, for pre-allocation)
            action_dim: Dimension of action (optional, for pre-allocation)

        Example:
            >>> buffer = UniformReplayBuffer(max_size=50000)
            >>> # Add transitions
            >>> buffer.add(Transition(state, action, reward, next_state, done))
            >>> # Sample batch - returns numpy arrays
            >>> states, actions, rewards, next_states, dones = buffer.sample(batch_size=32)
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def add(self, transition: Transition) -> None:
        """
        Add transition to buffer (FIFO, overwrites oldest when full).

        Args:
            transition: Transition(state, action, reward, next_state, done)
        """
        raise NotImplementedError(
            "Hint: Simply call self.buffer.append(transition). "
            "deque automatically handles max_size overflow."
        )

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sample batch_size transitions uniformly at random.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size, 1]
            next_states: [batch_size, state_dim]
            dones: [batch_size, 1] (boolean mask for episode end)

        SAMPLING STRATEGY:
            1. Draw batch_size random indices uniformly from [0, len(buffer))
            2. Extract transitions at those indices
            3. Stack arrays into batches as numpy arrays
            4. Return numpy arrays
        """
        raise NotImplementedError(
            "Hint: Use random.sample(self.buffer, batch_size) to get random transitions. "
            "Unpack each transition and stack into batch numpy arrays. "
            "Return as tuple of numpy arrays."
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        raise NotImplementedError(
            "Hint: Return len(self.buffer)"
        )


class PrioritizedExperienceReplay(BaseReplayBuffer):
    """
    Prioritized Experience Replay (PER).

    Assigns priorities to transitions based on TD-error magnitude.
    Samples high-priority (high TD-error) transitions more frequently.

    MOTIVATION:
        Some transitions are more "surprising" and informative:
        - Large TD-error: model prediction was very wrong
        - Should learn more from these high-error transitions
        - Standard uniform replay treats all equally

    PRIORITY:
        p_i = |r_i + γ max_a Q(s'_i, a) - Q(s_i, a_i)| + ε

        where ε is small constant to avoid zero-priority

    SAMPLING PROBABILITY:
        P(i) = p_i^α / ∑_j p_j^α

        where α ∈ [0, 1] controls how much prioritization
        α = 0: uniform sampling, α = 1: pure proportional prioritization

    IMPORTANCE SAMPLING WEIGHTS:
        w_i = (1 / (N * P(i)))^β

        where β is annealing schedule (starts low, increases to 1)
        Corrects bias from non-uniform sampling

    REFERENCE:
        Schaul et al. (2015): "Prioritized Experience Replay"
    """

    def __init__(self,
                 max_size: int = 100000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 epsilon: float = 1e-6):
        """
        Initialize prioritized replay buffer.

        Args:
            max_size: Maximum buffer capacity
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            epsilon: Small constant to avoid zero priority

        ANNEALING SCHEDULE:
            Typically start with low β and anneal to 1.0 during training
            (to focus on recent high-priority transitions early)

        Example:
            >>> buffer = PrioritizedExperienceReplay(max_size=50000, alpha=0.6, beta=0.4)
            >>> # Add transitions with initial max priority
            >>> buffer.add(transition)
            >>> # Sample with priority weights
            >>> states, actions, rewards, next_states, dones, weights, indices = buffer.sample(32)
            >>> # Update priorities after computing TD-error
            >>> td_errors = compute_td_error(...)
            >>> buffer.update_priorities(indices, td_errors)
        """
        self.max_size = max_size
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.epsilon = epsilon
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)

    def add(self, transition: Transition) -> None:
        """
        Add transition with maximum current priority.

        New experiences get highest priority to ensure they're sampled
        at least once before potential removal.

        Args:
            transition: Transition tuple
        """
        raise NotImplementedError(
            "Hint: Add transition to self.buffer. Get max priority from self.priorities "
            "(or use initial high value), add to self.priorities"
        )

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray
    ]:
        """
        Sample batch with prioritization and importance sampling weights.

        SAMPLING PROCESS:
            1. Compute probabilities: P(i) = (p_i^α) / (∑_j p_j^α)
            2. Sample indices according to P(i)
            3. Compute importance weights: w_i = (1 / (N * P(i)))^β
            4. Normalize weights: w_i ← w_i / max(w)  (for stability)

        Args:
            batch_size: Number of transitions to sample

        Returns:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size, 1]
            next_states: [batch_size, state_dim]
            dones: [batch_size, 1]
            weights: [batch_size] importance sampling weights
            indices: [batch_size] original buffer indices (for priority update)
        """
        raise NotImplementedError(
            "Hint: Compute priorities^alpha, normalize to get probabilities. "
            "Sample batch_size indices using np.random.choice(). "
            "Compute weights = (1 / (N * P))^beta. Normalize by max weight. "
            "Return transitions + weights + indices for later priority update."
        )

    def update_priorities(self,
                         indices: np.ndarray,
                         td_errors: np.ndarray) -> None:
        """
        Update transition priorities based on TD-error magnitude.

        Called after computing loss to reflect transition importance.
        High TD-error = high priority = more likely to be sampled

        Args:
            indices: Buffer indices of sampled transitions
            td_errors: |TD-error| for each transition (magnitude, positive)

        PRIORITY UPDATE:
            p_i = (|td_error_i| + ε)^α
        """
        raise NotImplementedError(
            "Hint: For each (index, td_error) pair, compute "
            "priority = (|td_error| + epsilon)^alpha and update "
            "self.priorities at that index"
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        raise NotImplementedError("Hint: Return len(self.buffer)")


class NStepReplayBuffer(BaseReplayBuffer):
    """
    N-step Experience Replay Buffer.

    Computes n-step returns instead of 1-step TD targets.
    Trades bias for variance: longer lookhead = lower bias but higher variance.

    MOTIVATION:
        1-step TD: target = r_t + γ V(s_{t+1})
        Low bias (single prediction), but high variance

        n-step TD: target = (∑_{i=0}^{n-1} γ^i r_{t+i}) + γ^n V(s_{t+n})
        n-step bootstrap: intermediate reward accumulation + final value estimate
        Lower variance (more real experience), but higher bias

    N-STEP RETURN:
        G_t^(n) = ∑_{i=0}^{n-1} γ^i r_{t+i} + γ^n V(s_{t+n})

    TRADE-OFF:
        n=1: high variance, low bias (pure TD)
        n=∞: low variance, high bias (pure Monte Carlo)
        n=λ: blend both via eligibility traces (TD(λ))

    Used in:
    - A3C (n-step Actor-Critic)
    - PPO (can use n-step returns)
    - Distributed agents (e.g., Ape-X)

    REFERENCE:
        Peng & Williams (1996): "Incremental Multi-Step Q-Learning"
    """

    def __init__(self,
                 max_size: int = 100000,
                 n_steps: int = 3,
                 gamma: float = 0.99):
        """
        Initialize n-step replay buffer.

        Args:
            max_size: Maximum buffer capacity
            n_steps: Number of steps to lookahead (default: 3)
            gamma: Discount factor

        Example:
            >>> buffer = NStepReplayBuffer(max_size=50000, n_steps=3)
            >>> # Add transitions sequentially
            >>> for transition in trajectory:
            ...     buffer.add(transition)
            >>> # Sample n-step transitions
            >>> states, actions, n_step_returns, next_states, dones = buffer.sample(32)
        """
        self.max_size = max_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=max_size)
        self.transition_cache = deque(maxlen=n_steps)

    def add(self, transition: Transition) -> None:
        """
        Add transition and accumulate n-step return.

        Must be called sequentially for n-step accumulation to work.

        Args:
            transition: Transition tuple
        """
        raise NotImplementedError(
            "Hint: Add transition to cache. When cache is full, compute n-step return "
            "and add complete transition to buffer. Handle episode boundaries (done=True)."
        )

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sample n-step transitions.

        Args:
            batch_size: Number of transitions

        Returns:
            states: Starting states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            n_step_returns: ∑_{i=0}^{n-1} γ^i r_{t+i} [batch_size, 1]
            next_states: States after n steps [batch_size, state_dim]
            dones: Episode termination flags [batch_size, 1]

        NOTE: n_step_returns already includes accumulated discounted
              rewards over n steps, no need to add discounting again
        """
        raise NotImplementedError(
            "Hint: Sample transitions from buffer, ensure they have "
            "n-step accumulated returns pre-computed"
        )

    def compute_n_step_return(self) -> Tuple[float, bool]:
        """
        Compute n-step return from cached transitions.

        MATH:
            G_t^(n) = ∑_{i=0}^{n-1} γ^i r_{t+i} + γ^n V(s_{t+n})

            But in replay buffer, we don't have V yet, so:
            G_t^(n) = ∑_{i=0}^{n-1} γ^i r_{t+i}

            The final bootstrap Q(s_{t+n}, a) is added during training

        Returns:
            n_step_return: Accumulated discounted reward over n steps
            done: Whether any transition in the n-step sequence ended episode

        EPISODE BOUNDARY HANDLING:
            If transition i has done=True, future rewards are zero:
            G_t^(n) = ∑_{i=0}^{k} γ^i r_{t+i}  where k < n-1 is first done
        """
        raise NotImplementedError(
            "Hint: Accumulate discounted rewards from cached transitions: "
            "n_step_return = sum(gamma^i * transition.reward for i, transition in cache). "
            "Stop accumulation if done=True. Return (n_step_return, any_done)"
        )

    def __len__(self) -> int:
        """Return buffer size."""
        raise NotImplementedError("Hint: Return len(self.buffer)")


class HindsightExperienceReplay(BaseReplayBuffer):
    """
    Hindsight Experience Replay (HER) for goal-conditioned RL.

    Addresses sparse reward problem by relabeling failed trajectories
    with achieved goals as targets.

    MOTIVATION:
        Sparse reward: agent rarely sees positive reward (e.g., goal-based tasks)
        Solution: Treat achieved state in failed trajectory as goal
                 Rewrite: "Failed to reach goal G, but reached state S"
                 as: "Successfully reached goal S"

    STRATEGY:
        1. Execute trajectory: s_0 →a_0 s_1 →a_1 ... →a_{T-1} s_T
        2. Record actual goal G and achieved final state S_T
        3. For each transition (s_t, a_t, s_{t+1}):
           a. Keep original: (s_t, a_t, reward, s_{t+1}, goal=G)
           b. Add hindsight: (s_t, a_t, reward, s_{t+1}, goal=S_T)
        4. Hindsight goal usually gives positive reward (s_{t+1} matches S_T)

    USED IN:
    - Multi-goal RL
    - Robotic manipulation (pick-and-place, etc.)
    - Any sparse reward environment

    REFERENCE:
        Andrychowicz et al. (2017): "Hindsight Experience Replay"
    """

    def __init__(self,
                 max_size: int = 100000,
                 goal_dim: int = 3,
                 hindsight_ratio: float = 0.8):
        """
        Initialize HER buffer.

        Args:
            max_size: Maximum buffer capacity
            goal_dim: Dimension of goal representation
            hindsight_ratio: Fraction of hindsight transitions to add
                           (typically 0.8: 80% hindsight, 20% original)

        Example:
            >>> buffer = HindsightExperienceReplay(max_size=50000, goal_dim=3)
            >>> # Add trajectory with original goal
            >>> trajectory = [transitions...]
            >>> buffer.add_trajectory(trajectory, goal, achieved_goal)
        """
        self.max_size = max_size
        self.goal_dim = goal_dim
        self.hindsight_ratio = hindsight_ratio
        self.buffer = deque(maxlen=max_size)

    def add_trajectory(self,
                       trajectory: List[Transition],
                       goal: np.ndarray,
                       achieved_goal: np.ndarray) -> None:
        """
        Add trajectory with original and hindsight relabeling.

        Args:
            trajectory: List of Transition objects
            goal: Original goal (what agent was trying to reach)
            achieved_goal: Final state achieved in trajectory
        """
        raise NotImplementedError(
            "Hint: For each transition in trajectory, add both original "
            "(goal=goal) and hindsight (goal=achieved_goal) versions. "
            "Use hindsight_ratio to probabilistically include hindsight transitions."
        )

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sample transitions with goals.

        Returns:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size, 1]
            next_states: [batch_size, state_dim]
            dones: [batch_size, 1]
            goals: [batch_size, goal_dim]
        """
        raise NotImplementedError(
            "Hint: Sample from buffer, return transitions with goal information"
        )

    def __len__(self) -> int:
        """Return buffer size."""
        raise NotImplementedError("Hint: Return len(self.buffer)")


# Aliases for common naming conventions
ReplayBuffer = UniformReplayBuffer
PrioritizedReplayBuffer = PrioritizedExperienceReplay

