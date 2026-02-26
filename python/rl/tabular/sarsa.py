"""
SARSA: On-Policy TD Control

SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm.
It learns Q-values by bootstrapping from the next action actually taken.

Theory:
    SARSA updates Q-values using the TD(0) update:
    Q(s,a) <- Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]

    Where a' is the action taken from s' under the current policy.
    This makes SARSA on-policy: it learns about the policy it's following.

Expected SARSA:
    Uses expected value over next actions instead of sampled action:
    Q(s,a) <- Q(s,a) + α [r + γ Σ_a' π(a'|s') Q(s',a') - Q(s,a)]

    This reduces variance while remaining on-policy.

References:
    - Sutton & Barto (2018), Chapter 6
    - Rummery & Niranjan (1994) "On-Line Q-Learning Using Connectionist Systems"
"""

import numpy as np
from typing import Tuple, Dict, Optional


class SARSA:
    """
    SARSA: On-policy TD control.

    Algorithm:
        1. Initialize Q(s,a) arbitrarily
        2. For each episode:
            a. Initialize s, choose a from s using policy (e.g., ε-greedy)
            b. For each step:
                - Take action a, observe r, s'
                - Choose a' from s' using policy
                - Q(s,a) <- Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                - s <- s', a <- a'
            c. Until s is terminal
    """

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1):
        """
        Initialize SARSA.

        Args:
            n_states: Number of states
            n_actions: Number of actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate for ε-greedy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        """Select action using ε-greedy policy."""
        raise NotImplementedError(
            "TODO: Implement ε-greedy action selection\\n"
            "Hint: With prob ε random, else argmax Q[state]"
        )

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool) -> float:
        """
        SARSA update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (from policy)
            done: Episode termination flag

        Returns:
            TD error
        """
        raise NotImplementedError(
            "TODO: Implement SARSA update\\n"
            "Hint: td_error = r + γ*Q(s',a') - Q(s,a) (0 if done)\\n"
            "      Q(s,a) += α * td_error"
        )


class ExpectedSARSA:
    """
    Expected SARSA: Uses expected value over next actions.

    Instead of sampling a', use:
    Q(s,a) <- Q(s,a) + α [r + γ Σ_a' π(a'|s') Q(s',a') - Q(s,a)]

    This reduces variance compared to SARSA.
    """

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        """Select action using ε-greedy policy."""
        raise NotImplementedError("Same as SARSA")

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """
        Expected SARSA update.

        Returns:
            TD error
        """
        raise NotImplementedError(
            "TODO: Implement Expected SARSA\\n"
            "Hint: Compute E[Q(s',a')] under ε-greedy policy"
        )
