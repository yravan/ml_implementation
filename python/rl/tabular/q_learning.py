"""
Q-Learning: Off-Policy TD Control

Q-Learning is an off-policy TD control algorithm that learns the optimal
Q-function directly, regardless of the policy being followed.

Theory:
    Q-Learning update:
    Q(s,a) <- Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

    Key insight: We bootstrap from the greedy action (max Q) even if we
    don't take that action. This makes Q-Learning off-policy.

Double Q-Learning:
    Addresses maximization bias by using two Q-functions:
    Q1(s,a) <- Q1(s,a) + α [r + γ Q2(s', argmax_a' Q1(s',a')) - Q1(s,a)]

    Swap Q1 and Q2 with probability 0.5.

References:
    - Watkins (1989) "Learning from Delayed Rewards"
    - Van Hasselt (2010) "Double Q-Learning"
    - Sutton & Barto (2018), Chapter 6
"""

import numpy as np
from typing import Tuple, Optional


class QLearning:
    """
    Q-Learning: Off-policy TD control.

    Algorithm:
        1. Initialize Q(s,a) arbitrarily
        2. For each episode:
            a. Initialize s
            b. For each step:
                - Choose a from s using policy (e.g., ε-greedy)
                - Take action a, observe r, s'
                - Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                - s <- s'
            c. Until s is terminal
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
        raise NotImplementedError(
            "TODO: Implement ε-greedy\\n"
            "Hint: With prob ε random, else argmax Q[state]"
        )

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """
        Q-Learning update.

        Returns:
            TD error
        """
        raise NotImplementedError(
            "TODO: Implement Q-Learning update\\n"
            "Hint: td_target = r + γ * max(Q[s']) (0 if done)\\n"
            "      td_error = td_target - Q[s,a]\\n"
            "      Q[s,a] += α * td_error"
        )


class DoubleQLearning:
    """
    Double Q-Learning: Addresses maximization bias.

    Uses two Q-tables to decouple action selection from evaluation:
    - Q1 for selecting best action
    - Q2 for evaluating that action (and vice versa)
    """

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        """Select using sum of both Q-tables."""
        raise NotImplementedError(
            "TODO: Use Q1 + Q2 for action selection"
        )

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """Double Q-Learning update."""
        raise NotImplementedError(
            "TODO: With prob 0.5 update Q1 using Q2, else update Q2 using Q1"
        )
