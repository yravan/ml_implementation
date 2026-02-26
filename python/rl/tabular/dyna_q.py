"""
Dyna-Q: Model-Based Reinforcement Learning

Dyna-Q combines model-free learning (Q-Learning) with model-based planning.
It learns a model of the environment and uses simulated experience to
accelerate learning.

Theory:
    Dyna-Q maintains:
    1. Q-table for value estimates
    2. Model: M(s,a) -> (r, s') predictions

    Each real experience:
    1. Update Q directly (Q-Learning)
    2. Update model M(s,a) <- (r, s')
    3. Perform n planning steps:
       - Sample (s,a) from previously seen pairs
       - Get (r, s') from model
       - Update Q using simulated experience

Architecture:
    Real Experience -> Q-Learning Update
                   -> Model Update
    Model -> Simulated Experience -> Q-Learning Updates (planning)

References:
    - Sutton (1990) "Integrated Architectures for Learning, Planning, and Reacting"
    - Sutton & Barto (2018), Chapter 8
"""

import numpy as np
from typing import Dict, Tuple, List


class DynaQ:
    """
    Dyna-Q: Tabular model-based RL.

    Combines:
    - Q-Learning for model-free control
    - Learned model for planning
    """

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1, n_planning_steps: int = 5):
        """
        Initialize Dyna-Q.

        Args:
            n_states: Number of states
            n_actions: Number of actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            n_planning_steps: Number of planning updates per real step
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps

        self.Q = np.zeros((n_states, n_actions))
        self.model: Dict[Tuple[int, int], Tuple[float, int]] = {}
        self.visited_pairs: List[Tuple[int, int]] = []

    def select_action(self, state: int) -> int:
        """Select action using ε-greedy."""
        raise NotImplementedError("ε-greedy action selection")

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        """
        Full Dyna-Q update.

        1. Q-Learning update from real experience
        2. Model update
        3. Planning steps
        """
        raise NotImplementedError(
            "TODO: Implement Dyna-Q\\n"
            "Hint: Q-Learning update, then model[(s,a)] = (r, s'),\\n"
            "      then n_planning_steps of simulated updates"
        )

    def planning_step(self) -> None:
        """Single planning step using model."""
        raise NotImplementedError(
            "TODO: Sample (s,a) from visited, get (r,s') from model, Q-update"
        )
