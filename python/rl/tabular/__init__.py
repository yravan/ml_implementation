"""
RL Tabular Methods Module

This module implements classic tabular reinforcement learning algorithms that work
with discrete state and action spaces. These methods form the foundation of modern RL
and are essential for understanding more complex deep RL approaches.

Key Algorithms:
- Policy Evaluation: Compute state value functions under a given policy
- Policy Iteration: Alternate between policy evaluation and improvement
- Value Iteration: Dynamic programming without explicit policy evaluation
- Monte Carlo: Learn from complete episode rollouts
- Temporal Difference (TD): Bootstrap value estimates using bootstrapping
- SARSA: On-policy TD control
- Q-Learning: Off-policy TD control
- Dyna-Q: Model-based planning with tabular learning
- MCTS: Monte Carlo Tree Search for planning

References:
- Sutton & Barto (2018): https://mitpress.mit.edu/9780262039246/reinforcement-learning/
- David Silver's RL Course: https://www.davidsilver.uk/teaching/
- Kocsis & Szepesvari (2006): https://arxiv.org/pdf/cs/0606153.pdf
"""

from .policy_evaluation import PolicyEvaluator
from .policy_iteration import PolicyIterator
from .value_iteration import ValueIterator
from .monte_carlo import MonteCarloPrediction, MonteCarloControl
from .td_learning import TDLearner, TDLambda, NStepTD
from .sarsa import SARSA, ExpectedSARSA
from .q_learning import QLearning, DoubleQLearning
from .dyna_q import DynaQ
from .mcts import MonteCarloTreeSearch

__all__ = [
    "PolicyEvaluator",
    "PolicyIterator",
    "ValueIterator",
    "MonteCarloPrediction",
    "MonteCarloControl",
    "TDLearner",
    "TDLambda",
    "NStepTD",
    "SARSA",
    "ExpectedSARSA",
    "QLearning",
    "DoubleQLearning",
    "DynaQ",
    "MonteCarloTreeSearch",
]

__version__ = "1.0.0"
