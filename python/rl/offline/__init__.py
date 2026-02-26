"""
Offline Reinforcement Learning - Learning from Fixed Datasets.

This module implements offline (batch) RL algorithms that learn policies
from fixed datasets without additional environment interaction. These
methods address the distribution shift and extrapolation error problems
inherent to offline settings.

Modules:
    - bcq: Batch-Constrained Q-learning
    - cql: Conservative Q-Learning
    - iql: Implicit Q-Learning
"""

from .bcq import BCQ, BCQVAE
from .cql import CQL
from .iql import IQL

__all__ = [
    'BCQ',
    'BCQVAE',
    'CQL',
    'IQL',
]
