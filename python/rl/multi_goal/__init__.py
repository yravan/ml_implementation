"""
Multi-Goal and Goal-Conditioned Reinforcement Learning.

This module implements goal-conditioned RL methods that learn policies
capable of achieving multiple goals with a single network.

Modules:
    - goal_conditioned: UVFA, HER, goal-conditioned DDPG
"""

from .goal_conditioned import (
    GoalConditionedPolicy,
    HindsightExperienceReplay,
    GoalConditionedDDPG,
    RelabelingStrategies,
    compute_sparse_reward,
    compute_dense_reward
)

__all__ = [
    'GoalConditionedPolicy',
    'HindsightExperienceReplay',
    'GoalConditionedDDPG',
    'RelabelingStrategies',
    'compute_sparse_reward',
    'compute_dense_reward',
]
