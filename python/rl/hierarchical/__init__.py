"""
Hierarchical Reinforcement Learning.

This module implements hierarchical RL methods that decompose complex tasks
into temporally abstracted subtasks, enabling better exploration, credit
assignment, and skill transfer.

Modules:
    - hrl: Options framework, Option-Critic, HIRO
"""

from .hrl import (
    Option,
    OptionCritic,
    HIRO
)

__all__ = [
    'Option',
    'OptionCritic',
    'HIRO',
]
