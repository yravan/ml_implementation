"""
Meta-Learning - Learning to Learn.

This module implements meta-learning algorithms that learn to quickly
adapt to new tasks from limited data.

Modules:
    - maml: Model-Agnostic Meta-Learning and variants
"""

from .maml import (
    MAML,
    FOMAML,
    Reptile,
    ProtoNet
)

__all__ = [
    'MAML',
    'FOMAML',
    'Reptile',
    'ProtoNet',
]
