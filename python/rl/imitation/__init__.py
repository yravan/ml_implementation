"""
Imitation Learning - Learning from Demonstrations.

This module implements various imitation learning algorithms for learning
policies from expert demonstrations, ranging from simple behavior cloning
to inverse reinforcement learning and adversarial methods.

Modules:
    - behavior_cloning: Supervised learning from demonstrations
    - dagger: Interactive imitation with dataset aggregation
    - inverse_rl: Recover reward functions from demonstrations
    - gail: Generative adversarial imitation learning
    - airl: Adversarial inverse reinforcement learning
"""

from .behavior_cloning import (
    BehaviorCloning,
    BCWithAugmentation,
    EnsembleBehaviorCloning,
    load_demonstrations,
    collect_demonstrations
)

from .dagger import (
    DAgger,
    SafeDAgger,
    HGDAgger
)

__all__ = [
    # Behavior Cloning
    'BehaviorCloning',
    'BCWithAugmentation',
    'EnsembleBehaviorCloning',
    'load_demonstrations',
    'collect_demonstrations',
    # DAgger variants
    'DAgger',
    'SafeDAgger',
    'HGDAgger',
]
