"""
RL Policy Gradient Module
=========================

Comprehensive implementations of policy gradient reinforcement learning algorithms.

This module provides educational implementations of foundational and state-of-the-art
policy gradient algorithms, from basic REINFORCE to advanced methods like PPO and TRPO.

Algorithms Included:
    - REINFORCE (baseline policy gradient, Williams 1992)
    - VPG (Vanilla Policy Gradient with baseline)
    - Natural Policy Gradient (Fisher Information Matrix)
    - A2C (Advantage Actor-Critic, synchronous updates)
    - TRPO (Trust Region Policy Optimization)
    - PPO (Proximal Policy Optimization) - discrete actions
    - PPO-Continuous (PPO for continuous action spaces)

Key Concepts:
    - Policy Gradient Theorem: Foundation for all algorithms
    - Advantage Estimation: Reducing variance in gradient estimates
    - Trust Regions: Constraining policy updates for stability
    - Clipped Objectives: Simple alternative to natural gradients

Usage Examples:
    >>> from rl.policy_gradient import REINFORCE, PPO, TRPO
    >>>
    >>> # Basic REINFORCE agent
    >>> agent = REINFORCE(
    ...     state_dim=4,
    ...     action_dim=2,
    ...     learning_rate=1e-2
    ... )
    >>>
    >>> # PPO with continuous actions
    >>> from rl.policy_gradient import PPOContinuous
    >>> agent = PPOContinuous(
    ...     state_dim=6,
    ...     action_dim=3,
    ...     learning_rate=3e-4
    ... )

References:
    - Policy Gradient Theorem (Sutton et al., 1999)
    - REINFORCE (Williams, 1992)
    - Natural Policy Gradient (Kakade, 2002)
    - Actor-Critic (Konda & Tsitsiklis, 2000)
    - TRPO (Schulman et al., 2015)
    - PPO (Schulman et al., 2017)

Paper URLs:
    - https://arxiv.org/abs/1708.02747 (PPO)
    - https://arxiv.org/abs/1502.05477 (TRPO)
    - https://arxiv.org/abs/1211.1541 (Natural PG)
"""

from .reinforce import REINFORCE
from .vpg import VanillaPolicyGradient
from .npg import NaturalPolicyGradient
from .a2c import A2C, AdvantageActorCritic
from .trpo import TRPO, TrustRegionPolicyOptimization
from .ppo import PPO, ProximalPolicyOptimization
from .ppo_continuous import PPOContinuous, PPOContinuousAction

__version__ = "1.0.0"
__author__ = "ML Implementation Team"

__all__ = [
    "REINFORCE",
    "VanillaPolicyGradient",
    "NaturalPolicyGradient",
    "A2C",
    "AdvantageActorCritic",
    "TRPO",
    "TrustRegionPolicyOptimization",
    "PPO",
    "ProximalPolicyOptimization",
    "PPOContinuous",
    "PPOContinuousAction",
]
