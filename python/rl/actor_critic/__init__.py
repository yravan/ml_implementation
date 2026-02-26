"""
Actor-Critic Reinforcement Learning Algorithms Module

This module provides implementations of state-of-the-art continuous control algorithms
based on the actor-critic framework. These algorithms learn both a policy (actor) and
a value function (critic) to solve continuous action space problems.

Implemented Algorithms:
- DDPG (Deep Deterministic Policy Gradient): Off-policy algorithm using deterministic policies
- TD3 (Twin Delayed DDPG): Improved DDPG with twin critics and delayed policy updates
- SAC (Soft Actor-Critic): Maximum entropy RL with automatic entropy regularization

The actor-critic approach combines the benefits of policy gradient methods (actor) with
value function approximation (critic), enabling efficient learning in high-dimensional
continuous control tasks.

Module Structure:
    ddpg: Deep Deterministic Policy Gradient implementation
    td3: Twin Delayed DDPG implementation with improved stability
    sac: Soft Actor-Critic implementation with entropy regularization

References:
    - Lillicrap et al. (2015): Continuous control with deep reinforcement learning
    - Fujimoto et al. (2018): Addressing Function Approximation Error in Actor-Critic Methods
    - Haarnoja et al. (2018): Soft Actor-Critic: Off-Policy Deep Reinforcement Learning
      with a Stochastic Actor
"""

from typing import Type
from .ddpg import DDPGAgent, DDPGConfig, ReplayBuffer
from .td3 import TD3Agent, TD3Config
from .sac import SACAgent, SACConfig

__all__ = [
    # DDPG
    "DDPGAgent",
    "DDPGConfig",
    "ReplayBuffer",

    # TD3
    "TD3Agent",
    "TD3Config",

    # SAC
    "SACAgent",
    "SACConfig",
]

__version__ = "1.0.0"
__author__ = "RL Research"
__doc_url__ = "https://arxiv.org/abs/1509.02971"  # DDPG paper
