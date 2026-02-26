"""
Value-Based Reinforcement Learning Module

Comprehensive implementations of deep Q-learning and its variants including:
- DQN: Deep Q-Network for learning value functions from high-dimensional inputs
- Double DQN: Reduces overestimation bias in Q-learning
- Dueling DQN: Learns value and advantage functions separately
- Prioritized DQN: Prioritizes important transitions in experience replay
- C51: Categorical DQN for distributional reinforcement learning
- Rainbow DQN: Combines all DQN improvements
- Fitted Q-Iteration: Batch-based offline Q-learning

All implementations follow DeepMind research standards with proper citations
and extensive documentation of the underlying theory.

References:
    - Mnih et al. (2015): Playing Atari with Deep Reinforcement Learning
      https://arxiv.org/abs/1312.5602
    - van Hasselt et al. (2015): Deep Reinforcement Learning with Double Q-learning
      https://arxiv.org/abs/1509.06461
    - Wang et al. (2016): Dueling Network Architectures for Deep Reinforcement Learning
      https://arxiv.org/abs/1511.06581
    - Schaul et al. (2016): Prioritized Experience Replay
      https://arxiv.org/abs/1511.05952
    - Bellemare et al. (2017): A Distributional Perspective on Reinforcement Learning
      https://arxiv.org/abs/1707.06887
    - Hessel et al. (2018): Rainbow: Combining Improvements in Deep Reinforcement Learning
      https://arxiv.org/abs/1710.02298
"""

from .dqn import DQN, ExperienceReplay, ReplayBuffer
from .double_dqn import DoubleDQN
from .dueling_dqn import DuelingDQN
from .prioritized_dqn import PrioritizedDQN, PrioritizedReplayBuffer
from .c51 import C51CategoricalDQN, DistributionBuffer
from .rainbow import RainbowDQN
from .fitted_q import FittedQIteration, BatchReplayBuffer

__all__ = [
    # DQN and base classes
    "DQN",
    "ExperienceReplay",
    "ReplayBuffer",
    # DQN variants
    "DoubleDQN",
    "DuelingDQN",
    "PrioritizedDQN",
    "PrioritizedReplayBuffer",
    "C51CategoricalDQN",
    "DistributionBuffer",
    "RainbowDQN",
    # Offline learning
    "FittedQIteration",
    "BatchReplayBuffer",
]

__version__ = "1.0.0"
