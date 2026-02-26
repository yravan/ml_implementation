"""
Neural Network Architectures for Reinforcement Learning

This module provides neural network building blocks commonly used in RL algorithms:
- Policy networks (for action selection)
- Value networks (for state/action value estimation)
- Feature extraction (CNN for images, preprocessing for features)
- Actor-Critic combined architectures

DESIGN PRINCIPLES:
    1. Modularity: Reusable components for different algorithms
    2. Flexibility: Support discrete/continuous actions, different input types
    3. Stability: Initialization, normalization for RL training
    4. Efficiency: Appropriate architecture choices for different problems

COMMON ARCHITECTURES:
    - MLP: Dense networks for low-dim state spaces
    - CNN: Convolutional for image inputs (Atari, robotic vision)
    - LSTM: Recurrent for partial observability
    - Actor-Critic: Shared trunk + separate policy/value heads

INITIALIZATION:
    Proper initialization is critical for RL training stability
    - Last layer: Small initialization (prevents overconfident policy/values)
    - Hidden layers: Standard initialization (Xavier/Kaiming)
    - Output layers: Task-specific (tanh for bounded actions, etc.)

REFERENCES:
    - DQN: Mnih et al. (2015)
    - A3C/A2C: Mnih et al. (2016)
    - PPO: Schulman et al. (2017)
    - Spinning Up architectures
    - Deep RL Hands-On: Lapan (2020)
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from abc import ABC, abstractmethod
from python.nn_core import Module, Parameter


class BaseNetwork(Module, ABC):
    """Abstract base class for RL neural networks."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        """Forward pass."""
        raise NotImplementedError()


class MLPNetwork(BaseNetwork):
    """
    Multi-Layer Perceptron (MLP) for low-dimensional state inputs.

    Simple, fully-connected architecture suitable for:
    - Continuous control (e.g., MuJoCo, robotic arms)
    - Low-dimensional discrete spaces
    - Feature-based (non-image) observations

    ARCHITECTURE:
        input → [Dense + ReLU] → [Dense + ReLU] → ... → output

    TYPICAL USAGE:
    - Policy network: state_dim → hidden → hidden → action_dim
    - Value network: state_dim → hidden → hidden → 1
    - Q-network: state_dim → hidden → hidden → num_actions
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = "relu",
                 output_activation: Optional[str] = None,
                 use_batch_norm: bool = False):
        """
        Initialize MLP network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_sizes: Tuple of hidden layer sizes (default: (64, 64))
            activation: Hidden layer activation ("relu", "tanh", "elu", default: "relu")
            output_activation: Output layer activation (None, "tanh", "sigmoid", etc.)
            use_batch_norm: Apply batch normalization after hidden layers

        Example:
            >>> # Policy network for continuous control
            >>> policy_net = MLPNetwork(
            ...     input_dim=17,      # state dimension
            ...     output_dim=6,      # action dimension
            ...     hidden_sizes=(64, 64),
            ...     output_activation="tanh"
            ... )
            >>> state = np.random.randn(32, 17)  # batch of 32 states
            >>> action = policy_net(state)   # [32, 6]

            >>> # Value network
            >>> value_net = MLPNetwork(
            ...     input_dim=17,
            ...     output_dim=1,
            ...     hidden_sizes=(64, 64)
            ... )
            >>> value = value_net(state)  # [32, 1]
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes

        # Map activation names to functions
        self.activation = activation  # Store activation name
        self.output_activation = output_activation  # Store output activation name

        raise NotImplementedError(
            "Hint: Build network as nn.Sequential with:\n"
            "1. Linear(input_dim, hidden_sizes[0])\n"
            "2. activation\n"
            "3. Repeat: Linear(hidden_sizes[i], hidden_sizes[i+1]) + activation\n"
            "4. Linear(hidden_sizes[-1], output_dim)\n"
            "5. output_activation (if provided)\n"
            "If use_batch_norm, add BatchNorm1d after each hidden linear layer"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input array [batch_size, input_dim]

        Returns:
            output: [batch_size, output_dim]
        """
        raise NotImplementedError(
            "Hint: Simply return self.net(x) - all logic in __init__"
        )


class CNNNetwork(BaseNetwork):
    """
    Convolutional Neural Network for image inputs.

    Used for high-dimensional visual observations (e.g., Atari games,
    robotic vision systems).

    ARCHITECTURE:
        Image → [Conv2d] → [Conv2d] → [Conv2d] → Flatten → [FC] → Output

    DESIGN FOR RL:
    - Use stride 2 for downsampling (better than pooling in RL)
    - Typical Atari: 3 conv layers (16, 32, 32 filters, kernel 3x3)
    - Output features fed to policy/value heads

    REFERENCES:
    - DQN architecture: Mnih et al. (2015)
    - Nature DQN paper improvements
    """

    def __init__(self,
                 input_channels: int = 4,
                 output_dim: int = 512,
                 conv_filters: Tuple[int, ...] = (16, 32, 32),
                 kernel_size: int = 3,
                 stride: int = 2):
        """
        Initialize CNN network.

        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB, 4 for stacked frames)
            output_dim: Output feature dimension (flattened)
            conv_filters: Number of filters in each conv layer
            kernel_size: Kernel size for conv layers
            stride: Stride for conv layers

        TYPICAL CONFIGURATIONS:
        - Atari: input_channels=4 (4-frame stack), conv_filters=(32,64,64), output_dim=512
        - Robotic vision: input_channels=3 (RGB), conv_filters=(16,32,64), output_dim=256

        Example:
            >>> # Atari CNN feature extractor
            >>> cnn = CNNNetwork(
            ...     input_channels=4,
            ...     output_dim=512,
            ...     conv_filters=(32, 64, 64)
            ... )
            >>> observations = np.random.randn(32, 4, 84, 84)  # Atari frames
            >>> features = cnn(observations)  # [32, 512]
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.conv_filters = conv_filters

        raise NotImplementedError(
            "Hint: Build convolutional layers:\n"
            "1. Conv2d(input_channels, conv_filters[0], kernel_size, stride)\n"
            "2. Conv2d(conv_filters[i], conv_filters[i+1], kernel_size, stride)\n"
            "3. Flatten()\n"
            "4. Compute flattened size using forward pass\n"
            "5. Linear(flattened_size, output_dim)\n"
            "Use ReLU between layers, no activation on output"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input image array [batch_size, channels, height, width]

        Returns:
            features: [batch_size, output_dim]
        """
        raise NotImplementedError(
            "Hint: Pass through conv layers with ReLU, flatten, pass through FC layer"
        )

    @staticmethod
    def compute_conv_output_size(
        input_size: int,
        kernel_size: int,
        stride: int,
        padding: int = 0
    ) -> int:
        """
        Compute output size after convolution.

        FORMULA:
            output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1

        Args:
            input_size: Input spatial dimension (H or W)
            kernel_size: Convolution kernel size
            stride: Stride
            padding: Padding (default: 0)

        Returns:
            output_size: Output spatial dimension
        """
        raise NotImplementedError(
            "Hint: Apply standard conv output formula"
        )


class PolicyNetwork(MLPNetwork):
    """
    Policy Network for action selection.

    Maps states to action distributions or deterministic actions.

    DISCRETE ACTIONS:
        Output: Logits [batch_size, num_actions]
        Action distribution: π(a|s) = softmax(logits)

    CONTINUOUS ACTIONS:
        Output: Mean and log-std [batch_size, 2*action_dim]
        Distribution: N(μ(s), σ(s)²)
        Usually squashed with tanh for bounded actions

    USAGE:
    - Actor-Critic: policy network for actor
    - PPO: policy network for policy gradient
    - Policy gradient methods (A2C, A3C, TRPO, PPO, etc.)
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 action_type: str = "continuous"):
        """
        Initialize policy network.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
            action_type: "discrete" or "continuous"

        Example:
            >>> # Continuous action policy
            >>> policy = PolicyNetwork(state_dim=17, action_dim=6, action_type="continuous")
            >>> state = np.random.randn(32, 17)
            >>> action_dist = policy(state)  # [32, 12] (mean + logstd)
        """
        self.action_type = action_type

        # For continuous: output both mean and log-std
        output_dim = action_dim if action_type == "discrete" else 2 * action_dim

        super().__init__(
            input_dim=state_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            output_activation=None  # No output activation (raw logits/params)
        )

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            state: [batch_size, state_dim]

        Returns:
            For discrete: logits [batch_size, action_dim]
            For continuous: [mean; log_std] [batch_size, 2*action_dim]
        """
        raise NotImplementedError(
            "Hint: Return parent forward pass (already implemented)"
        )


class ValueNetwork(MLPNetwork):
    """
    Value Network for value function approximation.

    Maps states to scalar value estimates V(s).

    OUTPUTS:
        Scalar value [batch_size, 1] representing E[cumulative reward | s]

    USAGE:
    - Actor-Critic (baseline)
    - PPO (value target)
    - A2C/A3C (advantage estimation)
    - General value function approximation
    """

    def __init__(self,
                 state_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64)):
        """
        Initialize value network.

        Args:
            state_dim: State dimension
            hidden_sizes: Hidden layer sizes

        Example:
            >>> value_net = ValueNetwork(state_dim=17)
            >>> state = np.random.randn(32, 17)
            >>> value = value_net(state)  # [32, 1]
        """
        super().__init__(
            input_dim=state_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            output_activation=None  # No activation on value output
        )

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            state: [batch_size, state_dim]

        Returns:
            value: [batch_size, 1]
        """
        raise NotImplementedError(
            "Hint: Return parent forward pass"
        )


class ActorCriticNetwork(Module):
    """
    Combined Actor-Critic Network with shared trunk.

    Efficient architecture that shares feature representation between:
    - Actor (policy network)
    - Critic (value network)

    ARCHITECTURE:
                        ┌─ [Dense layers] ─ [output_dim] → action logits/params (Actor)
        state → [Dense] → shared_trunk
                        └─ [Dense layers] ─ 1 → value estimate (Critic)

    ADVANTAGES:
    - Shared features reduce parameters
    - Faster training (shared gradient flow)
    - Better feature learning (both tasks help)
    - Natural for policy gradient methods

    USED IN:
    - A2C/A3C
    - PPO (policy + value)
    - TRPO
    - Most actor-critic algorithms

    REFERENCE:
        Mnih et al. (2016): "Asynchronous Methods for Deep RL (A3C)"
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 action_type: str = "continuous"):
        """
        Initialize actor-critic network.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_sizes: Sizes of shared trunk hidden layers
            action_type: "discrete" or "continuous"

        ARCHITECTURE:
            state → shared_trunk → [policy_head] → actor_output
                                → [value_head] → value

        Example:
            >>> ac_net = ActorCriticNetwork(state_dim=17, action_dim=6)
            >>> state = np.random.randn(32, 17)
            >>> actor_out, value = ac_net(state)
            >>> # actor_out: [32, 12] (mean + logstd for continuous)
            >>> # value: [32, 1]
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_type = action_type

        raise NotImplementedError(
            "Hint: Build three components:\n"
            "1. Shared trunk (MLP): state_dim → hidden_sizes → final_hidden\n"
            "2. Policy head: final_hidden → action_output_dim\n"
            "   (action_dim for discrete, 2*action_dim for continuous)\n"
            "3. Value head: final_hidden → 1\n"
            "Save as self.trunk, self.policy_head, self.value_head"
        )

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            state: [batch_size, state_dim]

        Returns:
            policy_out: Actor output [batch_size, action_dim or 2*action_dim]
            value: Value estimate [batch_size, 1]
        """
        raise NotImplementedError(
            "Hint: Pass state through trunk, then split:\n"
            "features = self.trunk(state)\n"
            "policy_out = self.policy_head(features)\n"
            "value = self.value_head(features)\n"
            "return policy_out, value"
        )


class DQNNetwork(MLPNetwork):
    """
    Deep Q-Network (DQN) for discrete action spaces.

    Maps states to Q-values for all actions.

    OUTPUT:
        Q-values [batch_size, num_actions]
        One Q-value per action

    USED IN:
    - DQN (Atari games)
    - Double DQN
    - Dueling DQN
    - Rainbow DQN

    REFERENCE:
        Mnih et al. (2015): "Human-level control through deep RL"
    """

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 hidden_sizes: Tuple[int, ...] = (128, 128)):
        """
        Initialize DQN network.

        Args:
            state_dim: State dimension
            num_actions: Number of discrete actions
            hidden_sizes: Hidden layer sizes

        Example:
            >>> dqn = DQNNetwork(state_dim=10, num_actions=4)
            >>> state = np.random.randn(32, 10)
            >>> q_values = dqn(state)  # [32, 4]
        """
        super().__init__(
            input_dim=state_dim,
            output_dim=num_actions,
            hidden_sizes=hidden_sizes,
            output_activation=None
        )

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            state: [batch_size, state_dim]

        Returns:
            q_values: [batch_size, num_actions]
        """
        raise NotImplementedError(
            "Hint: Return parent forward pass"
        )


class DuelingQNetwork(Module):
    """
    Dueling DQN Network combining value and advantage streams.

    Separates value and advantage for more stable learning.

    ARCHITECTURE:
                        ┌─ [V-layers] ─ 1 → V(s)
        state → trunk ──┤
                        └─ [A-layers] ─ num_actions → A(s,a)

        Q(s,a) = V(s) + [A(s,a) - mean_b(A(s,b))]

    ADVANTAGES:
    - Better value learning (less action confusion)
    - More stable DQN training
    - Improved sample efficiency

    REFERENCE:
        Wang et al. (2015): "Dueling Network Architectures for Deep RL"
    """

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 hidden_sizes: Tuple[int, ...] = (128, 128),
                 value_hidden: int = 128,
                 advantage_hidden: int = 128):
        """
        Initialize dueling Q-network.

        Args:
            state_dim: State dimension
            num_actions: Number of actions
            hidden_sizes: Trunk hidden sizes
            value_hidden: Value head hidden size
            advantage_hidden: Advantage head hidden size
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        raise NotImplementedError(
            "Hint: Build:\n"
            "1. Shared trunk: MLP with hidden_sizes\n"
            "2. Value head: trunk_output → value_hidden → 1\n"
            "3. Advantage head: trunk_output → advantage_hidden → num_actions\n"
            "Save as self.trunk, self.value_head, self.advantage_head"
        )

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            state: [batch_size, state_dim]

        Returns:
            q_values: [batch_size, num_actions]

        COMPUTATION:
            1. features = trunk(state)
            2. V = value_head(features)
            3. A = advantage_head(features)
            4. Q = V + (A - mean(A))  (subtract mean advantage for stability)
        """
        raise NotImplementedError(
            "Hint: Extract features from trunk, compute V and A, "
            "combine as Q = V + (A - A.mean(axis=1, keepdims=True))"
        )


# Initialization utilities
def orthogonal_init(module: Module, gain: float = np.sqrt(2)) -> None:
    """
    Orthogonal weight initialization for neural networks.

    Good for RL (especially policy networks and recurrent networks).

    MOTIVATION:
    - Preserves input norms (helps with deep networks)
    - Works well for policy networks
    - Better than default initialization for RL

    Args:
        module: Neural network module
        gain: Scaling factor (sqrt(2) for ReLU, 1.0 for tanh)

    Example:
        >>> net = MLPNetwork(10, 5)
        >>> orthogonal_init(net, gain=np.sqrt(2))
    """
    raise NotImplementedError(
        "Hint: Initialize weights using orthogonal decomposition. "
        "Initialize biases to zero using custom module utilities"
    )


def small_init_layer(layer, scale: float = 1e-2) -> None:
    """
    Initialize final layer to be very small (for stability).

    MOTIVATION:
    - Output layers should have small initial values
    - Prevents overly large initial actions/values
    - Improves training stability

    Args:
        layer: Linear layer to initialize
        scale: Scaling factor (default: 1e-2)
    """
    raise NotImplementedError(
        "Hint: Use uniform initialization with layer weights. "
        "Initialize weights to [-scale, scale]. "
        "Set biases to zero."
    )
