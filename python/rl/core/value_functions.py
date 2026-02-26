"""
Value Function Approximators for Reinforcement Learning

This module defines classes for estimating state values V(s) and action values Q(s,a),
which are central to many RL algorithms.

THEORY:
    Value functions estimate expected future rewards:

    1. State Value Function V(s):
       V(s) = E_π[∑_t γ^t r_t | s_0 = s]
       Expected cumulative discounted reward from state s under policy π

    2. Action Value Function Q(s,a):
       Q(s,a) = E_π[∑_t γ^t r_t | s_0 = s, a_0 = a]
       Expected cumulative reward from taking action a in state s

    3. Advantage Function A(s,a):
       A(s,a) = Q(s,a) - V(s)
       How much better is action a than average?

    These satisfy the Bellman equations:
    V(s) = E_a[r + γ V(s')] = ∑_a π(a|s) Q(s,a)
    Q(s,a) = E[r + γ V(s')]
    Q(s,a) = E[r + γ ∑_a' π(a'|s') Q(s',a')]

APPROXIMATORS:
    Tabular: Store exact values for small, discrete state/action spaces
    Linear: V(s) = w^T φ(s) where φ(s) are features
    Neural Networks: V(s) = network(s), Q(s,a) = network(s,a)

REFERENCES:
    - Sutton & Barto (2018), Chapters 3-6
    - Spinning Up on value functions
    - Dueling DQN: Wang et al. (2015)
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
from abc import ABC, abstractmethod
from python.nn_core import Module


class BaseValueFunction(ABC):
    """Abstract base class for value function approximators."""

    @abstractmethod
    def predict(self, states: Union[np.ndarray]) -> np.ndarray:
        """
        Predict value(s) for state(s).

        Args:
            states: State(s) to evaluate

        Returns:
            values: Predicted values
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, states: Union[np.ndarray],
               targets: Union[np.ndarray]) -> float:
        """
        Update value function using regression targets.

        Args:
            states: Batch of states
            targets: Target values (e.g., from Bellman backup)

        Returns:
            loss: Scalar loss for monitoring
        """
        raise NotImplementedError()


class TabularV(BaseValueFunction):
    """
    Tabular State Value Function for discrete state spaces.

    Stores exact V(s) in a lookup table. Suitable only for
    small state spaces (e.g., GridWorld).

    IMPLEMENTATION:
        V(s) = table[s]  where s is a discrete state index

    BELLMAN UPDATE:
        V(s) ← V(s) + α [r + γ V(s') - V(s)]  (TD update)
        V(s) ← average of returns from state s  (Monte Carlo)
    """

    def __init__(self,
                 num_states: int,
                 learning_rate: float = 0.1,
                 initialization: str = "zeros"):
        """
        Initialize tabular value function.

        Args:
            num_states: Number of discrete states
            learning_rate: Learning rate α for value updates
            initialization: "zeros" or "random"

        Example:
            >>> v = TabularV(num_states=100, learning_rate=0.1)
            >>> value = v.predict(state=5)  # Get V(state=5)
            >>> loss = v.update(states=[5,6,7], targets=[1.5, 2.0, 0.8])
        """
        self.num_states = num_states
        self.learning_rate = learning_rate

        if initialization == "zeros":
            self.table = np.zeros(num_states)
        elif initialization == "random":
            self.table = np.random.randn(num_states) * 0.1
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

    def predict(self, states: Union[np.ndarray, int]) -> np.ndarray:
        """
        Look up values in table.

        Args:
            states: State index/indices

        Returns:
            values: V(s) from lookup table
        """
        raise NotImplementedError(
            "Hint: If states is int, return self.table[states]. "
            "If array, return self.table[states]"
        )

    def update(self, states: np.ndarray, targets: np.ndarray) -> float:
        """
        Update values using learning rate and targets.

        Args:
            states: State indices [batch_size]
            targets: Target values [batch_size]

        Returns:
            loss: MSE between predictions and targets

        MATH:
            loss = mean((V(s) - target)^2)
            V(s) ← V(s) + α (target - V(s))
        """
        raise NotImplementedError(
            "Hint: Compute predictions, compute MSE loss, update table "
            "values using: self.table[states] += lr * (targets - predictions)"
        )


class TabularQ(BaseValueFunction):
    """
    Tabular Action Value Function for discrete state and action spaces.

    Stores exact Q(s,a) in a lookup table.

    BELLMAN EQUATION:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]  (Q-learning)
        Q(s,a) ← Q(s,a) + α [r + γ ∑_a' π(a'|s') Q(s',a') - Q(s,a)]  (SARSA)
    """

    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 learning_rate: float = 0.1):
        """
        Initialize tabular Q-function.

        Args:
            num_states: Number of discrete states
            num_actions: Number of discrete actions
            learning_rate: Learning rate α for Q-value updates

        Example:
            >>> q = TabularQ(num_states=100, num_actions=4)
            >>> q_sa = q.predict(state=5, action=2)
            >>> update_loss = q.update(states=[5,6], actions=[2,1], targets=[1.5, 2.0])
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.table = np.zeros((num_states, num_actions))

    def predict(self,
                states: Union[np.ndarray, int],
                actions: Optional[Union[np.ndarray, int]] = None) -> np.ndarray:
        """
        Look up Q-values in table.

        Args:
            states: State index/indices [batch_size]
            actions: Action index/indices [batch_size]
                    If None, return all Q(s,:) for all actions

        Returns:
            q_values: Q(s,a) or Q(s,:) from table
        """
        raise NotImplementedError(
            "Hint: If actions is None, return self.table[states]. "
            "Otherwise return self.table[states, actions]"
        )

    def update(self,
               states: np.ndarray,
               actions: np.ndarray,
               targets: np.ndarray) -> float:
        """
        Update Q-values using TD backup.

        Args:
            states: State indices [batch_size]
            actions: Action indices [batch_size]
            targets: Target Q-values [batch_size] (from Bellman backup)

        Returns:
            loss: MSE loss
        """
        raise NotImplementedError(
            "Hint: Compute predictions Q(s,a), compute MSE with targets, "
            "update: self.table[states, actions] += lr * (targets - predictions)"
        )


class NeuralV(BaseValueFunction):
    """
    Neural Network State Value Function.

    Uses a neural network to approximate V(s).
    Suitable for continuous or large discrete state spaces.

    ARCHITECTURE:
        state → [Dense] → [ReLU] → [Dense] → ... → [Dense] → V(s)

    TRAINING:
        Minimize MSE: L = mean((V(s) - target)^2)
        Using gradient descent on network parameters θ
    """

    def __init__(self,
                 state_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 learning_rate: float = 1e-3,
                 optimizer: str = "adam"):
        """
        Initialize neural value function.

        Args:
            state_dim: Dimension of state space
            hidden_sizes: Sizes of hidden layers (default: 2 layers of 64)
            learning_rate: Learning rate for optimizer
            optimizer: "adam" or "sgd"

        Example:
            >>> v = NeuralV(state_dim=10, hidden_sizes=(64, 64))
            >>> state = np.random.randn(32, 10)  # batch of 32 states
            >>> value = v.predict(state)  # [32, 1]
            >>> loss = v.update(state, targets=np.random.randn(32, 1))
        """
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate

        # Build network: state_dim -> hidden -> hidden -> 1 (scalar value)
        raise NotImplementedError(
            "Hint: Create a sequential network with Linear layers, "
            "ReLU activations, ending with single output. Save as self.network. "
            "Create optimizer (Adam or SGD) and save as self.optimizer"
        )

    def predict(self, states: Union[np.ndarray]) -> np.ndarray:
        """
        Predict value for state(s).

        Args:
            states: State(s) [state_dim] or [batch_size, state_dim]

        Returns:
            values: Predicted values [1] or [batch_size, 1]
        """
        raise NotImplementedError(
            "Hint: Convert states to torch tensor if needed, pass through network, "
            "detach and convert to numpy"
        )

    def update(self, states: Union[np.ndarray],
               targets: Union[np.ndarray]) -> float:
        """
        Update network using MSE loss.

        Args:
            states: Batch of states [batch_size, state_dim]
            targets: Target values [batch_size, 1]

        Returns:
            loss: Scalar MSE loss for monitoring
        """
        raise NotImplementedError(
            "Hint: Convert to tensors, do forward pass, compute MSE loss, "
            "backward, optimizer.step(), zero_grad()"
        )


class NeuralQ(BaseValueFunction):
    """
    Neural Network Action Value Function.

    Uses a neural network to approximate Q(s,a).

    ARCHITECTURES:
        1. State-Action: (s,a) → network → Q(s,a)
        2. State-Only: s → network → [Q(s,1), ..., Q(s,|A|)]

    Used in:
    - DQN
    - Double DQN
    - Dueling DQN
    - Rainbow DQN
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 learning_rate: float = 1e-3,
                 is_discrete: bool = False):
        """
        Initialize neural Q-function.

        Args:
            state_dim: State dimension
            action_dim: Action dimension (|A| for discrete, action_dim for continuous)
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate
            is_discrete: If True, output |A| Q-values for each state
                        If False, concatenate state and action as input

        Example:
            >>> # Discrete actions
            >>> q = NeuralQ(state_dim=10, action_dim=4, is_discrete=True)
            >>> state = np.random.randn(32, 10)
            >>> q_values = q.predict(state)  # [32, 4]
            >>> loss = q.update(state, actions, targets)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.learning_rate = learning_rate

        raise NotImplementedError(
            "Hint: If is_discrete, build network: state_dim → hidden → action_dim. "
            "Otherwise: (state_dim + action_dim) → hidden → 1. "
            "Create Adam optimizer."
        )

    def predict(self, states: Union[np.ndarray],
                actions: Optional[Union[np.ndarray]] = None) -> np.ndarray:
        """
        Predict Q-values.

        Args:
            states: States [batch_size, state_dim]
            actions: Actions [batch_size, action_dim] (required if not is_discrete)

        Returns:
            q_values: Q(s,a) values [batch_size, 1] or [batch_size, action_dim]
        """
        raise NotImplementedError(
            "Hint: If is_discrete, just pass states through network. "
            "If continuous, concatenate states and actions, pass through network."
        )

    def update(self, states: Union[np.ndarray],
               actions: Union[np.ndarray],
               targets: Union[np.ndarray]) -> float:
        """
        Update Q-function using TD targets.

        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            targets: TD targets [batch_size, 1] from: r + γ * Q_target(s', a')

        Returns:
            loss: MSE loss
        """
        raise NotImplementedError(
            "Hint: Predict Q(s,a), compute MSE(predictions, targets), "
            "backprop and update optimizer"
        )


class DuelingNetwork(Module):
    """
    Dueling Network Architecture for Q-learning (DQN variant).

    Separates value and advantage streams:
    Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]

    MOTIVATION:
    - Many actions have similar Q-values in a state
    - Separating V and A allows learning V more efficiently
    - Advantage is relative to average, improves stability

    MATH:
        Q(s,a) = V(s) + A(s,a) - mean_b A(s,b)

    where V and A share a common feature network but have
    separate output heads.

    REFERENCE:
    - Wang et al. (2015): "Dueling Network Architectures for Deep RL"
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 feature_dim: int = 128,
                 value_hidden: int = 128,
                 advantage_hidden: int = 128):
        """
        Initialize dueling network.

        Args:
            state_dim: Dimension of state
            action_dim: Number of actions (discrete)
            feature_dim: Dimension of shared feature layer
            value_hidden: Hidden dimension for V(s) head
            advantage_hidden: Hidden dimension for A(s,a) head

        Architecture:
            state → [feature layers] → feature_vector
                                    ├→ [V-layers] → V(s) [1]
                                    └→ [A-layers] → A(s,a) [action_dim]
            Q(s,a) = V(s) + [A(s,a) - mean(A)]

        Example:
            >>> dueling_q = DuelingNetwork(state_dim=10, action_dim=4)
            >>> states = np.random.randn(32, 10)
            >>> q_values = dueling_q(states)  # [32, 4]
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        raise NotImplementedError(
            "Hint: Build 3 components:\n"
            "1. Feature network: state_dim → feature_dim with ReLU\n"
            "2. Value head: feature_dim → value_hidden → 1 (scalar V)\n"
            "3. Advantage head: feature_dim → advantage_hidden → action_dim (advantage for each action)\n"
            "In forward(), combine: Q(s,a) = V(s) + [A(s,a) - A(s,:).mean()]"
        )

    def forward(self, states: np.ndarray) -> np.ndarray:
        """
        Forward pass combining V and A into Q.

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            q_values: Q-values [batch_size, action_dim]

        COMPUTATION:
            1. Extract features: f = feature_net(s)
            2. Compute V(s) = value_head(f)  # scalar
            3. Compute A(s,a) = advantage_head(f)  # vector of |A| elements
            4. Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        """
        raise NotImplementedError(
            "Hint: Pass states through feature network, compute both V and A heads, "
            "then combine using: Q = V + (A - A.mean(axis=-1, keepdims=True))"
        )


class DoubleQNetwork(Module):
    """
    Double Q-learning Network to reduce overestimation bias.

    Maintains two independent Q-networks to decouple selection and evaluation
    of next actions in the Bellman backup.

    MOTIVATION:
        Standard Q-learning uses: target = r + γ max_a Q(s',a')
        This can overestimate Q-values because max and expectation don't commute.

        Double Q-learning uses two networks:
        - Network θ for behavior (agent action selection)
        - Network θ' for evaluation (computing target)

    MATH:
        Standard: target = r + γ max_a Q_θ(s',a')
        Double: target = r + γ Q_θ'(s', argmax_a Q_θ(s',a'))

    Used in:
    - Double DQN
    - TD3
    - SAC

    REFERENCE:
    - van Hasselt et al. (2015): "Deep Reinforcement Learning with Double Q-learning"
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64)):
        """
        Initialize double Q-network.

        Args:
            state_dim: Dimension of state
            action_dim: Number of discrete actions
            hidden_sizes: Hidden layer dimensions

        Example:
            >>> main_q = DoubleQNetwork(state_dim=10, action_dim=4)
            >>> target_q = DoubleQNetwork(state_dim=10, action_dim=4)
            >>> # Update target_q periodically: target_q.load_state_dict(main_q.state_dict())
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        raise NotImplementedError(
            "Hint: Build a standard Q-network: state_dim → hidden → action_dim"
        )

    def forward(self, states: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions.

        Args:
            states: Batch of states [batch_size, state_dim]

        Returns:
            q_values: Q-values for all actions [batch_size, action_dim]
        """
        raise NotImplementedError(
            "Hint: Pass states through network layers with ReLU activations, "
            "output action_dim Q-values"
        )

    def compute_td_target(self,
                         rewards: np.ndarray,
                         next_states: np.ndarray,
                         dones: np.ndarray,
                         target_network: 'DoubleQNetwork',
                         gamma: float = 0.99) -> np.ndarray:
        """
        Compute TD target using Double Q-learning.

        Args:
            rewards: Rewards [batch_size, 1]
            next_states: Next states [batch_size, state_dim]
            dones: Done flags [batch_size, 1]
            target_network: Target network (parameter copy θ')
            gamma: Discount factor

        Returns:
            td_target: r + γ * Q_target(s', a*) where a* = argmax Q(s', ·)

        DOUBLE Q-LEARNING EQUATION:
            1. Use main network to select best action: a* = argmax_a Q_θ(s',a)
            2. Use target network to evaluate: Q_θ'(s', a*)
            3. Target = r + γ * Q_θ'(s', a*)
        """
        raise NotImplementedError(
            "Hint: Use self to select best actions: argmax(self(next_states)). "
            "Use target_network to evaluate: target_network(next_states). "
            "Select Q-values for best actions. Compute: r + gamma * Q_target * (1 - dones)"
        )


# Aliases for common naming conventions
QTable = TabularQ
ValueNetwork = NeuralV

