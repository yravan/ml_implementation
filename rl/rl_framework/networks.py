"""
Neural Network Architectures for RL
====================================

Provides flexible network architectures for:
- Policy networks (discrete and continuous actions)
- Value networks (V and Q functions)
- Actor-Critic networks
- Dueling architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Optional, List


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization for network layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    """Multi-Layer Perceptron with flexible architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
        output_activation: Optional[nn.Module] = None,
        use_layer_init: bool = True
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            if use_layer_init:
                linear = layer_init(linear)
            layers.append(linear)
            layers.append(activation())
            prev_dim = hidden_dim

        # Output layer
        output_layer = nn.Linear(prev_dim, output_dim)
        if use_layer_init:
            output_layer = layer_init(output_layer, std=0.01)
        layers.append(output_layer)

        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DiscretePolicy(nn.Module):
    """Policy network for discrete action spaces using softmax."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        self.network = MLP(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns action logits."""
        return self.network(state)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return with log probability."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for given state-action pairs."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class GaussianPolicy(nn.Module):
    """Policy network for continuous action spaces using Gaussian distribution."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2,
        state_dependent_std: bool = True
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std
        self.action_dim = action_dim

        # Mean network
        self.mean_net = MLP(state_dim, action_dim, hidden_dims)

        # Standard deviation
        if state_dependent_std:
            self.log_std_net = MLP(state_dim, action_dim, hidden_dims)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and log_std of the action distribution."""
        mean = self.mean_net(state)

        if self.state_dependent_std:
            log_std = self.log_std_net(state)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return with log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            dist = Normal(mean, std)
            action = dist.rsample()  # Reparameterization trick
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for given state-action pairs."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class SquashedGaussianPolicy(nn.Module):
    """
    Gaussian policy with tanh squashing for bounded action spaces.
    Used in SAC and other algorithms requiring bounded actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2,
        action_scale: float = 1.0,
        action_bias: float = 0.0
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = action_scale
        self.action_bias = action_bias

        # Shared feature network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, hidden_dim)))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = layer_init(nn.Linear(prev_dim, action_dim), std=0.01)
        self.log_std_head = layer_init(nn.Linear(prev_dim, action_dim), std=0.01)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action_raw = mean
        else:
            dist = Normal(mean, std)
            action_raw = dist.rsample()

        # Apply tanh squashing
        action = torch.tanh(action_raw) * self.action_scale + self.action_bias

        # Compute log probability with correction for tanh
        if deterministic:
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            # Correction for tanh squashing
            log_prob -= (2 * (np.log(2) - action_raw - F.softplus(-2 * action_raw))).sum(dim=-1)

        return action, log_prob


class ValueNetwork(nn.Module):
    """State value function V(s)."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        self.network = MLP(state_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class QNetwork(nn.Module):
    """Action-value function Q(s, a) for continuous actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        self.network = MLP(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x).squeeze(-1)


class DiscreteQNetwork(nn.Module):
    """Action-value function Q(s, a) for discrete actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        self.network = MLP(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions."""
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()

        # Shared feature network
        self.feature_net = MLP(state_dim, hidden_dims[-1], hidden_dims[:-1])

        # Value stream
        self.value_stream = nn.Sequential(
            layer_init(nn.Linear(hidden_dims[-1], hidden_dims[-1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dims[-1], 1), std=0.01)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            layer_init(nn.Linear(hidden_dims[-1], hidden_dims[-1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dims[-1], action_dim), std=0.01)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine using mean subtraction for identifiability
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class TwinQNetwork(nn.Module):
    """Twin Q-networks for TD3 and SAC (clipped double Q-learning)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(state, action)


class ActorCritic(nn.Module):
    """Combined Actor-Critic network with shared features."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        continuous: bool = False
    ):
        super().__init__()
        self.continuous = continuous

        # Shared feature extractor
        self.feature_net = MLP(state_dim, hidden_dims[-1], hidden_dims[:-1])

        # Actor head
        if continuous:
            self.actor_mean = layer_init(nn.Linear(hidden_dims[-1], action_dim), std=0.01)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor = layer_init(nn.Linear(hidden_dims[-1], action_dim), std=0.01)

        # Critic head
        self.critic = layer_init(nn.Linear(hidden_dims[-1], 1), std=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.feature_net(state))
        value = self.critic(features).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(features)
            log_std = self.actor_log_std.expand_as(mean)
            return (mean, log_std), value
        else:
            logits = self.actor(features)
            return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns action, log_prob, entropy, and value.
        If action is provided, evaluates that action instead of sampling.
        """
        features = F.relu(self.feature_net(state))
        value = self.critic(features).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(features)
            log_std = self.actor_log_std.expand_as(mean)
            std = log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                action = mean if deterministic else dist.rsample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.actor(features)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if action is None:
                action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value


# Bandit-specific networks
class LinearUCBModel(nn.Module):
    """Linear model for LinUCB contextual bandits with ridge regression."""

    def __init__(self, context_dim: int, n_arms: int, alpha: float = 1.0, lambda_reg: float = 1.0):
        super().__init__()
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # Initialize A matrices (one per arm) and b vectors
        self.register_buffer('A', torch.eye(context_dim).unsqueeze(0).repeat(n_arms, 1, 1) * lambda_reg)
        self.register_buffer('b', torch.zeros(n_arms, context_dim))

    def update(self, arm: int, context: torch.Tensor, reward: float):
        """Update the model with observed reward."""
        context = context.view(-1)
        self.A[arm] += torch.outer(context, context)
        self.b[arm] += reward * context

    def get_ucb(self, context: torch.Tensor) -> torch.Tensor:
        """Compute UCB values for all arms."""
        context = context.view(-1)
        ucb_values = torch.zeros(self.n_arms)

        for arm in range(self.n_arms):
            A_inv = torch.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]

            # UCB = theta^T x + alpha * sqrt(x^T A^{-1} x)
            mean = torch.dot(theta, context)
            uncertainty = self.alpha * torch.sqrt(context @ A_inv @ context)
            ucb_values[arm] = mean + uncertainty

        return ucb_values
