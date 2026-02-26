"""
IQL - Implicit Q-Learning.

Implementation Status: STUB
Complexity: ★★★★☆ (Advanced)
Prerequisites: rl/value_based/dqn, rl/actor_critic/sac

IQL learns Q-functions using expectile regression, avoiding querying OOD actions
entirely. This makes it simple, stable, and effective for offline RL.

References:
    - Kostrikov et al. (2022): Offline Reinforcement Learning with Implicit Q-Learning
      https://arxiv.org/abs/2110.06169
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any


# =============================================================================
# THEORY: IMPLICIT Q-LEARNING
# =============================================================================
"""
THE KEY INSIGHT:
===============

The problem with offline RL is evaluating max_a Q(s', a) for the Bellman backup.
IQL avoids this by learning V(s) as an expectile of Q(s,a):

    V(s) = E_τ[Q(s,a) | a ~ β(a|s)]

where E_τ is the τ-expectile (τ close to 1 gives something like max).

EXPECTILE REGRESSION:
====================

The τ-expectile is the solution to:
    V* = argmin_V E[(τ - 1[Q < V]) * (Q - V)²]

For τ → 1, this approaches max(Q).
For τ = 0.5, this is the mean.

Key property: expectile only depends on in-distribution (s,a) pairs!

IQL ALGORITHM:
=============

1. Learn Q(s,a) via standard Bellman backup:
   Q(s,a) ← r + γ V(s')

2. Learn V(s) via expectile regression on Q:
   V(s) ← E_τ[Q(s,a) | a ~ D]

3. Extract policy via advantage-weighted regression:
   π(a|s) ∝ exp(A(s,a)/β) where A = Q - V

No need to evaluate max_a Q(s',a) - just use V(s')!

WHY THIS WORKS:
==============

- V(s) approximates V^π*(s) when τ is high
- The Bellman backup uses V(s') not max_a Q(s', a)
- Q-function is trained only on in-distribution actions
- Policy extraction avoids OOD action issues via AWR
"""


class IQL:
    """
    Implicit Q-Learning for offline RL.

    IQL avoids querying OOD actions by learning V(s) as an expectile
    of Q(s,a), eliminating the need for max_a Q(s,a) in the Bellman backup.

    Theory:
        Standard offline RL methods fail because computing max_a Q(s',a)
        requires evaluating Q for actions not in the dataset. IQL sidesteps
        this by learning a value function V(s) through expectile regression,
        which only requires evaluating Q for (s,a) pairs in the dataset.
        A high expectile (τ → 1) approximates the max without ever querying
        OOD actions.

    Mathematical Formulation:
        V-function update (expectile regression):
            L_V = E_{s,a~D}[L_τ(Q(s,a) - V(s))]
            where L_τ(u) = |τ - 1[u<0]| * u²

        Q-function update:
            L_Q = E_{s,a,r,s'~D}[(r + γV(s') - Q(s,a))²]

        Policy extraction (AWR):
            L_π = E_{s,a~D}[exp(β * A(s,a)) * (-log π(a|s))]
            where A(s,a) = Q(s,a) - V(s)

    References:
        - Kostrikov et al. (2022): Offline RL with Implicit Q-Learning
          https://arxiv.org/abs/2110.06169

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        expectile: τ for expectile regression (default 0.7)
        temperature: β for advantage weighting (default 3.0)
        discount: Discount factor
        tau: Target network update rate
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        expectile: float = 0.7,
        temperature: float = 3.0,
        discount: float = 0.99,
        tau: float = 0.005,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4
    ):
        """Initialize IQL."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.tau = tau

        # Networks
        self.q1 = None
        self.q2 = None
        self.q1_target = None
        self.q2_target = None
        self.v = None  # Value function
        self.policy = None

        self._build_networks(hidden_dims, learning_rate)

    def _build_networks(
        self,
        hidden_dims: List[int],
        learning_rate: float
    ) -> None:
        """
        Build all IQL networks.

        Implementation Hints:
            Q-networks: MLP [state, action] -> 1
            V-network: MLP [state] -> 1
            Policy: Gaussian policy with tanh squashing
        """
        raise NotImplementedError(
            "Build IQL networks:\n"
            "- Q1, Q2: [state_dim + action_dim] -> 1\n"
            "- Q1_target, Q2_target: copies\n"
            "- V: [state_dim] -> 1\n"
            "- Policy: Gaussian [state_dim] -> action_dim"
        )

    def expectile_loss(
        self,
        diff: np.ndarray
    ) -> np.ndarray:
        """
        Compute asymmetric expectile loss.

        L_τ(u) = |τ - 1[u<0]| * u²

        Args:
            diff: Q - V differences

        Returns:
            Expectile loss values
        """
        raise NotImplementedError(
            "Expectile loss:\n"
            "- weight = np.where(diff > 0, expectile, 1 - expectile)\n"
            "- Return weight * diff^2"
        )

    def compute_v_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute value function loss via expectile regression.

        L_V = E[L_τ(Q(s,a) - V(s))]

        Args:
            states: Batch of states
            actions: Batch of dataset actions

        Returns:
            v_loss: Value function loss
            info: Loss components
        """
        raise NotImplementedError(
            "V-function loss:\n"
            "- Q = min(Q1_target(s,a), Q2_target(s,a))\n"
            "- V = V(states)\n"
            "- diff = Q - V\n"
            "- loss = mean(expectile_loss(diff))\n"
            "- Return loss"
        )

    def compute_q_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Q-function loss with V(s') as target.

        L_Q = E[(r + γV(s') - Q(s,a))²]

        Note: Uses V(s') instead of max_a Q(s',a)!

        Returns:
            q_loss: Q-function loss
            info: Loss components
        """
        raise NotImplementedError(
            "Q-function loss:\n"
            "- V_next = V(next_states)\n"
            "- target = rewards + (1 - dones) * discount * V_next\n"
            "- Q1_pred = Q1(states, actions)\n"
            "- Q2_pred = Q2(states, actions)\n"
            "- loss = MSE(Q1_pred, target) + MSE(Q2_pred, target)\n"
            "- Return loss"
        )

    def compute_policy_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute policy loss via advantage-weighted regression.

        L_π = -E[exp(β * A(s,a)) * log π(a|s)]

        Returns:
            policy_loss: AWR policy loss
            info: Loss components
        """
        raise NotImplementedError(
            "AWR policy loss:\n"
            "- Q = min(Q1_target(s,a), Q2_target(s,a))\n"
            "- V = V(states)\n"
            "- A = Q - V  (advantage)\n"
            "- weights = exp(temperature * A)\n"
            "- weights = weights / mean(weights)  # normalize\n"
            "- log_prob = policy.log_prob(actions | states)\n"
            "- loss = -mean(weights * log_prob)\n"
            "- Return loss"
        )

    def update_v(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> Dict[str, float]:
        """Update value function."""
        raise NotImplementedError(
            "V update:\n"
            "- Compute V loss\n"
            "- Backprop\n"
            "- Optimizer step\n"
            "- Return loss"
        )

    def update_q(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Dict[str, float]:
        """Update Q-functions."""
        raise NotImplementedError(
            "Q update:\n"
            "- Compute Q loss\n"
            "- Backprop\n"
            "- Optimizer step\n"
            "- Return loss"
        )

    def update_policy(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> Dict[str, float]:
        """Update policy via AWR."""
        raise NotImplementedError(
            "Policy update:\n"
            "- Compute AWR policy loss\n"
            "- Backprop\n"
            "- Optimizer step\n"
            "- Return loss"
        )

    def update_targets(self) -> None:
        """Soft update target Q-networks."""
        raise NotImplementedError(
            "Soft update:\n"
            "- Q1_target = tau * Q1 + (1-tau) * Q1_target\n"
            "- Q2_target = tau * Q2 + (1-tau) * Q2_target"
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """Select action from learned policy."""
        raise NotImplementedError(
            "Select action:\n"
            "- If deterministic: policy mean\n"
            "- Else: sample from policy"
        )

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform one IQL training step.

        Order matters: V → Q → Policy → Targets

        Returns:
            All training metrics
        """
        raise NotImplementedError(
            "IQL training step:\n"
            "1. Update V (expectile regression on Q)\n"
            "2. Update Q (Bellman with V as target)\n"
            "3. Update Policy (AWR)\n"
            "4. Soft update targets\n"
            "5. Return all metrics"
        )

    def train(
        self,
        dataset: Dict[str, np.ndarray],
        n_iterations: int = 1000000,
        batch_size: int = 256,
        eval_freq: int = 5000,
        eval_env=None
    ) -> Dict[str, List]:
        """Train IQL on offline dataset."""
        raise NotImplementedError(
            "IQL training loop:\n"
            "- For each iteration:\n"
            "  - Sample batch\n"
            "  - train_step()\n"
            "  - Periodically evaluate\n"
            "- Return history"
        )
