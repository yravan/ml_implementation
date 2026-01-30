"""
Proximal Policy Optimization (PPO).

PPO constrains policy updates to prevent destructively large steps.
It uses a clipped surrogate objective instead of a hard KL constraint.

Key equations:
    Importance sampling ratio:
        r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

    PPO-Clip objective (per timestep):
        L_t^CLIP = min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)

    Total PPO loss:
        L = -E_t[L_t^CLIP] + c_1 L_value - c_2 H[π_θ]

    where:
        - A_t is the advantage estimate (e.g., from GAE)
        - ε is the clip range (typically 0.1-0.3)
        - L_value = (V_θ(s_t) - G_t)² is the value function loss
        - H is the entropy bonus for exploration

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import numpy as np


def compute_ratio(new_log_probs, old_log_probs):
    """
    Compute the importance sampling ratio between new and old policies.

        r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
                = exp(log π_θ(a_t|s_t) - log π_{θ_old}(a_t|s_t))

    Parameters:
        new_log_probs: np.ndarray of shape (T,) - log π_θ(a_t|s_t).
        old_log_probs: np.ndarray of shape (T,) - log π_{θ_old}(a_t|s_t).

    Returns:
        ratio: np.ndarray of shape (T,) - Importance sampling ratios.
    """
    ratio = None
    return ratio


def ppo_clipped_objective(ratio, advantages, epsilon):
    """
    Compute the PPO clipped surrogate objective for each timestep.

        L_t = min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)

    When A_t > 0 (good action): caps r_t at (1+ε) so the policy can't
    move too far toward this action.
    When A_t < 0 (bad action): caps r_t at (1-ε) so the policy can't
    move too far away from this action.

    Parameters:
        ratio: np.ndarray of shape (T,) - Importance sampling ratios r_t.
        advantages: np.ndarray of shape (T,) - Advantage estimates A_t.
        epsilon: float - Clip range ε (typically 0.1-0.3).

    Returns:
        objective: np.ndarray of shape (T,) - Per-timestep clipped objective.
    """
    objective = None
    return objective


def ppo_loss(new_log_probs, old_log_probs, advantages, epsilon):
    """
    Compute the full PPO policy loss (to be minimized).

    Loss = -mean(L_t^CLIP)

    This is the negative of the clipped objective, averaged over timesteps.

    Parameters:
        new_log_probs: np.ndarray of shape (T,) - log π_θ(a_t|s_t).
        old_log_probs: np.ndarray of shape (T,) - log π_{θ_old}(a_t|s_t).
        advantages: np.ndarray of shape (T,) - Advantage estimates A_t.
        epsilon: float - Clip range.

    Returns:
        loss: float - PPO policy loss (scalar, to be minimized).
    """
    loss = None
    return loss


def value_function_loss(predicted_values, target_returns):
    """
    Compute the value function loss (MSE).

        L_value = mean((V_θ(s_t) - G_t)²)

    Parameters:
        predicted_values: np.ndarray of shape (T,) - V_θ(s_t).
        target_returns: np.ndarray of shape (T,) - Target returns G_t.

    Returns:
        loss: float - Mean squared error.
    """
    loss = None
    return loss


def entropy_bonus(probs):
    """
    Compute the entropy of a discrete action distribution.

        H(π) = -Σ_a π(a) log π(a)

    Higher entropy encourages exploration. The entropy bonus is added
    to the PPO objective to prevent premature convergence.

    Parameters:
        probs: np.ndarray of shape (n_actions,) or (T, n_actions)
            Action probabilities π(a|s). Must sum to 1 along last axis.

    Returns:
        entropy: float or np.ndarray of shape (T,) - Entropy value(s).
    """
    entropy = None
    return entropy
