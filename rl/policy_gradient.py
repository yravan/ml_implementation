"""
Policy gradient methods: REINFORCE and advantage estimation.

Key ideas:
    - Directly parameterize and optimize the policy π_θ(a|s).
    - Policy gradient theorem:
        ∇_θ J(θ) = E_π [∇_θ log π_θ(a|s) · G_t]
    - Variance reduction via baselines:
        ∇_θ J(θ) = E_π [∇_θ log π_θ(a|s) · (G_t - b(s))]
    - Advantage function A(s,a) = Q(s,a) - V(s) further reduces variance.
    - GAE (Generalized Advantage Estimation) interpolates between
      bias and variance via parameter λ.

REINFORCE algorithm:
    1. Collect a full trajectory τ = (s_0, a_0, r_0, s_1, ..., s_T).
    2. Compute returns G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}.
    3. Update: θ <- θ + α Σ_t ∇_θ log π_θ(a_t|s_t) · G_t.
"""

import numpy as np


def discounted_returns(rewards, gamma):
    """
    Compute discounted returns G_t for each timestep in an episode.

        G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ... + γ^{T-t} r_T

    Efficient computation works backwards:
        G_T = r_T
        G_t = r_t + γ G_{t+1}

    Parameters:
        rewards: np.ndarray of shape (T,) - Rewards [r_0, r_1, ..., r_{T-1}].
        gamma: float - Discount factor in [0, 1].

    Returns:
        returns: np.ndarray of shape (T,) - Discounted returns [G_0, G_1, ..., G_{T-1}].
    """
    if gamma == 0:
        return rewards
    powers = gamma ** np.arange(len(rewards))
    discounted_rewards = rewards * powers
    G = np.cumsum(discounted_rewards[::-1], axis=0)[::-1]
    G = G / powers
    returns = G
    return returns


def softmax_policy(theta, state_features):
    """
    Compute action probabilities under a softmax (linear) policy.

        π(a|s) = exp(θ_a^T φ(s)) / Σ_{a'} exp(θ_{a'}^T φ(s))

    Parameters:
        theta: np.ndarray of shape (n_actions, n_features) - Policy parameters.
        state_features: np.ndarray of shape (n_features,) - Feature vector φ(s).

    Returns:
        probs: np.ndarray of shape (n_actions,) - Action probabilities (sums to 1).
    """
    single_timestep = False
    if state_features.ndim == 1:
        state_features = state_features[np.newaxis, :]
        single_timestep = True
    logits = (theta @ state_features.T).T
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    if single_timestep:
        probs = probs[0, :]
    return probs


def log_softmax_policy(theta, state_features):
    """
    Compute log action probabilities under a softmax policy (numerically stable).

        log π(a|s) = θ_a^T φ(s) - log Σ_{a'} exp(θ_{a'}^T φ(s))

    Uses the log-sum-exp trick for stability.

    Parameters:
        theta: np.ndarray of shape (n_actions, n_features) - Policy parameters.
        state_features: np.ndarray of shape (n_features,) - Feature vector φ(s).

    Returns:
        log_probs: np.ndarray of shape (n_actions,) - Log action probabilities.
    """
    single_timestep = False
    if state_features.ndim == 1:
        state_features = state_features[np.newaxis, :]
        single_timestep = True
    logits = (theta @ state_features.T).T
    total_prob = np.sum(np.exp(logits), axis=1, keepdims=True)
    log_probs = logits - np.log(total_prob)
    if single_timestep:
        log_probs = log_probs[0, :]
    return log_probs


def reinforce_loss(log_probs, returns):
    """
    Compute the REINFORCE (policy gradient) loss.

    The loss is the negative of the policy gradient objective:
        L = -Σ_t log π_θ(a_t|s_t) · G_t

    Minimizing this loss maximizes expected return.

    Parameters:
        log_probs: np.ndarray of shape (T,) - log π(a_t|s_t) for each timestep.
        returns: np.ndarray of shape (T,) - Discounted returns G_t.

    Returns:
        loss: float - Scalar REINFORCE loss (to be minimized).
    """
    loss = - (log_probs * returns).sum()
    return loss


def reinforce_gradient(theta, states, actions, returns):
    """
    Compute the REINFORCE policy gradient for a softmax policy.

    ∇_θ J ≈ Σ_t (φ(s_t) (e_{a_t} - π_θ(·|s_t))) · G_t

    For the softmax policy, the score function ∇_θ log π(a|s) is:
        ∇_{θ_a} log π(a|s) = φ(s) (1_{a=a_t} - π(a|s))

    Parameters:
        theta: np.ndarray of shape (n_actions, n_features) - Policy parameters.
        states: np.ndarray of shape (T, n_features) - State features for each timestep.
        actions: np.ndarray of shape (T,) dtype int - Actions taken at each timestep.
        returns: np.ndarray of shape (T,) - Discounted returns G_t.

    Returns:
        grad: np.ndarray of shape (n_actions, n_features) - Policy gradient ∇_θ J.
    """
    T = len(states)
    n_actions = theta.shape[0]
    action_dist = softmax_policy(theta, states) # T, n_actions
    probs = action_dist[np.arange(T), actions] # T,
    actions = np.eye(n_actions)[actions] # T, n_actions -> one hot
    grad = states.T @ ((actions - probs[..., None]) * returns[..., None])
    grad = grad.T
    return grad


def gae(rewards, values, gamma, lam):
    """
    Generalized Advantage Estimation (GAE).

    GAE(γ, λ) interpolates between high-bias/low-variance (λ=0, TD residual)
    and low-bias/high-variance (λ=1, Monte Carlo) advantage estimates.

        δ_t = r_t + γ V(s_{t+1}) - V(s_t)          (TD error)
        A_t^GAE = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}   (exponentially weighted sum)

    Efficient backward computation:
        A_{T-1} = δ_{T-1}
        A_t = δ_t + γ λ A_{t+1}

    Parameters:
        rewards: np.ndarray of shape (T,) - Rewards [r_0, ..., r_{T-1}].
        values: np.ndarray of shape (T+1,) - Value estimates [V(s_0), ..., V(s_T)].
            The last element V(s_T) is the bootstrap value (0 if terminal).
        gamma: float - Discount factor.
        lam: float - GAE parameter λ in [0, 1].

    Returns:
        advantages: np.ndarray of shape (T,) - GAE advantage estimates.
    """
    advantages = None
    return advantages
