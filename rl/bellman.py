"""
Bellman equations and dynamic programming for model-based RL.

Covers the Bellman expectation/optimality equations, policy evaluation,
value iteration, and policy iteration.

Key equations:
    Bellman expectation (for policy π):
        V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]

    Bellman optimality:
        V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]

    Q-value form:
        Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a')

Assumptions:
    - Finite MDP with |S| states and |A| actions
    - P[s', s, a] = probability of transitioning to s' given state s, action a
    - R[s, a] = expected immediate reward for taking action a in state s
    - policy[s, a] = π(a|s), probability of taking action a in state s
"""

import numpy as np


def bellman_backup_v(V, policy, P, R, gamma):
    """
    One full sweep of the Bellman expectation backup for V^π.

    For each state s:
        V_new(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]

    Parameters:
        V: np.ndarray of shape (n_states,) - Current value function estimate.
        policy: np.ndarray of shape (n_states, n_actions) - π(a|s).
        P: np.ndarray of shape (n_states, n_states, n_actions) - P[s', s, a].
        R: np.ndarray of shape (n_states, n_actions) - R(s, a).
        gamma: float - Discount factor in [0, 1).

    Returns:
        V_new: np.ndarray of shape (n_states,) - Updated value function.
    """

    next_state_value = V[:, None, None] # n_states, n_states, n_actions
    expected_next_state_value = (P * V[:, None, None]).sum(axis = 0) # n_states, n_actions
    expected_action_value = R + expected_next_state_value * gamma
    expected_value = (policy * expected_action_value).sum(axis = 1)
    V_new = expected_value
    return V_new


def bellman_backup_q(Q, P, R, gamma):
    """
    One full sweep of the Bellman optimality backup for Q*.

    For each (s, a):
        Q_new(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q(s', a')

    Parameters:
        Q: np.ndarray of shape (n_states, n_actions) - Current Q-value estimate.
        P: np.ndarray of shape (n_states, n_states, n_actions) - P[s', s, a].
        R: np.ndarray of shape (n_states, n_actions) - R(s, a).
        gamma: float - Discount factor in [0, 1).

    Returns:
        Q_new: np.ndarray of shape (n_states, n_actions) - Updated Q-values.
    """
    expected_next_state_value = (P * np.max(Q,axis=1)[:, None, None]).sum(axis = 0)
    expected_action_value = R + expected_next_state_value * gamma
    Q_new = expected_action_value
    return Q_new


def policy_evaluation(policy, P, R, gamma, tol=1e-6, max_iter=1000):
    """
    Iterative policy evaluation: compute V^π by repeated Bellman backups.

    Iterate V <- T^π V until ||V_new - V||_∞ < tol.

    Parameters:
        policy: np.ndarray of shape (n_states, n_actions) - π(a|s).
        P: np.ndarray of shape (n_states, n_states, n_actions) - P[s', s, a].
        R: np.ndarray of shape (n_states, n_actions) - R(s, a).
        gamma: float - Discount factor in [0, 1).
        tol: float - Convergence threshold on ||V_new - V||_∞.
        max_iter: int - Maximum number of iterations.

    Returns:
        V: np.ndarray of shape (n_states,) - Converged value function V^π.
        n_iter: int - Number of iterations until convergence.
    """
    n_states, n_actions = policy.shape
    V_old = np.zeros((n_states))
    for n_iter in range(max_iter):
        V = bellman_backup_v(V_old, policy, P, R, gamma)
        if np.absolute(V_old- V).max() < tol:
            break
        V_old = V
    return V, n_iter


def extract_greedy_policy(V, P, R, gamma):
    """
    Extract a greedy deterministic policy from a value function.

    For each state s:
        π(s) = argmax_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]

    The returned policy is deterministic: policy[s, a] = 1 if a is the
    greedy action, 0 otherwise.

    Parameters:
        V: np.ndarray of shape (n_states,) - Value function.
        P: np.ndarray of shape (n_states, n_states, n_actions) - P[s', s, a].
        R: np.ndarray of shape (n_states, n_actions) - R(s, a).
        gamma: float - Discount factor.

    Returns:
        policy: np.ndarray of shape (n_states, n_actions) - Greedy policy.
    """
    n_states, n_actions = R.shape
    expected_next_state_value = (P * V[:, None, None]).sum(axis = 0)
    expected_action_value = R + expected_next_state_value * gamma
    best_action = np.argmax(expected_action_value, axis = 1)
    policy = np.eye(n_actions)[best_action]
    return policy


def value_iteration(P, R, gamma, tol=1e-6, max_iter=1000):
    """
    Value iteration: compute V* via repeated Bellman optimality backups.

    For each state s:
        V_new(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]

    Iterate until ||V_new - V||_∞ < tol, then extract the greedy policy.

    Parameters:
        P: np.ndarray of shape (n_states, n_states, n_actions) - P[s', s, a].
        R: np.ndarray of shape (n_states, n_actions) - R(s, a).
        gamma: float - Discount factor in [0, 1).
        tol: float - Convergence threshold.
        max_iter: int - Maximum iterations.

    Returns:
        V: np.ndarray of shape (n_states,) - Optimal value function V*.
        policy: np.ndarray of shape (n_states, n_actions) - Optimal policy π*.
        n_iter: int - Number of iterations.
    """
    Q_old = np.zeros_like(R)
    for n_iter in range(max_iter):
        print(Q_old)
        Q = bellman_backup_q(Q_old, P, R, gamma)
        if np.absolute(Q_old - Q).max() < tol:
            break
        Q_old = Q
    V = np.max(Q_old, axis = 1)
    print(V)
    policy = extract_greedy_policy(V, P, R, gamma)
    return V, policy, n_iter


def policy_iteration(P, R, gamma, tol=1e-6, max_iter=100):
    """
    Policy iteration: alternate between policy evaluation and improvement.

    1. Policy evaluation: compute V^π (using iterative policy evaluation).
    2. Policy improvement: π_new = greedy(V^π).
    3. If π_new == π, stop. Otherwise π <- π_new and go to 1.

    Parameters:
        P: np.ndarray of shape (n_states, n_states, n_actions) - P[s', s, a].
        R: np.ndarray of shape (n_states, n_actions) - R(s, a).
        gamma: float - Discount factor in [0, 1).
        tol: float - Tolerance for policy evaluation convergence.
        max_iter: int - Maximum outer iterations (policy improvement steps).

    Returns:
        V: np.ndarray of shape (n_states,) - Optimal value function.
        policy: np.ndarray of shape (n_states, n_actions) - Optimal policy.
        n_iter: int - Number of policy improvement iterations.
    """
    n_states, n_actions = R.shape
    policy_old = np.zeros((n_states, n_actions))
    policy_old[:,:] = 1/n_actions
    for n_iter in range(max_iter):
        V, _ = policy_evaluation(policy_old, P, R, gamma, tol, max_iter)
        policy = extract_greedy_policy(V, P, R, gamma)
        if np.array_equal(policy, policy_old):
            break
        policy_old = policy
    V, _ = policy_evaluation(policy_old, P, R, gamma, tol, max_iter)
    return V, policy, n_iter
