"""
Model-free value-based RL: Q-learning, SARSA, and TD methods.

Key ideas:
    - TD learning uses bootstrapping: estimate values from other estimates.
    - Q-learning is off-policy (uses max over next actions).
    - SARSA is on-policy (uses the action actually taken).

TD target:
    y = r + γ V(s')                     (TD(0) for state values)
    y = r + γ max_{a'} Q(s', a')        (Q-learning)
    y = r + γ Q(s', a')                 (SARSA)

TD error (δ):
    δ = y - Q(s, a)

Update rule:
    Q(s, a) <- Q(s, a) + α δ
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon-greedy action selection.

    With probability (1 - ε): choose argmax_a Q(state, a)  (exploit)
    With probability ε: choose a random action             (explore)

    Parameters:
        Q: np.ndarray of shape (n_states, n_actions) - Q-value table.
        state: int - Current state index.
        epsilon: float - Exploration probability in [0, 1].

    Returns:
        action: int - Selected action index.
    """
    n_states, n_actions = Q.shape
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state, :])


def td_target(reward, gamma, next_value):
    """
    Compute the TD(0) target.

        y = r + γ V(s')

    Parameters:
        reward: float - Immediate reward r.
        gamma: float - Discount factor.
        next_value: float - Estimated value of the next state V(s').

    Returns:
        target: float - TD target.
    """
    target = reward + gamma * next_value
    return target


def td_error(reward, gamma, current_value, next_value):
    """
    Compute the TD error (temporal difference).

        δ = r + γ V(s') - V(s)

    Parameters:
        reward: float - Immediate reward r.
        gamma: float - Discount factor.
        current_value: float - Value of current state V(s).
        next_value: float - Value of next state V(s').

    Returns:
        delta: float - TD error.
    """
    targ = td_target(reward, gamma, next_value)
    delta = np.abs(current_value - targ)
    return delta


def q_learning_update(Q, state, action, reward, next_state, alpha, gamma):
    """
    One-step Q-learning (off-policy TD) update.

    Q-learning uses the greedy next action for the bootstrap target:
        target = r + γ max_{a'} Q(s', a')
        Q(s, a) <- Q(s, a) + α [target - Q(s, a)]

    Parameters:
        Q: np.ndarray of shape (n_states, n_actions) - Q-value table (modified in-place).
        state: int - Current state s.
        action: int - Action taken a.
        reward: float - Reward received r.
        next_state: int - Next state s'.
        alpha: float - Learning rate.
        gamma: float - Discount factor.

    Returns:
        Q: np.ndarray of shape (n_states, n_actions) - Updated Q-table.
        td_err: float - The TD error for this update.
    """
    next_state_value = np.max(Q[next_state, :])
    current_value = Q[state, action]
    td_err = td_error(reward, gamma,current_value, next_state_value)
    Q[state, action] = Q[state, action] + alpha * td_err
    return Q, td_err


def sarsa_update(Q, state, action, reward, next_state, next_action, alpha, gamma):
    """
    One-step SARSA (on-policy TD) update.

    SARSA uses the actual next action for the bootstrap target:
        target = r + γ Q(s', a')
        Q(s, a) <- Q(s, a) + α [target - Q(s, a)]

    Parameters:
        Q: np.ndarray of shape (n_states, n_actions) - Q-value table (modified in-place).
        state: int - Current state s.
        action: int - Action taken a.
        reward: float - Reward r.
        next_state: int - Next state s'.
        next_action: int - Next action a' (actually chosen by the policy).
        alpha: float - Learning rate.
        gamma: float - Discount factor.

    Returns:
        Q: np.ndarray of shape (n_states, n_actions) - Updated Q-table.
        td_err: float - The TD error for this update.
    """
    next_state_value = Q[next_state, next_action]
    current_value = Q[state, action]
    td_err = td_error(reward, gamma, current_value, next_state_value)
    Q[state, action] = Q[state, action] + alpha * td_err
    return Q, td_err


def n_step_return(rewards, gamma, bootstrap_value=0.0):
    """
    Compute the n-step return from a sequence of rewards.

        G = r_0 + γ r_1 + γ² r_2 + ... + γ^{n-1} r_{n-1} + γ^n V(s_n)

    Parameters:
        rewards: np.ndarray of shape (n,) - Sequence of rewards [r_0, ..., r_{n-1}].
        gamma: float - Discount factor.
        bootstrap_value: float - Bootstrap value V(s_n) for the state after
            the last reward. Use 0 for terminal episodes.

    Returns:
        G: float - n-step discounted return.
    """
    powers = np.arange(len(rewards))
    G = (rewards * np.power(gamma, powers)).sum() + gamma ** len(rewards) * bootstrap_value
    return G
