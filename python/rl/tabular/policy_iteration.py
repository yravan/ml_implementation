"""
Policy Iteration - Dynamic Programming Algorithm for Finding Optimal Policies

Implementation Status: STUB - Ready for implementation
Complexity: O(k * |S|^2 * |A|) per iteration, where k is policy eval iterations
Prerequisites: Policy Evaluation, Bellman optimality equations, MDP theory

Policy Iteration is a fundamental dynamic programming algorithm that solves MDPs by
alternating between two steps: policy evaluation (computing V^π) and policy improvement
(updating π to be greedy with respect to V). The algorithm is guaranteed to find the
optimal policy π* and optimal value function V* in finite time for finite MDPs.

The key insight is that policies generated are monotonically improving (V^π_1 ≤ V^π_2
until convergence to optimality), which guarantees convergence. Policy iteration often
converges faster than value iteration despite requiring full policy evaluation in each step,
because early iterations benefit from the exponential speedup of policy updates.

Mathematical Foundation:

Policy Improvement Theorem:
If V^π(s) = E[R_t + γV^π(S_{t+1}) | S_t=s] and π'(a|s) = 1 if a = argmax_a Q^π(s,a),
then V^π'(s) ≥ V^π(s) for all s, meaning π' is a better or equal policy.

Bellman Optimality Equation:
V*(s) = max_a E[R_{t+1} + γV*(S_{t+1}) | S_t=s, A_t=a]
     = max_a Σ_{s',r} p(s',r|s,a) [r + γV*(s')]

Policy iteration updates policies greedily: π(s) = argmax_a Q(s,a)

References:
- Sutton & Barto (2018), Chapter 4.2-4.3: https://mitpress.mit.edu/9780262039246/reinforcement-learning/
- David Silver's Lecture 3: https://www.davidsilver.uk/teaching/
- Konda & Tsitsiklis (2000): https://ieeexplore.ieee.org/document/868307
"""

from typing import Tuple, Dict, Optional, Callable
import numpy as np
from .policy_evaluation import PolicyEvaluator


class PolicyIterator:
    """
    Policy Iteration Algorithm for finding optimal policies in MDPs.

    Policy Iteration alternates between:
    1. Policy Evaluation: Compute V^π given current policy π
    2. Policy Improvement: Update π to be greedy: π(s) = argmax_a Q^π(s,a)

    This process is guaranteed to converge to the optimal policy π* and value V*.

    Attributes:
        num_states (int): Number of discrete states
        num_actions (int): Number of discrete actions
        gamma (float): Discount factor (0 ≤ γ ≤ 1)
        policy (np.ndarray): Current policy π of shape (num_states, num_actions)
        V (np.ndarray): Current value function of shape (num_states,)
        iteration_history (list): History of value functions across iterations
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma: float = 0.99,
        theta: float = 1e-6,
        policy_init: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize PolicyIterator.

        Args:
            num_states: Number of discrete states
            num_actions: Number of discrete actions
            gamma: Discount factor for future rewards
            theta: Convergence threshold for policy evaluation
            policy_init: Optional initial policy. If None, uses uniform random.
                        Shape should be (num_states, num_actions)

        Raises:
            ValueError: If dimensions are inconsistent or gamma not in [0,1]

        Example:
            >>> pi = PolicyIterator(num_states=16, num_actions=4, gamma=0.99)
            >>> print(pi.policy.shape)
            (16, 4)
        """
        raise NotImplementedError(
            "Implement __init__ to:\n"
            "1. Validate inputs and store num_states, num_actions, gamma, theta\n"
            "2. Initialize policy: uniform random or from policy_init\n"
            "3. Initialize V as zeros of shape (num_states,)\n"
            "4. Create PolicyEvaluator instance\n"
            "5. Initialize iteration_history as empty list"
        )

    def solve(
        self,
        transition_model: Callable,
        reward_model: Callable,
        max_iterations: int = 100,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Solve the MDP using policy iteration.

        Main loop:
        ```
        repeat:
            1. Policy Evaluation: Compute V^π
            2. Policy Improvement: π_new = greedy(Q^π)
            3. Check if π_new == π
        until policy doesn't change or max_iterations reached
        ```

        Args:
            transition_model: Callable P(s'|s,a) or P(s',r|s,a)
                             Signature: (s, a) -> dict of {(s', r): prob}
            reward_model: Callable R(s,a) returning expected reward
                         Signature: (s, a) -> float
            max_iterations: Maximum policy iteration steps
            verbose: Print convergence information

        Returns:
            Tuple of:
            - Optimal policy π* of shape (num_states, num_actions)
            - Optimal value function V* of shape (num_states,)
            - Dictionary with keys:
              - 'iterations': Number of iterations until convergence
              - 'converged': Boolean indicating convergence
              - 'policy_stable': Boolean, True if policy stopped changing
              - 'v_history': List of value functions per iteration
              - 'policy_changes': List of number of policy changes per iteration

        Example:
            >>> def P(s, a):
            ...     return {(s+a, 1): 1.0}
            >>> def R(s, a):
            ...     return 1.0 if s == goal else 0.0
            >>> pi_opt, V_opt, info = iterator.solve(P, R, max_iterations=50)
            >>> print(f"Converged: {info['converged']}")
        """
        raise NotImplementedError(
            "Implement solve to:\n"
            "1. Loop up to max_iterations times:\n"
            "   a. Call PolicyEvaluator.evaluate() to get V^π\n"
            "   b. Compute Q^π values for all state-action pairs\n"
            "   c. Perform policy improvement:\n"
            "      For each state s: π_new(s) = one-hot(argmax_a Q^π(s,a))\n"
            "   d. Track number of policy changes\n"
            "   e. Check if policy is stable (no changes)\n"
            "2. Return π*, V*, and info dict"
        )

    def solve_with_matrix(
        self,
        P: np.ndarray,
        R: np.ndarray,
        max_iterations: int = 100,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Solve MDP using policy iteration with matrix form.

        Uses precomputed transition and reward matrices for efficiency.

        Args:
            P: Transition probability tensor (num_states, num_actions, num_states)
               P[s,a,s'] = p(s'|s,a)
            R: Reward matrix (num_states, num_actions)
               R[s,a] = E[R_t | S_t=s, A_t=a]
            max_iterations: Maximum iterations
            verbose: Print progress

        Returns:
            Tuple of (π*, V*, info_dict)

        Example:
            >>> P = np.random.dirichlet(np.ones(10), size=(5, 3, 10))
            >>> R = np.random.randn(5, 3)
            >>> pi, V, info = iterator.solve_with_matrix(P, R)
        """
        raise NotImplementedError(
            "Implement solve_with_matrix to:\n"
            "1. Use PolicyEvaluator.evaluate_with_matrix for policy eval\n"
            "2. Compute Q values using matrix multiplication:\n"
            "   Q[s,a] = R[s,a] + γ * Σ_s' P[s,a,s'] * V[s']\n"
            "3. Extract greedy policies from Q values\n"
            "4. Return π*, V*, and convergence info"
        )

    def policy_improvement_step(self) -> Tuple[np.ndarray, int]:
        """
        Perform single policy improvement step given current value function.

        Computes Q^π for all state-action pairs and updates policy to be greedy.

        Returns:
            Tuple of:
            - New policy π_new of shape (num_states, num_actions)
            - Number of state-actions where policy changed

        Mathematical Basis:
        π_new(s) = argmax_a Σ_{s'} p(s'|s,a)[R(s,a) + γV(s')]

        Note:
            When multiple actions have equal Q-value, one is chosen arbitrarily.
            For reproducibility, use np.argmax which returns the first maximum.
        """
        raise NotImplementedError(
            "Implement policy_improvement_step to:\n"
            "1. Initialize Q array of shape (num_states, num_actions)\n"
            "2. For each state-action pair, compute:\n"
            "   Q[s,a] = Σ_s' p(s'|s,a) * [r(s,a,s') + γ*V[s']]\n"
            "3. For each state s:\n"
            "   a. Find best action: a* = argmax_a Q[s,a]\n"
            "   b. Set π_new[s, a*] = 1, all others = 0\n"
            "4. Count number of states where policy changed\n"
            "5. Return π_new and change count"
        )

    def is_policy_stable(self) -> bool:
        """
        Check if current policy is stable (no improvement possible).

        A policy is stable when the greedy action under the current value function
        is the same as the current policy action. This indicates optimality.

        Returns:
            Boolean indicating if policy is stable

        Example:
            >>> if iterator.is_policy_stable():
            ...     print("Optimal policy found!")
        """
        raise NotImplementedError(
            "Implement is_policy_stable to:\n"
            "1. For each state s, compute greedy action a* = argmax_a Q(s,a)\n"
            "2. Check if greedy action equals current policy action\n"
            "3. Return True if all states match (policy stable)\n"
            "4. Return False if any state has different action"
        )

    def get_policy(self, stochastic: bool = False) -> np.ndarray:
        """
        Get the current policy.

        Args:
            stochastic: If True, return stochastic policy (probabilities).
                       If False, return deterministic policy (one-hot).

        Returns:
            Policy array of shape (num_states, num_actions)

        Example:
            >>> det_policy = iterator.get_policy(stochastic=False)
            >>> print(det_policy[0])  # Action probabilities for state 0
        """
        raise NotImplementedError(
            "Implement get_policy to return current policy array"
        )

    def get_action_probabilities(self, state: int) -> np.ndarray:
        """
        Get action probabilities for a given state.

        Args:
            state: State index

        Returns:
            Probability distribution over actions for given state

        Example:
            >>> probs = iterator.get_action_probabilities(state=5)
            >>> print(f"Action 0 probability: {probs[0]}")
        """
        raise NotImplementedError(
            "Implement get_action_probabilities to return self.policy[state]"
        )

    def get_greedy_action(self, state: int) -> int:
        """
        Get the greedy action (highest probability action) for a state.

        Args:
            state: State index

        Returns:
            Action index with highest probability

        Example:
            >>> action = iterator.get_greedy_action(state=0)
        """
        raise NotImplementedError(
            "Implement get_greedy_action to return argmax of policy for state"
        )

    def get_value_function(self) -> np.ndarray:
        """
        Get current value function estimate.

        Returns:
            Value function V of shape (num_states,)
        """
        raise NotImplementedError(
            "Implement get_value_function to return copy of self.V"
        )

    def reset(self, policy_init: Optional[np.ndarray] = None) -> None:
        """
        Reset the policy iterator.

        Args:
            policy_init: Optional initial policy. If None, uses uniform random.
        """
        raise NotImplementedError(
            "Implement reset to reinitialize policy and V"
        )

    def get_convergence_stats(self) -> Dict[str, any]:
        """
        Get statistics about convergence.

        Returns:
            Dictionary with:
            - 'iterations': Number of iterations performed
            - 'policy_stable': Whether final policy is stable
            - 'v_history': List of value functions per iteration
            - 'policy_change_history': List of policy changes per iteration
            - 'final_value_range': (min_V, max_V) for final V

        Example:
            >>> stats = iterator.get_convergence_stats()
            >>> print(f"Converged in {stats['iterations']} iterations")
        """
        raise NotImplementedError(
            "Implement get_convergence_stats to return dict with convergence info"
        )


def compute_q_values(
    V: np.ndarray,
    transition_model: Callable,
    reward_model: Callable,
    gamma: float,
) -> np.ndarray:
    """
    Compute Q-values for all state-action pairs given a value function.

    Q^π(s,a) = E[R_t | S_t=s, A_t=a] + γ E[V(S_{t+1}) | S_t=s, A_t=a]
             = Σ_{s'} p(s'|s,a) [R(s,a,s') + γ V(s')]

    Args:
        V: Current value function of shape (num_states,)
        transition_model: Callable for state transitions
        reward_model: Callable for rewards
        gamma: Discount factor

    Returns:
        Q-values array of shape (num_states, num_actions)

    Example:
        >>> Q = compute_q_values(V, P, R, gamma=0.99)
    """
    raise NotImplementedError(
        "Implement compute_q_values to:\n"
        "1. Initialize Q array of shape (num_states, num_actions)\n"
        "2. For each state s and action a:\n"
        "   Q[s,a] = Σ_s' P(s'|s,a) * [R(s,a,s') + γ*V[s']]\n"
        "3. Return Q array"
    )


def greedy_policy_from_q(Q: np.ndarray) -> np.ndarray:
    """
    Extract deterministic greedy policy from Q-values.

    For each state, selects the action with maximum Q-value.

    Args:
        Q: Q-values array of shape (num_states, num_actions)

    Returns:
        Deterministic policy of shape (num_states, num_actions)
        where each row is one-hot encoded (only one 1 per row)

    Example:
        >>> policy = greedy_policy_from_q(Q)
        >>> assert policy[0].sum() == 1  # One-hot encoded
    """
    raise NotImplementedError(
        "Implement greedy_policy_from_q to:\n"
        "1. Find argmax action for each state: a* = argmax_a Q[s,a]\n"
        "2. Create one-hot encoded policy\n"
        "3. Return policy of shape (num_states, num_actions)"
    )


def count_policy_changes(
    policy_old: np.ndarray, policy_new: np.ndarray
) -> int:
    """
    Count number of states where policy changed.

    Compares deterministic policies (assumed one-hot encoded) and counts
    states where the greedy action changed.

    Args:
        policy_old: Old policy of shape (num_states, num_actions)
        policy_new: New policy of shape (num_states, num_actions)

    Returns:
        Number of states where the greedy action changed

    Example:
        >>> changes = count_policy_changes(pi_old, pi_new)
        >>> print(f"Policy changed in {changes} states")
    """
    raise NotImplementedError(
        "Implement count_policy_changes to:\n"
        "1. For each state, find argmax action in old and new policy\n"
        "2. Count states where actions differ\n"
        "3. Return count"
    )
