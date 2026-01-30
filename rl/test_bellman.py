import numpy as np
import pytest
from rl.bellman import (
    bellman_backup_v, bellman_backup_q,
    policy_evaluation, value_iteration, policy_iteration,
    extract_greedy_policy,
)


def make_simple_mdp():
    """
    Simple 3-state, 2-action MDP for testing.

    States: 0, 1, 2
    Actions: 0 (left), 1 (right)

    State 0 is "start", state 2 is "goal" (absorbing with reward 1).
    """
    n_states, n_actions = 3, 2
    # P[s', s, a]: transition probabilities
    P = np.zeros((n_states, n_states, n_actions))
    # From state 0: action 0 stays, action 1 goes to state 1
    P[0, 0, 0] = 1.0
    P[1, 0, 1] = 1.0
    # From state 1: action 0 goes to state 0, action 1 goes to state 2
    P[0, 1, 0] = 1.0
    P[2, 1, 1] = 1.0
    # State 2 absorbing: all actions stay in state 2
    P[2, 2, 0] = 1.0
    P[2, 2, 1] = 1.0

    # R[s, a]: rewards
    R = np.zeros((n_states, n_actions))
    R[1, 1] = 1.0  # Reward for reaching state 2
    R[2, :] = 0.0  # No more reward in absorbing state

    return P, R


def make_uniform_policy(n_states, n_actions):
    """Uniform random policy."""
    return np.ones((n_states, n_actions)) / n_actions


class TestBellmanBackupV:
    """Tests for Bellman expectation backup on V."""

    def test_shape(self):
        P, R = make_simple_mdp()
        V = np.zeros(3)
        policy = make_uniform_policy(3, 2)
        V_new = bellman_backup_v(V, policy, P, R, 0.9)
        assert V_new.shape == (3,)

    def test_zero_discount(self):
        """With gamma=0, V(s) = Σ_a π(a|s) R(s,a)."""
        P, R = make_simple_mdp()
        V = np.zeros(3)
        policy = make_uniform_policy(3, 2)
        V_new = bellman_backup_v(V, policy, P, R, 0.0)
        expected = np.sum(policy * R, axis=1)
        np.testing.assert_array_almost_equal(V_new, expected)

    def test_absorbing_state(self):
        """Absorbing state with 0 reward should have value 0."""
        P, R = make_simple_mdp()
        V = np.zeros(3)
        policy = make_uniform_policy(3, 2)
        V_new = bellman_backup_v(V, policy, P, R, 0.9)
        np.testing.assert_almost_equal(V_new[2], 0.0)

    def test_one_step_lookahead(self):
        """Check V(s=1) with known V and right-going policy."""
        P, R = make_simple_mdp()
        V = np.array([0.0, 0.0, 0.0])
        # Policy: always go right
        policy = np.array([[0, 1], [0, 1], [0.5, 0.5]])
        V_new = bellman_backup_v(V, policy, P, R, 0.9)
        # V_new(1) = π(right|1) * [R(1,right) + γ * V(2)] = 1 * [1 + 0] = 1.0
        np.testing.assert_almost_equal(V_new[1], 1.0)


class TestBellmanBackupQ:
    """Tests for Bellman optimality backup on Q."""

    def test_shape(self):
        P, R = make_simple_mdp()
        Q = np.zeros((3, 2))
        Q_new = bellman_backup_q(Q, P, R, 0.9)
        assert Q_new.shape == (3, 2)

    def test_zero_discount(self):
        """With gamma=0, Q(s,a) = R(s,a)."""
        P, R = make_simple_mdp()
        Q = np.zeros((3, 2))
        Q_new = bellman_backup_q(Q, P, R, 0.0)
        np.testing.assert_array_almost_equal(Q_new, R)

    def test_one_step(self):
        """Q(1, right) should include reward for reaching goal."""
        P, R = make_simple_mdp()
        Q = np.zeros((3, 2))
        Q_new = bellman_backup_q(Q, P, R, 0.9)
        # Q(1, right) = R(1,1) + γ max_a Q(2,a) = 1 + 0.9*0 = 1
        np.testing.assert_almost_equal(Q_new[1, 1], 1.0)


class TestPolicyEvaluation:
    """Tests for iterative policy evaluation."""

    def test_converges(self):
        P, R = make_simple_mdp()
        policy = make_uniform_policy(3, 2)
        V, n_iter = policy_evaluation(policy, P, R, 0.9)
        assert V.shape == (3,)
        assert n_iter > 0

    def test_optimal_policy_values(self):
        """Evaluating the optimal policy should give V*."""
        P, R = make_simple_mdp()
        # Optimal: go right everywhere
        policy = np.array([[0, 1], [0, 1], [0.5, 0.5]])
        V, _ = policy_evaluation(policy, P, R, 0.9)
        # V(1) = 1 + 0.9*0 = 1.0 (one step from goal)
        np.testing.assert_almost_equal(V[1], 1.0, decimal=5)
        # V(0) = 0 + 0.9*V(1) = 0.9 (two steps)
        np.testing.assert_almost_equal(V[0], 0.9, decimal=5)
        # V(2) = 0 (absorbing, no reward)
        np.testing.assert_almost_equal(V[2], 0.0, decimal=5)

    def test_fixed_point(self):
        """Result should be a fixed point of the Bellman operator."""
        P, R = make_simple_mdp()
        policy = make_uniform_policy(3, 2)
        V, _ = policy_evaluation(policy, P, R, 0.9)
        V_backup = bellman_backup_v(V, policy, P, R, 0.9)
        np.testing.assert_array_almost_equal(V, V_backup, decimal=5)


class TestExtractGreedyPolicy:
    """Tests for greedy policy extraction."""

    def test_deterministic(self):
        """Greedy policy should be deterministic (one-hot rows)."""
        P, R = make_simple_mdp()
        V = np.array([0.5, 1.0, 0.0])
        policy = extract_greedy_policy(V, P, R, 0.9)
        for s in range(3):
            np.testing.assert_almost_equal(policy[s].sum(), 1.0)
            assert np.max(policy[s]) == 1.0  # One-hot

    def test_chooses_best_action(self):
        """Should choose the action that maximizes Q."""
        P, R = make_simple_mdp()
        V = np.array([0.0, 0.0, 0.0])
        policy = extract_greedy_policy(V, P, R, 0.9)
        # At state 1: R(1,right)=1 vs R(1,left)=0 -> should pick right (action 1)
        assert np.argmax(policy[1]) == 1


class TestValueIteration:
    """Tests for value iteration."""

    def test_converges(self):
        P, R = make_simple_mdp()
        V, policy, n_iter = value_iteration(P, R, 0.9)
        assert V.shape == (3,)
        assert policy.shape == (3, 2)
        assert n_iter > 0

    def test_optimal_values(self):
        """Should find V*."""
        P, R = make_simple_mdp()
        V, _, _ = value_iteration(P, R, 0.9)
        # Optimal: go right from 0->1->2
        # V*(2) = 0, V*(1) = 1, V*(0) = 0.9
        np.testing.assert_almost_equal(V[2], 0.0, decimal=4)
        np.testing.assert_almost_equal(V[1], 1.0, decimal=4)
        np.testing.assert_almost_equal(V[0], 0.9, decimal=4)

    def test_optimal_policy(self):
        """Should find π*: go right from states 0 and 1."""
        P, R = make_simple_mdp()
        _, policy, _ = value_iteration(P, R, 0.9)
        assert np.argmax(policy[0]) == 1  # Go right
        assert np.argmax(policy[1]) == 1  # Go right


class TestPolicyIteration:
    """Tests for policy iteration."""

    def test_converges(self):
        P, R = make_simple_mdp()
        V, policy, n_iter = policy_iteration(P, R, 0.9)
        assert V.shape == (3,)
        assert policy.shape == (3, 2)
        assert n_iter >= 1

    def test_matches_value_iteration(self):
        """Policy iteration and value iteration should find the same V*."""
        P, R = make_simple_mdp()
        V_vi, _, _ = value_iteration(P, R, 0.9)
        V_pi, _, _ = policy_iteration(P, R, 0.9)
        np.testing.assert_array_almost_equal(V_vi, V_pi, decimal=4)

    def test_optimal_policy(self):
        """Should find the same optimal policy as value iteration."""
        P, R = make_simple_mdp()
        _, policy, _ = policy_iteration(P, R, 0.9)
        assert np.argmax(policy[0]) == 1
        assert np.argmax(policy[1]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
