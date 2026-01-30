import numpy as np
import pytest
from rl.q_learning import (
    epsilon_greedy, q_learning_update, sarsa_update,
    td_target, td_error, n_step_return,
)


class TestEpsilonGreedy:
    """Tests for epsilon-greedy action selection."""

    def test_greedy_action(self):
        """With epsilon=0, should always pick the greedy action."""
        Q = np.array([[1.0, 5.0, 3.0]])  # 1 state, 3 actions
        for _ in range(20):
            action = epsilon_greedy(Q, 0, 0.0)
            assert action == 1  # argmax

    def test_returns_valid_action(self):
        """Action should be a valid index."""
        Q = np.zeros((5, 3))
        for _ in range(50):
            action = epsilon_greedy(Q, 0, 0.5)
            assert 0 <= action < 3

    def test_explores_with_high_epsilon(self):
        """With epsilon=1, should eventually pick all actions."""
        Q = np.array([[100.0, 0.0, 0.0]])  # Strong preference for 0
        np.random.seed(42)
        actions = set()
        for _ in range(200):
            actions.add(epsilon_greedy(Q, 0, 1.0))
        assert len(actions) == 3  # All actions selected at least once

    def test_mostly_greedy_with_small_epsilon(self):
        """With small epsilon, should mostly pick greedy."""
        Q = np.array([[0.0, 10.0]])
        np.random.seed(42)
        actions = [epsilon_greedy(Q, 0, 0.05) for _ in range(1000)]
        greedy_frac = sum(a == 1 for a in actions) / len(actions)
        assert greedy_frac > 0.9


class TestTDTarget:
    """Tests for TD target computation."""

    def test_basic(self):
        target = td_target(1.0, 0.9, 5.0)
        np.testing.assert_almost_equal(target, 1.0 + 0.9 * 5.0)

    def test_terminal(self):
        """At terminal state, next_value=0."""
        target = td_target(10.0, 0.9, 0.0)
        np.testing.assert_almost_equal(target, 10.0)

    def test_zero_discount(self):
        target = td_target(3.0, 0.0, 100.0)
        np.testing.assert_almost_equal(target, 3.0)


class TestTDError:
    """Tests for TD error computation."""

    def test_basic(self):
        delta = td_error(1.0, 0.9, 2.0, 5.0)
        # delta = 1 + 0.9*5 - 2 = 1 + 4.5 - 2 = 3.5
        np.testing.assert_almost_equal(delta, 3.5)

    def test_zero_when_correct(self):
        """TD error is 0 when V(s) = r + γ V(s')."""
        delta = td_error(1.0, 0.5, 3.0, 4.0)
        # delta = 1 + 0.5*4 - 3 = 1 + 2 - 3 = 0
        np.testing.assert_almost_equal(delta, 0.0)

    def test_positive_means_underestimate(self):
        """Positive TD error means current value is too low."""
        delta = td_error(5.0, 0.9, 0.0, 0.0)
        assert delta > 0


class TestQLearningUpdate:
    """Tests for Q-learning update."""

    def test_shape_preserved(self):
        Q = np.zeros((3, 2))
        Q_new, _ = q_learning_update(Q, 0, 0, 1.0, 1, 0.1, 0.9)
        assert Q_new.shape == (3, 2)

    def test_only_updates_sa_pair(self):
        """Only Q(s, a) should change."""
        Q = np.zeros((3, 2))
        Q_new, _ = q_learning_update(Q.copy(), 1, 0, 1.0, 2, 0.5, 0.9)
        # All entries except (1, 0) should be unchanged
        for s in range(3):
            for a in range(2):
                if not (s == 1 and a == 0):
                    np.testing.assert_almost_equal(Q_new[s, a], 0.0)

    def test_uses_max_next_q(self):
        """Q-learning should bootstrap from max Q(s', a')."""
        Q = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 7.0]])
        Q_new, td_err = q_learning_update(Q.copy(), 1, 0, 1.0, 2, 0.1, 0.9)
        # target = 1.0 + 0.9 * max(3, 7) = 1 + 6.3 = 7.3
        # td_err = 7.3 - 0 = 7.3
        # Q_new(1,0) = 0 + 0.1 * 7.3 = 0.73
        np.testing.assert_almost_equal(td_err, 7.3)
        np.testing.assert_almost_equal(Q_new[1, 0], 0.73)

    def test_returns_td_error(self):
        Q = np.zeros((2, 2))
        _, td_err = q_learning_update(Q.copy(), 0, 0, 5.0, 1, 0.1, 0.9)
        # target = 5 + 0 = 5, Q(0,0)=0, err = 5
        np.testing.assert_almost_equal(td_err, 5.0)

    def test_convergence_direction(self):
        """Repeated updates should move Q toward target."""
        Q = np.zeros((2, 2))
        for _ in range(100):
            Q, _ = q_learning_update(Q, 0, 0, 1.0, 1, 0.1, 0.0)
        # With gamma=0, Q(0,0) should converge to R=1.0
        np.testing.assert_almost_equal(Q[0, 0], 1.0, decimal=2)


class TestSarsaUpdate:
    """Tests for SARSA update."""

    def test_shape_preserved(self):
        Q = np.zeros((3, 2))
        Q_new, _ = sarsa_update(Q, 0, 0, 1.0, 1, 1, 0.1, 0.9)
        assert Q_new.shape == (3, 2)

    def test_uses_actual_next_action(self):
        """SARSA should bootstrap from Q(s', a'), not max."""
        Q = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 7.0]])
        # next_action = 0, so uses Q(2, 0) = 3, not max = 7
        Q_new, td_err = sarsa_update(Q.copy(), 1, 0, 1.0, 2, 0, 0.1, 0.9)
        # target = 1 + 0.9 * 3 = 3.7
        np.testing.assert_almost_equal(td_err, 3.7)
        np.testing.assert_almost_equal(Q_new[1, 0], 0.37)

    def test_differs_from_q_learning(self):
        """SARSA and Q-learning should give different results when max != actual."""
        Q = np.array([[0.0, 0.0], [2.0, 8.0]])
        Q_sarsa, _ = sarsa_update(Q.copy(), 0, 0, 1.0, 1, 0, 0.5, 0.9)
        Q_ql, _ = q_learning_update(Q.copy(), 0, 0, 1.0, 1, 0.5, 0.9)
        # SARSA uses Q(1,0)=2, Q-learning uses max(2,8)=8
        assert Q_sarsa[0, 0] != Q_ql[0, 0]


class TestNStepReturn:
    """Tests for n-step discounted return."""

    def test_single_step(self):
        """1-step return = r + γ V(s')."""
        G = n_step_return(np.array([5.0]), 0.9, bootstrap_value=10.0)
        np.testing.assert_almost_equal(G, 5.0 + 0.9 * 10.0)

    def test_two_steps_no_bootstrap(self):
        """2-step return with terminal episode."""
        G = n_step_return(np.array([1.0, 2.0]), 0.9, bootstrap_value=0.0)
        # G = 1 + 0.9*2 + 0.81*0 = 2.8
        np.testing.assert_almost_equal(G, 2.8)

    def test_constant_rewards(self):
        """Constant reward r for n steps: G = r(1-γ^n)/(1-γ) + γ^n V."""
        r = 1.0
        n = 5
        gamma = 0.9
        rewards = np.full(n, r)
        G = n_step_return(rewards, gamma, bootstrap_value=0.0)
        expected = r * (1 - gamma**n) / (1 - gamma)
        np.testing.assert_almost_equal(G, expected)

    def test_zero_discount(self):
        """With gamma=0, only first reward matters."""
        G = n_step_return(np.array([3.0, 100.0, 200.0]), 0.0, bootstrap_value=999.0)
        np.testing.assert_almost_equal(G, 3.0)

    def test_full_episode(self):
        """Full episode with rewards [1, 2, 3], gamma=0.5."""
        G = n_step_return(np.array([1.0, 2.0, 3.0]), 0.5, bootstrap_value=0.0)
        # G = 1 + 0.5*2 + 0.25*3 = 1 + 1 + 0.75 = 2.75
        np.testing.assert_almost_equal(G, 2.75)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
