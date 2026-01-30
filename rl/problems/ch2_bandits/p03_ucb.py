"""
Problem 3: Upper Confidence Bound (UCB)
=======================================

Implement UCB1 algorithm for multi-armed bandits.

Algorithm from Lecture Notes:
    At time t, pull arm a_t = argmax_a [μ̂_a + sqrt(2 * log(t) / N_a(t))]

where:
    - μ̂_a: empirical mean reward of arm a
    - N_a(t): number of times arm a has been pulled
    - The second term is the confidence bonus

Expected behavior:
    - O(sqrt(KT log T)) regret bound
    - Balances exploration and exploitation automatically
    - Should outperform epsilon-greedy in most cases

References:
    - Lecture notes Section 2.3
    - Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem"
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../..')  # Add framework to path

from rl_framework.environments import MultiArmedBandit


class UCB:
    """Upper Confidence Bound algorithm for multi-armed bandits."""

    def __init__(self, n_arms: int, c: float = 2.0):
        """
        Args:
            n_arms: Number of arms
            c: Exploration constant (default sqrt(2) ≈ 1.41, using 2 for more exploration)
        """
        self.n_arms = n_arms
        self.c = c

        # TODO: Initialize tracking variables
        # - counts[a]: number of times arm a was pulled
        # - values[a]: empirical mean reward of arm a
        # - total_rewards[a]: sum of rewards from arm a (for computing mean)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_rewards = np.zeros(n_arms)
        self.t = 0

    def select_arm(self) -> int:
        """
        Select an arm using UCB1 formula.

        Returns:
            Index of the selected arm

        TODO: Implement UCB arm selection
            1. If any arm hasn't been pulled, pull it (initialization)
            2. Otherwise, compute UCB values and select argmax

        UCB formula: μ̂_a + c * sqrt(log(t) / N_a)
        """
        self.t += 1

        # Phase 1: Pull each arm once
        # TODO: Check if any arm has count 0, return that arm

        # Phase 2: UCB selection
        # TODO: Compute UCB values for all arms
        # TODO: Return arm with highest UCB value

        raise NotImplementedError("Implement UCB arm selection")

    def update(self, arm: int, reward: float):
        """
        Update statistics after observing a reward.

        Args:
            arm: The arm that was pulled
            reward: The observed reward

        TODO: Update counts, total_rewards, and values for the pulled arm
        """
        raise NotImplementedError("Implement update method")

    def get_ucb_values(self) -> np.ndarray:
        """
        Compute current UCB values for all arms.

        Returns:
            Array of UCB values

        Useful for visualization and debugging.
        """
        ucb_values = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            if self.counts[a] == 0:
                ucb_values[a] = float('inf')
            else:
                # TODO: Compute UCB value
                pass

        return ucb_values


def run_experiment(
    n_arms: int = 10,
    n_rounds: int = 10000,
    n_trials: int = 20,
    c: float = 2.0
):
    """
    Run UCB experiment and plot results.

    Args:
        n_arms: Number of arms
        n_rounds: Number of rounds per trial
        n_rounds: Number of independent trials
        c: UCB exploration constant
    """
    all_regrets = []

    for trial in range(n_trials):
        # Create bandit instance
        bandit = MultiArmedBandit(n_arms=n_arms, seed=trial)
        ucb = UCB(n_arms=n_arms, c=c)

        regrets = []
        for t in range(n_rounds):
            # Select arm
            arm = ucb.select_arm()

            # Pull arm and observe reward
            reward = bandit.pull(arm)

            # Update UCB
            ucb.update(arm, reward)

            # Track regret
            regrets.append(bandit.get_regret())

        all_regrets.append(regrets)

    # Plot results
    all_regrets = np.array(all_regrets)
    mean_regret = all_regrets.mean(axis=0)
    std_regret = all_regrets.std(axis=0)

    plt.figure(figsize=(10, 6))

    # Plot empirical regret
    plt.plot(mean_regret, label='UCB Regret')
    plt.fill_between(
        range(n_rounds),
        mean_regret - std_regret,
        mean_regret + std_regret,
        alpha=0.2
    )

    # Plot theoretical bound: O(sqrt(KT log T))
    T = np.arange(1, n_rounds + 1)
    theoretical = np.sqrt(n_arms * T * np.log(T + 1))  # Simplified bound
    plt.plot(T, theoretical, '--', label='Theoretical O(√(KT log T))')

    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title(f'UCB1 Algorithm (K={n_arms} arms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ucb_regret.png', dpi=150)
    plt.show()

    print(f"Final regret: {mean_regret[-1]:.2f} ± {std_regret[-1]:.2f}")


def compare_with_epsilon_greedy(
    n_arms: int = 10,
    n_rounds: int = 10000,
    n_trials: int = 20
):
    """
    Compare UCB with epsilon-greedy algorithm.

    TODO: Implement epsilon-greedy and compare
    """
    pass


if __name__ == "__main__":
    print("=" * 50)
    print("Problem 3: UCB Algorithm")
    print("=" * 50)
    print("\nTODO: Implement the UCB class methods:")
    print("  1. select_arm(): UCB arm selection")
    print("  2. update(): Update statistics after reward")
    print("\nRun this file after implementation to test.")
    print("=" * 50)

    # Uncomment after implementation:
    # run_experiment()
