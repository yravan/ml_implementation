"""
Reproducibility and Seeding Utilities
=====================================

Tools for ensuring reproducible experiments through proper random state management.

Theory
------
Reproducibility is crucial in ML research. Without it, you can't:
- Debug effectively (can't reproduce the bug)
- Compare experiments fairly (different randomness each time)
- Publish verifiable results (others can't reproduce)

Sources of randomness in ML:
1. Weight initialization
2. Data shuffling / batch sampling
3. Dropout and other stochastic regularization
4. Data augmentation
5. Some optimizers (e.g., with random perturbations)

To achieve full reproducibility:
1. Set seeds for all random number generators (NumPy, Python's random, etc.)
2. Use deterministic algorithms where possible
3. Control batch ordering
4. Be aware that some operations are non-deterministic (e.g., some GPU operations)

Math
----
# Pseudorandom number generators (PRNGs) produce deterministic sequences
# given the same seed. NumPy uses Mersenne Twister by default.
#
# For a PRNG with state s:
#   x_0, s_1 = f(s_0)  # Generate first number, update state
#   x_1, s_2 = f(s_1)  # Generate second number, update state
#   ...
#
# Same initial state (seed) -> same sequence of numbers

References
----------
- PyTorch Reproducibility Guide
  https://pytorch.org/docs/stable/notes/randomness.html
- NumPy Random Generator Documentation
  https://numpy.org/doc/stable/reference/random/generator.html
- "Reproducing Deep Learning" (excellent blog post)
  https://mmcdermott.github.io/reproducible_dl/
- Uber's ML Reproducibility Paper
  https://arxiv.org/abs/1909.06674

Implementation Notes
--------------------
- Always log the seed used for experiments
- Use separate generators for data vs model operations when needed
- Be careful with multi-processing: each worker needs different seed
- Some hardware (GPU) operations may be non-deterministic
"""

# Implementation Status: NOT STARTED
# Complexity: Easy
# Prerequisites: None (foundational module)

import numpy as np
import random
from typing import Optional, Any
import hashlib


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - NumPy's random number generator
    - Python's built-in random module
    - Attempts to set for other common libraries if available

    Args:
        seed: Integer seed value

    Example:
        >>> set_seed(42)
        >>> np.random.rand(3)
        array([0.37454012, 0.95071431, 0.73199394])
        >>> set_seed(42)  # Reset
        >>> np.random.rand(3)  # Same sequence
        array([0.37454012, 0.95071431, 0.73199394])

    Usage Pattern:
        # At start of experiment
        set_seed(config.seed)
        model = create_model()
        optimizer = create_optimizer(model)
        for epoch in range(num_epochs):
            train(model, data, optimizer)
    """
    np.random.seed(seed)
    random.seed(seed)


def get_random_generator(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a new NumPy random generator with optional seed.

    Using explicit generators instead of global state is more robust
    and allows different parts of code to have independent randomness.

    Args:
        seed: Optional seed for reproducibility

    Returns:
        NumPy Generator instance

    Example:
        >>> rng = get_random_generator(42)
        >>> rng.random(3)
        array([0.77395605, 0.43887844, 0.85859792])
        >>> rng2 = get_random_generator(42)  # Same seed
        >>> rng2.random(3)  # Same sequence
        array([0.77395605, 0.43887844, 0.85859792])

    Advantages over np.random.seed():
        - Thread-safe (each thread can have its own generator)
        - Clearer code (explicit where randomness comes from)
        - Better for parallelism
    """
    return np.random.default_rng(seed)


def seed_from_string(s: str) -> int:
    """
    Generate a deterministic seed from a string.

    Useful for creating reproducible seeds from experiment names,
    configuration strings, or timestamps.

    Args:
        s: Any string

    Returns:
        Integer seed derived from string hash

    Example:
        >>> seed_from_string("experiment_v1")
        1234567890  # Some deterministic integer
        >>> seed_from_string("experiment_v1")  # Same string -> same seed
        1234567890
    """
    hash_bytes = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(hash_bytes[:4], 'big')


def get_worker_seed(base_seed: int, worker_id: int) -> int:
    """
    Generate unique seed for a data loading worker.

    When using multiple workers for data loading, each worker needs
    a different seed to avoid loading duplicate batches.

    Args:
        base_seed: Base experiment seed
        worker_id: Worker index (0, 1, 2, ...)

    Returns:
        Unique seed for this worker

    Example:
        >>> base_seed = 42
        >>> [get_worker_seed(base_seed, i) for i in range(4)]
        [42, 43, 44, 45]  # Or some deterministic unique values

    Usage:
        # In data loader worker initialization
        def worker_init_fn(worker_id):
            seed = get_worker_seed(base_seed, worker_id)
            set_seed(seed)
    """
    return base_seed + worker_id

class RandomState:
    """
    Context manager for temporary random state.

    Allows running code with a specific seed without affecting
    the global random state.

    Example:
        >>> np.random.seed(0)
        >>> a = np.random.rand()
        >>> with RandomState(42):
        ...     b = np.random.rand()  # Uses seed 42
        >>> c = np.random.rand()  # Continues from original state
        >>> # a, c are part of seed-0 sequence; b is from seed-42

    Use Case:
        - Testing with deterministic randomness
        - Running evaluation with fixed seed during training
    """

    def __init__(self, seed: int):
        """
        Args:
            seed: Seed to use within the context
        """
        self.seed = seed

    def __enter__(self) -> "RandomState":
        """Save current state and set new seed."""
        self._np_state = np.random.get_state()
        self._py_state = random.get_state()
        set_seed(self.seed)
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore original state."""
        np.random.set_state(self._np_state)
        random.setstate(self._py_state)


def generate_seed_sequence(base_seed: int, n: int) -> np.ndarray:
    """
    Generate a sequence of n independent seeds from a base seed.

    Useful for:
    - Multiple experiment runs
    - Cross-validation folds
    - Ensemble members

    Args:
        base_seed: Starting seed
        n: Number of seeds to generate

    Returns:
        Array of n integer seeds

    Example:
        >>> seeds = generate_seed_sequence(42, 5)
        >>> len(seeds)
        5
        >>> len(set(seeds)) == 5  # All unique
        True
    """
    rng = np.random.default_rng(seed=base_seed)
    return rng.integers(0, 2 ** 31, size=n)


def check_reproducibility(func: callable, seed: int, n_checks: int = 3) -> bool:
    """
    Verify that a function produces reproducible results with a given seed.

    Useful for testing that your seeding is working correctly.

    Args:
        func: Function to test (should take no arguments and return array-like)
        seed: Seed to test
        n_checks: Number of times to run and compare

    Returns:
        True if all runs produce identical results

    Example:
        >>> def my_init():
        ...     return np.random.randn(10)
        >>> check_reproducibility(my_init, seed=42)
        True  # Same values each time with seed=42
    """
    results = []
    for _ in range(n_checks):
        set_seed(seed)
        results.append(func())
    return all(np.allclose(results[0], r) for r in results[1:])



def get_rng(seed: int = None) -> np.random.Generator:
    """
    Get a random number generator.

    Args:
        seed: Optional seed for reproducibility

    Returns:
        NumPy random Generator
    """
    raise NotImplementedError(
        "TODO: Return np.random.default_rng(seed)"
    )
