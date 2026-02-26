"""
Change of Variables Formula - Foundation for Normalizing Flows

This module implements the fundamental change of variables formula that underpins
all normalizing flow models. The key insight is that if we have a simple distribution
p(z) and learn an invertible transformation T: z -> x, we can model complex distributions.

Theory:
-------
For an invertible transformation T: R^d -> R^d with inverse T^{-1}, the probability
density is related by the change of variables formula:

    p_x(x) = p_z(T^{-1}(x)) * |det(∂T^{-1}(x)/∂x)|
           = p_z(z) * |det(∂T/∂z)|^{-1}

where |det(∂T/∂z)| is the absolute value of the determinant of the Jacobian of T.

In log space (for numerical stability):
    log p_x(x) = log p_z(z) - log |det(∂T/∂z)|

Key Properties:
    1. Invertibility: The transformation must be bijective (one-to-one and onto)
    2. Tractable Jacobian: The determinant must be computable efficiently
    3. Composition: Multiple flows can be composed by chaining transformations

Mathematical Details:
    For a transformation z = T^{-1}(x):
    - Jacobian J = ∂T/∂z is a d×d matrix
    - We need log|det(J)| which can be:
      * Direct computation: O(d³) for general matrices
      * Triangular: O(d) for triangular Jacobians
      * Coupling layers: O(d) by design

References:
    [1] Rezende, D. J., & Mohamed, S. (2015). "Variational Inference with
        Normalizing Flows." ICML.
        https://arxiv.org/abs/1505.05424

    [2] Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., &
        Lakshminarayanan, B. (2021). "Normalizing Flows for Probabilistic
        Modeling and Inference." JMLR.
        https://arxiv.org/abs/1912.02762

    [3] Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., &
        Duvenaud, D. (2019). "FFJORD: Free-form Continuous Dynamics for
        Scalable Reversible Learning." NeurIPS.
        https://arxiv.org/abs/1810.01367
"""

import numpy as np
from typing import Tuple, Optional
import math
from python.nn_core import Module, Parameter


class ChangeOfVariablesFlow(Module):
    """
    Base class implementing the change of variables formula.

    This serves as the foundation for all normalizing flow architectures.
    Subclasses should implement:
        - forward(): Apply transformation z -> x
        - inverse(): Apply inverse transformation x -> z
        - log_det_jacobian(): Compute log|det(∂T/∂z)|

    Attributes:
        base_dist: The simple base distribution p(z) (e.g., standard normal)
        transforms: List of invertible transformations
    """

    def __init__(self, base_dist=None):
        """
        Initialize the change of variables flow.

        Args:
            base_dist: A distribution instance (default: Standard normal).
                      Default: Standard normal distribution
        """
        super().__init__()
        self.base_dist = base_dist or None  # Standard normal by default

    def forward(
        self,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform samples from base distribution to data distribution.

        Args:
            z: Samples from base distribution, shape (batch_size, d)

        Returns:
            x: Transformed samples, shape (batch_size, d)
            log_det_j: Log determinant of Jacobian, shape (batch_size,)

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            "Subclass must implement forward() transformation.\n"
            "Example:\n"
            "    x = T(z)  # Apply invertible transformation\n"
            "    log_det = self.log_det_jacobian(z)\n"
            "    return x, log_det"
        )

    def inverse(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform samples from data distribution back to base distribution.

        Args:
            x: Data samples, shape (batch_size, d)

        Returns:
            z: Transformed samples in base distribution, shape (batch_size, d)
            log_det_j_inv: Log determinant of inverse Jacobian, shape (batch_size,)

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            "Subclass must implement inverse() transformation.\n"
            "For numerical stability, return the negative of the forward log_det.\n"
            "Example:\n"
            "    z = T_inv(x)  # Apply inverse transformation\n"
            "    log_det_inv = -self.log_det_jacobian(z)\n"
            "    return z, log_det_inv"
        )

    def log_det_jacobian(
        self,
        z: np.ndarray
    ) -> np.ndarray:
        """
        Compute log|det(∂T/∂z)| - the log absolute determinant of the Jacobian.

        This is crucial for the change of variables formula:
            log p_x(x) = log p_z(z) - log|det(∂T/∂z)|

        Args:
            z: Input to the transformation, shape (batch_size, d)

        Returns:
            log_det: Log absolute determinant of Jacobian, shape (batch_size,)

        Raises:
            NotImplementedError: Must be implemented by subclass

        Notes:
            - For numerical stability, always work in log space
            - Never compute |det(J)| directly, use log|det(J)|
            - For triangular matrices, log|det| = sum(log|diag|)
        """
        raise NotImplementedError(
            "Subclass must implement log_det_jacobian().\n"
            "This computes log|det(∂T/∂z)| for the change of variables formula.\n"
            "Strategy depends on the structure:\n"
            "  1. Triangular Jacobian: Use sum of log(diag) - O(d)\n"
            "  2. Orthogonal transform: log|det| = 0 or constant\n"
            "  3. Coupling layers: Compute only affected dimensions\n"
            "  4. Autoregressive: Use product rule for causal dependencies"
        )

    def log_prob(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Compute log probability under the model using change of variables.

        Implements: log p_x(x) = log p_z(T^{-1}(x)) - log|det(∂T/∂z)|

        Args:
            x: Data samples, shape (batch_size, d)

        Returns:
            log_px: Log probability of x under the model, shape (batch_size,)

        Raises:
            NotImplementedError: Requires inverse() and log_det_jacobian()
        """
        raise NotImplementedError(
            "Implement log_prob() using change of variables:\n"
            "    z, log_det_inv = self.inverse(x)\n"
            "    log_pz = standard_normal_logprob(z).sum(axis=-1)\n"
            "    log_px = log_pz + log_det_inv\n"
            "    return log_px"
        )

    def sample(
        self,
        num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from the learned distribution.

        Args:
            num_samples: Number of samples to generate

        Returns:
            x: Generated samples, shape (num_samples, d)
            log_px: Log probability of samples, shape (num_samples,)

        Raises:
            NotImplementedError: Requires forward() and base distribution sampling
        """
        raise NotImplementedError(
            "Implement sample() to generate from the model:\n"
            "    z = np.random.standard_normal((num_samples, d))\n"
            "    x, log_det = self.forward(z)\n"
            "    log_pz = standard_normal_logprob(z).sum(axis=-1)\n"
            "    log_px = log_pz - log_det\n"
            "    return x, log_px"
        )


class JacobianTheory:
    """
    Mathematical reference for Jacobian computation in flows.

    The Jacobian matrix J = ∂T/∂z contains partial derivatives:
        J[i,j] = ∂T_i/∂z_j

    For multidimensional integrals (change of variables):
        ∫_x f(x) dx = ∫_z f(T(z)) |det(∂T/∂z)| dz

    This gives us the density transformation:
        p_x(x) = p_z(T^{-1}(x)) |det(∂T^{-1}(∂x))|
               = p_z(z) |det(∂T/∂z)|^{-1}

    Computational Strategies:

    1. TRIANGULAR JACOBIAN (Autoregressive Flows, Coupling Layers):
       det(J) = product of diagonal elements
       log|det(J)| = sum(log|diag(J)|)  [O(d) complexity]

       Example:
       ```
       J = [[a,  0,  0],
            [b,  c,  0],
            [d,  e,  f]]
       log|det(J)| = log|a| + log|c| + log|f|
       ```

    2. ORTHOGONAL TRANSFORMATION:
       If T is orthogonal: T^T T = I
       Then: det(J) = ±1, so log|det(J)| = 0

       Example: Householder reflections, QR flows

    3. PERMUTATIONS:
       If T permutes dimensions: det(J) = ±1
       log|det(J)| = 0

    4. COUPLING LAYERS:
       If T = [I, 0; t(x_1), I] then:
       det(J) = 1 (only lower block affects)
       or can compute det of scale function

    5. AFFINE TRANSFORMATIONS:
       T(z) = W z + b where W is invertible
       log|det(J)| = log|det(W)|

       For diagonal W: log|det(W)| = sum(log|diag(W)|)
    """

    @staticmethod
    def compute_log_det_jacobian_autoregressive(log_scale: np.ndarray) -> np.ndarray:
        """
        Compute log det Jacobian for autoregressive transformations.

        Args:
            log_scale: Log of scale factors, shape (batch_size, d)

        Returns:
            log_det: Sum of log scales, shape (batch_size,)
        """
        return log_scale.sum(axis=-1)

    @staticmethod
    def compute_log_det_jacobian_affine(
        log_det_weight: np.ndarray
    ) -> np.ndarray:
        """
        Compute log det Jacobian for affine transformations.

        Args:
            log_det_weight: Log determinant of weight matrix

        Returns:
            log_det: log determinant per sample
        """
        return log_det_weight


if __name__ == "__main__":
    # Example usage pattern
    print("Change of Variables Formula - Foundation for Normalizing Flows")
    print("=" * 70)
    print("\nKey Formula:")
    print("  log p_x(x) = log p_z(T^{-1}(x)) - log|det(∂T/∂z)|")
    print("\nImplementation Checklist:")
    print("  [ ] forward(z) -> (x, log_det_jacobian)")
    print("  [ ] inverse(x) -> (z, -log_det_jacobian)")
    print("  [ ] log_prob(x) using change of variables")
    print("  [ ] sample() from base distribution")
    print("\nRefer to subclasses for specific implementations:")
    print("  - nice.py: NICE (Non-linear ICA)")
    print("  - realnvp.py: RealNVP (affine coupling layers)")
    print("  - glow.py: Glow (1x1 convolutions + coupling)")
    print("  - flow_matching.py: Modern flow matching approach")


def flow_log_prob(z: np.ndarray, log_det_jacobian: np.ndarray,
                  base_log_prob_fn=None) -> np.ndarray:
    """
    Compute log probability under a normalizing flow.

    Uses the change of variables formula:
    log p(x) = log p(z) + log |det(dz/dx)|

    Args:
        z: Samples in base space
        log_det_jacobian: Log absolute determinant of Jacobian
        base_log_prob_fn: Function to compute base distribution log prob
                         (defaults to standard normal)

    Returns:
        Log probabilities for each sample
    """
    raise NotImplementedError(
        "TODO: Implement flow log probability\\n"
        "Hint: log_prob = base_log_prob(z) + log_det_jacobian"
    )

