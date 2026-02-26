"""
NICE: Non-linear Independent Components Estimation

This module implements the NICE model, which introduced coupling layers as a way
to construct normalizing flows with tractable Jacobian determinants.

Theory:
-------
NICE addresses the challenge of computing the log-determinant of the Jacobian by
using coupling layers that partition the dimensions into two groups:
    - One group passes through unchanged (identity)
    - Other group undergoes an invertible transformation

This creates a triangular Jacobian structure with determinant = 1!

The key insight: Instead of transforming all dimensions, we partition:
    x1 = z1  (unchanged)
    x2 = z2 + m_θ(z1)  (translated by neural network)

where m_θ is a learned function. The Jacobian is:
    J = [I_d1    0   ]
        [∂m_θ   I_d2 ]

This is lower triangular, so det(J) = det(I_d1) * det(I_d2) = 1!

By alternating which dimensions are masked, we can build deep expressive models.

Architecture:
    1. Partition dimensions: z = [z1, z2]
    2. Pass z1 unchanged: x1 = z1
    3. Transform z2: x2 = z2 + m_θ(z1)
    4. Alternate masking in next layer: swap roles
    5. Stack multiple coupling layers

Key Properties:
    - Log determinant is always 0 (det(J) = 1)
    - Very efficient: only one network per layer
    - Invertible by design
    - Expressive when stacked deep

Limitations:
    - Volume preserving (det = 1) can be limiting for some densities
    - Weaker expressiveness per layer compared to RealNVP
    - Information leakage between partitions can be slow

References:
    [1] Dinh, L., Krueger, D., & Bengio, Y. (2014).
        "NICE: Non-linear Independent Components Estimation."
        arXiv preprint arXiv:1410.8516.
        https://arxiv.org/abs/1410.8516

    [2] Papamakarios, G., et al. (2021).
        "Normalizing Flows for Probabilistic Modeling and Inference."
        Journal of Machine Learning Research.
        https://arxiv.org/abs/1912.02762

Mathematical Background:
    Coupling Layer (Additive):
        Forward:  x1 = z1,  x2 = z2 + m_θ(z1)
        Inverse:  z1 = x1,  z2 = x2 - m_θ(x1)
        log|det(J)| = 0

    Stacked Coupling Layers:
        T(z) = T_k ∘ T_{k-1} ∘ ... ∘ T_1 (z)
        log|det(∂T/∂z)| = sum_i log|det(∂T_i/∂z_i)| = 0
"""

import numpy as np
from typing import Tuple, Optional, List
import math
from python.nn_core import Module, Parameter


class AdditiveCouplingLayer(Module):
    """
    Additive coupling layer for NICE.

    Partitions input into two parts [z1, z2]:
        x1 = z1
        x2 = z2 + m_θ(z1)

    where m_θ is a neural network.

    The Jacobian is lower triangular with diagonal 1s, so det(J) = 1.

    Attributes:
        mask_type: 'even' or 'odd' - determines which indices are masked
        coupling_net: Neural network implementing m_θ
        dim: Dimensionality of input
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        mask_type: str = 'even'
    ):
        """
        Initialize additive coupling layer.

        Args:
            dim: Input dimensionality
            hidden_dim: Hidden dimension of coupling network
            num_layers: Number of hidden layers in coupling network
            mask_type: 'even' (indices 0,2,4,...) or 'odd' (indices 1,3,5,...)
                      Determines which dimensions are transformed

        Raises:
            ValueError: If mask_type not in ['even', 'odd']
        """
        super().__init__()
        if mask_type not in ['even', 'odd']:
            raise ValueError(f"mask_type must be 'even' or 'odd', got {mask_type}")

        self.dim = dim
        self.mask_type = mask_type

        # Create binary mask
        mask = np.zeros(dim, dtype=bool)
        if mask_type == 'even':
            mask[::2] = True  # Mask even indices
        else:
            mask[1::2] = True  # Mask odd indices

        self.register_buffer('mask', mask)

        # Coupling network: maps masked part to transformation for unmasked part
        masked_dim = int(mask.sum())
        unmasked_dim = dim - masked_dim

        raise NotImplementedError(
            "AdditiveCouplingLayer.forward() is not implemented.\n\n"
            "TODO: Implement the forward pass for additive coupling:\n"
            "  1. Split input using self.mask: z_masked, z_unmasked\n"
            "  2. Compute coupling function: shift = self.coupling_net(z_masked)\n"
            "  3. Add shift to unmasked part: x_unmasked = z_unmasked + shift\n"
            "  4. Recombine: x = [x_masked, x_unmasked] with proper ordering\n"
            "  5. Return (x, np.zeros(z.shape[0]))\n"
            "     (log_det = 0 because Jacobian has det = 1)\n\n"
            "Mathematical Details:\n"
            "  - Masked dimensions pass through unchanged: x_masked = z_masked\n"
            "  - Unmasked dimensions get shifted: x_unmasked = z_unmasked + m_θ(z_masked)\n"
            "  - Jacobian J = [I,  0; ∂m_θ, I] is lower triangular\n"
            "  - det(J) = 1, so log|det(J)| = 0\n\n"
            "Hint: Use np.where(mask, a, b) to select between masked/unmasked values"
        )

    def forward(
        self,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply additive coupling transformation.

        Args:
            z: Input, shape (batch_size, dim)

        Returns:
            x: Transformed output, shape (batch_size, dim)
            log_det_jacobian: Log determinant (always 0), shape (batch_size,)

        Raises:
            NotImplementedError: Needs implementation
        """
        raise NotImplementedError(
            "Implement forward() coupling transformation.\n"
            "See class docstring for mathematical details."
        )

    def inverse(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply inverse additive coupling transformation.

        Args:
            x: Input, shape (batch_size, dim)

        Returns:
            z: Transformed output, shape (batch_size, dim)
            log_det_jacobian: Log determinant (always 0), shape (batch_size,)

        Raises:
            NotImplementedError: Needs implementation
        """
        raise NotImplementedError(
            "Implement inverse() - subtract instead of add:\n"
            "  z_unmasked = x_unmasked - m_θ(x_masked)\n"
            "  Return (z, zeros) since Jacobian determinant is 1"
        )


class NICE(Module):
    """
    NICE: Non-linear Independent Components Estimation

    Builds a normalizing flow by stacking multiple coupling layers with
    alternating masks.

    Architecture:
        Input z -> Coupling Layer 1 -> Coupling Layer 2 -> ... -> x Output

    Key Feature: Volume Preserving (det = 1)
        The composition of multiple coupling layers maintains det = 1:
        log|det(∂T/∂z)| = sum_i log|det(∂T_i/∂z_i)| = 0

    This means the learned distribution has the same volume as the base
    distribution, which can be a limitation but also provides stability.

    Attributes:
        dim: Dimensionality
        num_layers: Number of coupling layers
        hidden_dim: Hidden dimension of coupling networks
        layers: List of coupling layers
        base_dist: Base distribution p(z)
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        hidden_dim: int = 128,
        coupling_layers: int = 3,
        base_dist=None
    ):
        """
        Initialize NICE model.

        Args:
            dim: Input dimensionality
            num_layers: Number of coupling layers to stack
            hidden_dim: Hidden dimension in coupling networks
            coupling_layers: Number of hidden layers per coupling network
            base_dist: Base distribution (default: standard normal)

        Raises:
            NotImplementedError: Requires coupling layer implementation
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Initialize base distribution
        if base_dist is None:
            base_dist = None  # Will use standard normal in sampling/likelihood
        self.base_dist = base_dist

        raise NotImplementedError(
            "NICE.__init__() needs to build the coupling layers.\n\n"
            "TODO:\n"
            "  1. Create list of AdditiveCouplingLayer instances\n"
            "  2. Alternate mask types: 'even', 'odd', 'even', 'odd', ...\n"
            "  3. Store in self.layers as a list or similar container\n\n"
            "Code skeleton:\n"
            "    self.layers = []\n"
            "    for i in range(num_layers):\n"
            "        mask_type = 'even' if i % 2 == 0 else 'odd'\n"
            "        layer = AdditiveCouplingLayer(\n"
            "            dim=dim,\n"
            "            hidden_dim=hidden_dim,\n"
            "            num_layers=coupling_layers,\n"
            "            mask_type=mask_type\n"
            "        )\n"
            "        self.layers.append(layer)"
        )

    def forward(
        self,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform base distribution samples to data distribution.

        Args:
            z: Samples from base distribution, shape (batch_size, dim)

        Returns:
            x: Transformed samples, shape (batch_size, dim)
            log_det_j: Log determinant of Jacobian (always 0), shape (batch_size,)

        Raises:
            NotImplementedError: Needs layer implementation
        """
        raise NotImplementedError(
            "Implement forward by composing coupling layers:\n"
            "    x = z\n"
            "    for layer in self.layers:\n"
            "        x, _ = layer(x)  # log_det is always 0\n"
            "    log_det_j = np.zeros(z.shape[0])\n"
            "    return x, log_det_j"
        )

    def inverse(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data back to base distribution.

        Args:
            x: Data samples, shape (batch_size, dim)

        Returns:
            z: Base distribution samples, shape (batch_size, dim)
            log_det_j_inv: Log determinant of inverse (always 0), shape (batch_size,)

        Raises:
            NotImplementedError: Needs layer implementation
        """
        raise NotImplementedError(
            "Implement inverse by applying layers in reverse order:\n"
            "    z = x\n"
            "    for layer in reversed(self.layers):\n"
            "        z, _ = layer.inverse(z)\n"
            "    log_det_j_inv = np.zeros(x.shape[0])\n"
            "    return z, log_det_j_inv"
        )

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log probability using change of variables.

        log p_x(x) = log p_z(z) - log|det(∂T/∂z)|
                   = log p_z(z) - 0  (since det = 1)
                   = log p_z(z)

        Args:
            x: Data samples, shape (batch_size, dim)

        Returns:
            log_px: Log probability, shape (batch_size,)

        Raises:
            NotImplementedError: Needs inverse implementation
        """
        raise NotImplementedError(
            "Implement log_prob using change of variables:\n"
            "    z, log_det_inv = self.inverse(x)\n"
            "    log_pz = standard_normal_logprob(z).sum(axis=-1)\n"
            "    log_px = log_pz + log_det_inv  # log_det_inv = 0\n"
            "    return log_px"
        )

    def sample(
        self,
        num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from the learned distribution.

        Args:
            num_samples: Number of samples

        Returns:
            x: Generated samples, shape (num_samples, dim)
            log_px: Log probability, shape (num_samples,)

        Raises:
            NotImplementedError: Needs forward implementation
        """
        raise NotImplementedError(
            "Implement sample():\n"
            "    z = np.random.standard_normal((num_samples, self.dim))\n"
            "    x, log_det = self.forward(z)\n"
            "    log_pz = standard_normal_logprob(z).sum(axis=-1)\n"
            "    log_px = log_pz - log_det  # log_det = 0\n"
            "    return x, log_px"
        )


class NICETraining:
    """
    Training utilities for NICE.

    Standard approach:
        1. Maximize log p(x) = log p_z(z) (since det = 1)
        2. Use gradient descent on -log p(x)
        3. Since log|det| = 0, only the base distribution matters!

    This means training NICE is equivalent to learning an invertible
    mapping that makes the data look like the base distribution.

    Loss function:
        L = E_x[-log p(x)]
          = E_x[-log p_z(T^{-1}(x))]
    """

    @staticmethod
    def compute_loss(
        model: NICE,
        x: np.ndarray
    ) -> float:
        """
        Compute negative log-likelihood loss.

        Args:
            model: NICE model
            x: Data batch

        Returns:
            loss: Scalar loss (negative log-likelihood)

        Raises:
            NotImplementedError: Needs log_prob implementation
        """
        raise NotImplementedError(
            "Implement loss computation:\n"
            "    log_px = model.log_prob(x)\n"
            "    loss = -log_px.mean()\n"
            "    return loss"
        )


if __name__ == "__main__":
    print("NICE: Non-linear Independent Components Estimation")
    print("=" * 70)
    print("\nKey Innovation: Coupling Layers with det(J) = 1")
    print("\nArchitecture:")
    print("  z -> Coupling(even) -> Coupling(odd) -> Coupling(even) -> ... -> x")
    print("\nKey Property:")
    print("  log|det(∂T/∂z)| = 0 (volume preserving)")
    print("\nAdvantages:")
    print("  + Simple: Jacobian determinant is always 1")
    print("  + Efficient: Only one network per layer")
    print("  + Invertible by design")
    print("\nDisadvantages:")
    print("  - Less expressive than RealNVP (no scaling)")
    print("  - Information bottleneck between partitions")
    print("\nImplementation Checklist:")
    print("  [ ] AdditiveCouplingLayer.forward()")
    print("  [ ] AdditiveCouplingLayer.inverse()")
    print("  [ ] NICE.__init__() - build layers")
    print("  [ ] NICE.forward() - compose layers")
    print("  [ ] NICE.inverse() - reverse composition")
    print("  [ ] NICE.log_prob() - change of variables")
    print("  [ ] NICE.sample() - generate from base dist")
