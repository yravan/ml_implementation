"""
Neural Network Pruning Module.

Pruning removes redundant weights or structures from neural networks to reduce
computation and memory requirements while maintaining accuracy.

Theory:
    Pruning exploits the over-parameterization of neural networks. Not all weights
    contribute equally to the output - many can be removed with minimal impact.
    The Lottery Ticket Hypothesis suggests that dense networks contain sparse
    subnetworks that can train to similar accuracy from the same initialization.

Types of Pruning:
    1. Unstructured (Magnitude): Remove individual weights based on magnitude
    2. Structured: Remove entire filters/neurons/heads
    3. Dynamic: Prune during training vs post-training

References:
    - "Learning both Weights and Connections" (Han et al., 2015)
      https://arxiv.org/abs/1506.02626
    - "The Lottery Ticket Hypothesis" (Frankle & Carlin, 2019)
      https://arxiv.org/abs/1803.03635

Implementation Status: STUB
Complexity: Intermediate
Prerequisites: nn_core, optimization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod

__all__ = ['MagnitudePruning', 'StructuredPruning', 'LotteryTicket']


class PruningBase(ABC):
    """
    Abstract base class for pruning methods.

    Theory:
        Pruning methods identify and remove unimportant weights from neural networks.
        The key challenge is determining which weights are "unimportant" - this can
        be done based on magnitude, gradients, activation patterns, or learned masks.

    Math:
        For a weight matrix W, pruning creates a binary mask M:
            W_pruned = W ⊙ M
        where ⊙ is element-wise multiplication and M ∈ {0, 1}^{|W|}.
    """

    @abstractmethod
    def compute_mask(self, weights: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Compute binary mask for pruning.

        Args:
            weights: Weight tensor to prune
            sparsity: Target sparsity (fraction of weights to remove)

        Returns:
            Binary mask (1 = keep, 0 = prune)
        """
        raise NotImplementedError(
            "Implement mask computation. "
            "Hint: Use np.percentile to find threshold, return weights > threshold"
        )

    @abstractmethod
    def apply_pruning(self, model: Dict[str, np.ndarray], sparsity: float) -> Dict[str, np.ndarray]:
        """
        Apply pruning to model weights.

        Args:
            model: Dictionary of layer_name -> weights
            sparsity: Target sparsity level

        Returns:
            Dictionary of pruned weights
        """
        raise NotImplementedError


class MagnitudePruning(PruningBase):
    """
    Magnitude-based weight pruning.

    Theory:
        The simplest pruning method removes weights with smallest absolute values.
        This is based on the intuition that small weights contribute less to the
        output than large weights. Despite its simplicity, magnitude pruning is
        surprisingly effective and serves as a strong baseline.

    Math:
        Given weight W and sparsity s:
            threshold = percentile(|W|, s * 100)
            M[i,j] = 1 if |W[i,j]| > threshold else 0
            W_pruned = W ⊙ M

    Example:
        >>> pruner = MagnitudePruning()
        >>> weights = {'layer1.weight': np.random.randn(100, 50)}
        >>> pruned = pruner.apply_pruning(weights, sparsity=0.5)
        >>> # 50% of weights are now zero

    References:
        - "Learning both Weights and Connections" (Han et al., 2015)
          https://arxiv.org/abs/1506.02626
    """

    def __init__(self, global_pruning: bool = False):
        """
        Initialize magnitude pruning.

        Args:
            global_pruning: If True, compute global threshold across all layers.
                           If False, prune each layer independently.
        """
        self.global_pruning = global_pruning
        self.masks: Dict[str, np.ndarray] = {}

    def compute_mask(self, weights: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Compute binary mask based on weight magnitude.

        Implementation hints:
            1. Compute absolute values: abs_weights = np.abs(weights)
            2. Find threshold: threshold = np.percentile(abs_weights, sparsity * 100)
            3. Create mask: mask = (abs_weights > threshold).astype(np.float32)
        """
        raise NotImplementedError(
            "Implement magnitude-based mask computation. "
            "Use np.percentile(np.abs(weights), sparsity * 100) to find threshold."
        )

    def apply_pruning(
        self,
        model: Dict[str, np.ndarray],
        sparsity: float
    ) -> Dict[str, np.ndarray]:
        """
        Apply magnitude pruning to model.

        Implementation hints:
            1. If global_pruning: concatenate all weights, compute single threshold
            2. Otherwise: compute mask for each layer independently
            3. Store masks for potential fine-tuning
            4. Return weights multiplied by masks
        """
        raise NotImplementedError(
            "Implement pruning application. "
            "Iterate through model weights, compute masks, apply element-wise."
        )

    def get_sparsity(self, weights: Dict[str, np.ndarray]) -> float:
        """Compute actual sparsity of pruned weights."""
        total_params = sum(w.size for w in weights.values())
        zero_params = sum(np.sum(w == 0) for w in weights.values())
        return zero_params / total_params


class StructuredPruning(PruningBase):
    """
    Structured pruning removes entire filters, neurons, or attention heads.

    Theory:
        Unlike unstructured pruning which creates irregular sparsity patterns,
        structured pruning removes entire structures (filters, channels, neurons).
        This produces models that can run efficiently on standard hardware without
        specialized sparse matrix operations.

    Types:
        - Filter pruning: Remove entire convolutional filters
        - Channel pruning: Remove input channels
        - Neuron pruning: Remove neurons in fully-connected layers
        - Head pruning: Remove attention heads in transformers

    Math:
        For filter pruning with importance score s_i for filter i:
            Keep filters where s_i > percentile(s, sparsity * 100)

        Common importance metrics:
            - L1 norm: s_i = ||F_i||_1
            - L2 norm: s_i = ||F_i||_2
            - Batch normalization scale: s_i = |γ_i|

    References:
        - "Pruning Filters for Efficient ConvNets" (Li et al., 2017)
          https://arxiv.org/abs/1608.08710
        - "Learning Efficient Convolutional Networks" (Liu et al., 2017)
          https://arxiv.org/abs/1708.06519
    """

    def __init__(
        self,
        pruning_type: str = 'filter',
        importance_metric: str = 'l1'
    ):
        """
        Initialize structured pruning.

        Args:
            pruning_type: Type of structure to prune ('filter', 'channel', 'neuron', 'head')
            importance_metric: Metric to rank importance ('l1', 'l2', 'bn_scale')
        """
        self.pruning_type = pruning_type
        self.importance_metric = importance_metric

    def compute_importance(self, weights: np.ndarray, axis: int) -> np.ndarray:
        """
        Compute importance scores for structured elements.

        Implementation hints:
            - For L1: np.sum(np.abs(weights), axis=axis)
            - For L2: np.sqrt(np.sum(weights**2, axis=axis))
            - Return 1D array of importance scores
        """
        raise NotImplementedError(
            "Implement importance computation. "
            "Reduce along the appropriate axis using the selected metric."
        )

    def compute_mask(self, weights: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Compute structured mask.

        Implementation hints:
            1. Compute importance scores for each structure
            2. Find threshold from importance scores
            3. Create mask that preserves entire structures
        """
        raise NotImplementedError(
            "Implement structured mask computation. "
            "Mask entire filters/neurons, not individual weights."
        )

    def apply_pruning(
        self,
        model: Dict[str, np.ndarray],
        sparsity: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
        """
        Apply structured pruning and return pruned model architecture.

        Returns:
            - Pruned weights with reduced dimensions
            - Dictionary mapping layer names to kept indices

        Implementation hints:
            1. For each layer, compute importance scores
            2. Determine which structures to keep
            3. Slice weights to remove pruned structures
            4. Handle dependent layers (e.g., when removing conv filter,
               adjust next layer's input channels)
        """
        raise NotImplementedError(
            "Implement structured pruning. "
            "Return both sliced weights and kept indices for architecture modification."
        )


class LotteryTicket(PruningBase):
    """
    Lottery Ticket Hypothesis implementation.

    Theory:
        The Lottery Ticket Hypothesis states that randomly-initialized neural networks
        contain sparse subnetworks (winning tickets) that, when trained in isolation,
        can match the full network's accuracy. These subnetworks are found through
        iterative pruning and rewinding to initial weights.

    Algorithm:
        1. Initialize network with weights W_0
        2. Train to get weights W_T
        3. Prune p% of weights based on magnitude
        4. Reset remaining weights to W_0 (rewind)
        5. Repeat steps 2-4 until desired sparsity

    Key insight:
        The success of winning tickets depends on both the sparse structure AND
        the original initialization. Rewinding to initial weights is crucial.

    References:
        - "The Lottery Ticket Hypothesis" (Frankle & Carlin, 2019)
          https://arxiv.org/abs/1803.03635
        - "Stabilizing the Lottery Ticket Hypothesis" (Frankle et al., 2019)
          https://arxiv.org/abs/1903.01611

    Example:
        >>> lottery = LotteryTicket(pruning_rate=0.2, iterations=5)
        >>> # This achieves 1 - (1-0.2)^5 ≈ 67% sparsity
        >>> winning_ticket = lottery.find_winning_ticket(model, train_fn, init_weights)
    """

    def __init__(
        self,
        pruning_rate: float = 0.2,
        iterations: int = 5,
        rewind_epoch: int = 0
    ):
        """
        Initialize Lottery Ticket pruning.

        Args:
            pruning_rate: Fraction of remaining weights to prune per iteration
            iterations: Number of pruning iterations
            rewind_epoch: Epoch to rewind to (0 = initial, k = early epoch)
        """
        self.pruning_rate = pruning_rate
        self.iterations = iterations
        self.rewind_epoch = rewind_epoch
        self.masks: Dict[str, np.ndarray] = {}
        self.initial_weights: Dict[str, np.ndarray] = {}

    def compute_mask(self, weights: np.ndarray, sparsity: float) -> np.ndarray:
        """Compute mask using magnitude pruning."""
        raise NotImplementedError(
            "Use magnitude-based pruning. "
            "Same as MagnitudePruning.compute_mask"
        )

    def save_initial_weights(self, model: Dict[str, np.ndarray]) -> None:
        """Save initial weights for rewinding."""
        self.initial_weights = {k: v.copy() for k, v in model.items()}

    def rewind_weights(
        self,
        model: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Rewind weights to initial values, applying current masks.

        Implementation hints:
            1. For each layer, get initial weights
            2. Apply current mask to initial weights
            3. Return rewound model
        """
        raise NotImplementedError(
            "Implement weight rewinding. "
            "Return self.initial_weights[k] * self.masks[k] for each layer."
        )

    def find_winning_ticket(
        self,
        model: Dict[str, np.ndarray],
        train_fn: Callable,
        eval_fn: Callable,
        target_sparsity: float = 0.9
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Find winning ticket through iterative pruning.

        Args:
            model: Initial model weights
            train_fn: Function to train model, returns trained weights
            eval_fn: Function to evaluate model, returns accuracy
            target_sparsity: Target final sparsity

        Returns:
            - Winning ticket weights (sparse)
            - Mask dictionary

        Implementation hints:
            1. Save initial weights
            2. Loop for self.iterations:
                a. Train model
                b. Compute current sparsity
                c. If sparsity < target: prune pruning_rate of remaining weights
                d. Rewind to initial weights
            3. Return final sparse model
        """
        raise NotImplementedError(
            "Implement iterative magnitude pruning with rewinding. "
            "Train -> Prune -> Rewind -> Repeat"
        )

    def apply_pruning(
        self,
        model: Dict[str, np.ndarray],
        sparsity: float
    ) -> Dict[str, np.ndarray]:
        """Apply masks from lottery ticket search."""
        raise NotImplementedError("Apply stored masks to model weights.")


# Utility functions

def compute_sparsity(weights: Dict[str, np.ndarray]) -> float:
    """
    Compute sparsity of a model.

    Returns:
        Fraction of weights that are zero
    """
    total = sum(w.size for w in weights.values())
    zeros = sum(np.sum(w == 0) for w in weights.values())
    return zeros / total


def iterative_pruning_schedule(
    initial_sparsity: float,
    final_sparsity: float,
    num_iterations: int
) -> List[float]:
    """
    Generate iterative pruning schedule.

    Uses cubic sparsity schedule from Zhu & Gupta (2017).

    Returns:
        List of sparsity targets for each pruning iteration
    """
    sparsities = []
    for i in range(num_iterations):
        t = i / (num_iterations - 1) if num_iterations > 1 else 1
        s = final_sparsity + (initial_sparsity - final_sparsity) * (1 - t) ** 3
        sparsities.append(s)
    return sparsities
