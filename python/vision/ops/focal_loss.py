"""
Focal Loss
==========

Focal Loss from "Focal Loss for Dense Object Detection" (RetinaNet).
https://arxiv.org/abs/1708.02002

Focal loss addresses class imbalance by down-weighting easy examples,
focusing training on hard negatives.
"""

import numpy as np


def sigmoid_focal_loss(
    inputs: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> np.ndarray:
    """
    Compute Focal Loss with sigmoid activation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y = 1, else 1 - p

    The modulating factor (1 - p_t)^gamma reduces loss for well-classified
    examples (where p_t is large), focusing on hard examples.

    Args:
        inputs: (N, *) raw logits (before sigmoid)
        targets: (N, *) binary targets (0 or 1)
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               gamma=0 is equivalent to cross-entropy
               Higher gamma = more focus on hard examples
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Focal loss values with specified reduction

    Note:
        alpha=0.25 means positive samples get weight 0.25,
        negative samples get weight 0.75. This counter-balances
        the typical class imbalance where negatives >> positives.
    """
    raise NotImplementedError("TODO: Implement sigmoid_focal_loss")
