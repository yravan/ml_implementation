"""
Bayesian Deep Learning.

This module implements Bayesian approaches to neural networks,
enabling uncertainty quantification and probabilistic predictions.

Modules:
    - bnn: Bayesian Neural Networks (Bayes by Backprop, MC Dropout, SWAG)
"""

from .bnn import (
    BayesianLinear,
    BayesianMLP,
    MCDropout,
    SWAG
)

__all__ = [
    'BayesianLinear',
    'BayesianMLP',
    'MCDropout',
    'SWAG',
]
