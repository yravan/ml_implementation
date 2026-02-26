"""
Model Interpretability.

This module implements methods for understanding and explaining
neural network predictions.

Modules:
    - saliency: Gradient-based attribution methods
"""

from .saliency import (
    VanillaGradients,
    GradientTimesInput,
    IntegratedGradients,
    SmoothGrad,
    GradCAM,
    SHAP,
    visualize_attribution
)

__all__ = [
    'VanillaGradients',
    'GradientTimesInput',
    'IntegratedGradients',
    'SmoothGrad',
    'GradCAM',
    'SHAP',
    'visualize_attribution',
]
