"""
Adversarial Attacks.

This module implements adversarial attack methods for evaluating
model robustness.

Modules:
    - adversarial: FGSM, PGD, C&W, DeepFool
"""

from .adversarial import (
    FGSM,
    PGD,
    CarliniWagner,
    DeepFool,
    compute_attack_success_rate,
    compute_perturbation_stats
)

__all__ = [
    'FGSM',
    'PGD',
    'CarliniWagner',
    'DeepFool',
    'compute_attack_success_rate',
    'compute_perturbation_stats',
]
