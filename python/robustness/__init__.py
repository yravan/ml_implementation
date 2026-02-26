"""
Robustness and Adversarial ML.

This module implements methods for evaluating and improving
model robustness against adversarial examples and distribution shift.

Submodules:
    - attacks: Adversarial attack methods (FGSM, PGD, C&W)
    - defenses: Adversarial training, certified defenses
    - augmentation: Data augmentation for robustness
"""

from .attacks import (
    FGSM,
    PGD,
    CarliniWagner,
    DeepFool,
    compute_attack_success_rate,
    compute_perturbation_stats
)

__all__ = [
    # Attacks
    'FGSM',
    'PGD',
    'CarliniWagner',
    'DeepFool',
    'compute_attack_success_rate',
    'compute_perturbation_stats',
]
