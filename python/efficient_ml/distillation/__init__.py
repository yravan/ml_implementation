"""
Knowledge Distillation Module.

Knowledge distillation transfers knowledge from a large "teacher" model to a
smaller "student" model, enabling deployment of efficient models that retain
much of the teacher's performance.

Theory:
    Hinton et al. showed that soft probability outputs from a teacher contain
    more information than hard labels. The "dark knowledge" in the soft targets
    reveals relationships between classes that the student can learn from.

Loss Function:
    L = α * L_CE(y, σ(z_s)) + (1-α) * T² * L_KL(σ(z_t/T), σ(z_s/T))

    Where:
    - L_CE: Cross-entropy with hard labels
    - L_KL: KL divergence with soft targets
    - T: Temperature (higher = softer probabilities)
    - α: Balance between hard and soft targets

Types of Distillation:
    1. Response-based: Match output logits/probabilities
    2. Feature-based: Match intermediate representations
    3. Relation-based: Match relationships between samples

References:
    - "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
      https://arxiv.org/abs/1503.02531
    - "FitNets: Hints for Thin Deep Nets" (Romero et al., 2015)
      https://arxiv.org/abs/1412.6550

Implementation Status: STUB
Complexity: Intermediate
Prerequisites: nn_core, optimization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod

__all__ = ['KnowledgeDistillation', 'FeatureDistillation', 'SelfDistillation']


class DistillationBase(ABC):
    """
    Abstract base class for distillation methods.

    Theory:
        Distillation trains a student model to mimic a teacher model's behavior.
        The student learns from both the ground truth labels and the teacher's
        soft predictions, which contain rich inter-class relationship information.
    """

    @abstractmethod
    def compute_loss(
        self,
        student_output: np.ndarray,
        teacher_output: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute distillation loss.

        Args:
            student_output: Student model logits
            teacher_output: Teacher model logits
            labels: Ground truth labels

        Returns:
            - Total loss
            - Dictionary of individual loss components
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self,
        student: Dict[str, np.ndarray],
        teacher: Dict[str, np.ndarray],
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Perform one training step.

        Returns:
            - Loss value
            - Updated student weights
        """
        raise NotImplementedError


class KnowledgeDistillation(DistillationBase):
    """
    Standard Knowledge Distillation (Response-based).

    Theory:
        The student learns from both hard labels and soft targets from the teacher.
        Soft targets are created using temperature scaling, which reveals the
        teacher's confidence about class relationships.

    Math:
        Softmax with temperature:
            σ(z_i/T) = exp(z_i/T) / Σ_j exp(z_j/T)

        Distillation loss:
            L_distill = KL(σ(z_t/T) || σ(z_s/T))

        Combined loss:
            L = α * CE(y, σ(z_s)) + (1-α) * T² * KL(σ(z_t/T) || σ(z_s/T))

        Note: T² factor compensates for gradient magnitude with temperature.

    Example:
        >>> kd = KnowledgeDistillation(temperature=4.0, alpha=0.9)
        >>> # alpha=0.9 means 90% weight on distillation loss
        >>> loss, info = kd.compute_loss(student_logits, teacher_logits, labels)

    References:
        - "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
          https://arxiv.org/abs/1503.02531
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.9,
        reduction: str = 'mean'
    ):
        """
        Initialize knowledge distillation.

        Args:
            temperature: Softmax temperature (higher = softer)
            alpha: Weight for distillation loss (1-alpha for hard labels)
            reduction: Loss reduction method ('mean', 'sum')
        """
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def softmax_temperature(self, logits: np.ndarray, T: float) -> np.ndarray:
        """
        Compute softmax with temperature scaling.

        Implementation hints:
            1. Divide logits by temperature: z_T = logits / T
            2. Subtract max for numerical stability
            3. Apply softmax: exp(z_T) / sum(exp(z_T))
        """
        raise NotImplementedError(
            "Implement temperature-scaled softmax. "
            "Use np.exp(logits/T - max) / sum for stability."
        )

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence: KL(p || q).

        Implementation hints:
            KL = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
            Handle numerical issues with eps for log.
        """
        raise NotImplementedError(
            "Implement KL divergence. "
            "KL(p||q) = sum(p * log(p/q))"
        )

    def cross_entropy(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Implementation hints:
            1. Compute softmax probabilities
            2. CE = -sum(one_hot(labels) * log(probs))
        """
        raise NotImplementedError("Implement cross-entropy loss.")

    def compute_loss(
        self,
        student_output: np.ndarray,
        teacher_output: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Implementation hints:
            1. Compute soft targets: teacher_soft = softmax_temperature(teacher, T)
            2. Compute student soft: student_soft = softmax_temperature(student, T)
            3. Compute KL loss: L_kl = T² * KL(teacher_soft, student_soft)
            4. Compute CE loss: L_ce = cross_entropy(student, labels)
            5. Combined: L = alpha * L_kl + (1-alpha) * L_ce
        """
        raise NotImplementedError(
            "Implement combined loss. "
            "Balance KL divergence (soft) and CE (hard) losses."
        )

    def train_step(
        self,
        student: Dict[str, np.ndarray],
        teacher: Dict[str, np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        student_forward: Callable,
        teacher_forward: Callable,
        optimizer
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Perform one distillation training step.

        Implementation hints:
            1. Forward pass through teacher (no gradient)
            2. Forward pass through student
            3. Compute distillation loss
            4. Backpropagate and update student
        """
        raise NotImplementedError(
            "Implement training step. "
            "Teacher forward (detached) -> Student forward -> Loss -> Backward"
        )


class FeatureDistillation(DistillationBase):
    """
    Feature-based Knowledge Distillation (FitNets).

    Theory:
        Beyond matching final outputs, feature distillation trains the student
        to match the teacher's intermediate representations. This provides
        stronger supervisory signal and enables training deeper students.

    Architecture:
        Since teacher and student may have different dimensions, we use
        a "regressor" network to project student features to teacher dimensions:

        Student Features → Regressor → Matches → Teacher Features

    Math:
        Feature matching loss:
            L_feat = ||r(F_s) - F_t||² / (H * W * C)

        Where r is the regressor and F_s, F_t are feature maps.

    References:
        - "FitNets: Hints for Thin Deep Nets" (Romero et al., 2015)
          https://arxiv.org/abs/1412.6550
    """

    def __init__(
        self,
        hint_layers: List[str],
        guided_layers: List[str],
        regressor_dims: List[Tuple[int, int]] = None
    ):
        """
        Initialize feature distillation.

        Args:
            hint_layers: Teacher layer names to use as hints
            guided_layers: Corresponding student layer names to guide
            regressor_dims: List of (in_dim, out_dim) for regressors
        """
        self.hint_layers = hint_layers
        self.guided_layers = guided_layers
        self.regressor_dims = regressor_dims
        self.regressors: List[Dict[str, np.ndarray]] = []

    def init_regressors(self) -> None:
        """
        Initialize regressor networks for dimension matching.

        Implementation hints:
            For each (in_dim, out_dim) pair:
            - Create 1x1 conv or linear layer to match dimensions
        """
        raise NotImplementedError(
            "Initialize regressor networks. "
            "Use 1x1 conv for spatial features, linear for vectors."
        )

    def compute_feature_loss(
        self,
        student_features: List[np.ndarray],
        teacher_features: List[np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute feature matching loss.

        Implementation hints:
            1. For each (student_feat, teacher_feat) pair:
                a. Apply regressor to student_feat
                b. Compute MSE with teacher_feat
                c. Normalize by feature dimensions
            2. Sum/average all feature losses
        """
        raise NotImplementedError(
            "Implement feature matching loss. "
            "L2 distance between (regressed) student and teacher features."
        )

    def compute_loss(
        self,
        student_output: np.ndarray,
        teacher_output: np.ndarray,
        labels: np.ndarray,
        student_features: List[np.ndarray] = None,
        teacher_features: List[np.ndarray] = None,
        feature_weight: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined output and feature loss.

        Implementation hints:
            1. Compute output distillation loss (like KD)
            2. Compute feature matching loss
            3. Combine: L = L_output + feature_weight * L_feature
        """
        raise NotImplementedError(
            "Combine output and feature distillation losses."
        )

    def train_step(
        self,
        student: Dict[str, np.ndarray],
        teacher: Dict[str, np.ndarray],
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Perform feature distillation training step."""
        raise NotImplementedError(
            "Implement training with feature extraction. "
            "Need to capture intermediate features during forward pass."
        )


class SelfDistillation(DistillationBase):
    """
    Self-Distillation (Born-Again Networks).

    Theory:
        In self-distillation, the student has the same architecture as the teacher.
        Surprisingly, training a student to match a teacher of identical capacity
        often improves upon the original. This can be repeated multiple generations.

    Process:
        1. Train teacher model normally
        2. Train identical student with soft targets from teacher
        3. Student often outperforms teacher!
        4. Repeat: new student uses previous student as teacher

    Hypotheses for improvement:
        - Soft labels provide regularization
        - Averaging over multiple teachers reduces variance
        - Soft labels smooth decision boundaries

    References:
        - "Born-Again Neural Networks" (Furlanello et al., 2018)
          https://arxiv.org/abs/1805.04770
        - "Self-Training with Noisy Student" (Xie et al., 2020)
          https://arxiv.org/abs/1911.04252
    """

    def __init__(
        self,
        generations: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.5
    ):
        """
        Initialize self-distillation.

        Args:
            generations: Number of student generations to train
            temperature: Softmax temperature for soft targets
            alpha: Weight for soft target loss
        """
        self.generations = generations
        self.temperature = temperature
        self.alpha = alpha
        self.teachers: List[Dict[str, np.ndarray]] = []

    def compute_loss(
        self,
        student_output: np.ndarray,
        teacher_output: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute self-distillation loss.

        Same as standard KD but teacher = previous generation student.
        """
        raise NotImplementedError("Same as KnowledgeDistillation.compute_loss")

    def train_generation(
        self,
        model_fn: Callable,
        train_data: Tuple[np.ndarray, np.ndarray],
        epochs: int,
        generation: int
    ) -> Dict[str, np.ndarray]:
        """
        Train one generation of self-distillation.

        Implementation hints:
            1. If generation == 0: train normally with hard labels
            2. If generation > 0: train with soft targets from previous generation
            3. Store trained model as new teacher
        """
        raise NotImplementedError(
            "Implement one generation of training. "
            "First gen is normal training, subsequent gens use previous as teacher."
        )

    def train(
        self,
        model_fn: Callable,
        train_data: Tuple[np.ndarray, np.ndarray],
        epochs_per_generation: int
    ) -> List[Dict[str, np.ndarray]]:
        """
        Train multiple generations of self-distillation.

        Returns:
            List of models from each generation
        """
        raise NotImplementedError(
            "Train multiple generations. "
            "Each generation uses previous as teacher."
        )

    def train_step(
        self,
        student: Dict[str, np.ndarray],
        teacher: Dict[str, np.ndarray],
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Perform one training step."""
        raise NotImplementedError("Standard KD training step.")


# Utility functions

def ensemble_distillation(
    teachers: List[Dict[str, np.ndarray]],
    teacher_forwards: List[Callable],
    x: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute soft targets from ensemble of teachers.

    Returns:
        Average soft probabilities across all teachers
    """
    soft_targets = []
    for model, forward in zip(teachers, teacher_forwards):
        logits = forward(model, x)
        # Temperature softmax
        exp_logits = np.exp(logits / temperature)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        soft_targets.append(probs)

    return np.mean(soft_targets, axis=0)


def label_smoothing(
    labels: np.ndarray,
    num_classes: int,
    smoothing: float = 0.1
) -> np.ndarray:
    """
    Apply label smoothing.

    Hard labels become: (1-smoothing) for correct class,
                       smoothing/(num_classes-1) for others
    """
    one_hot = np.eye(num_classes)[labels]
    smooth_labels = one_hot * (1 - smoothing) + smoothing / num_classes
    return smooth_labels
