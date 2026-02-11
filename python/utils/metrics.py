"""
Evaluation Metrics
==================

Metrics for evaluating classification, regression, and ranking models.

Theory
------
Proper evaluation is crucial for understanding model performance:

**Classification Metrics**:
- Accuracy: Fraction correct. Misleading for imbalanced classes.
- Precision: Of predicted positives, fraction actually positive. High = few false positives.
- Recall: Of actual positives, fraction predicted positive. High = few false negatives.
- F1: Harmonic mean of precision and recall. Balances both.
- AUC-ROC: Area under ROC curve. Measures ranking quality regardless of threshold.

**Regression Metrics**:
- MSE: Mean squared error. Emphasizes large errors (quadratic penalty).
- MAE: Mean absolute error. More robust to outliers.
- RMSE: Root MSE. Same units as target.
- R²: Coefficient of determination. 1 = perfect, 0 = mean baseline, <0 = worse than mean.

**When to use what**:
- Balanced classification: Accuracy is fine
- Imbalanced classification: Use F1, AUC-ROC, precision@k
- Regression: MSE if outliers matter, MAE if robust to outliers
- Ranking: AUC-ROC, NDCG, MAP

Math
----
# Confusion Matrix for binary classification:
#                  Predicted
#                  Pos    Neg
# Actual  Pos      TP     FN
#         Neg      FP     TN
#
# Precision = TP / (TP + FP)  -- "Of predicted positives, how many are correct?"
# Recall    = TP / (TP + FN)  -- "Of actual positives, how many did we find?"
# F1        = 2 * P * R / (P + R)  -- Harmonic mean
#
# ROC Curve: Plot TPR vs FPR for different thresholds
#   TPR = TP / (TP + FN) = Recall
#   FPR = FP / (FP + TN)
#
# AUC = Area under ROC curve = P(score(pos) > score(neg))

# Regression:
# MSE  = (1/n) * Σ(y_i - ŷ_i)²
# MAE  = (1/n) * Σ|y_i - ŷ_i|
# R²   = 1 - SS_res/SS_tot = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²

References
----------
- scikit-learn Metrics documentation
  https://scikit-learn.org/stable/modules/model_evaluation.html
- "The Relationship Between Precision-Recall and ROC Curves"
  https://www.biostat.wisc.edu/~page/rocpr.pdf
- Google's ML Crash Course: Classification Metrics
  https://developers.google.com/machine-learning/crash-course/classification

Implementation Notes
--------------------
- For multi-class, compute per-class then average (micro, macro, weighted)
- AUC requires predicted probabilities, not just class labels
- Be careful with edge cases (all same class, division by zero)
- Confidence intervals: bootstrap or exact binomial for accuracy
"""

# Implementation Status: NOT STARTED
# Complexity: Easy
# Prerequisites: None (foundational module)

import numpy as np
from typing import Tuple, Optional, Dict, List


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Accuracy = (correct predictions) / (total predictions)

    Simple but can be misleading for imbalanced datasets.
    E.g., 99% accuracy means nothing if 99% of data is one class.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)

    Returns:
        Accuracy score in [0, 1]

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> accuracy(y_true, y_pred)
        0.8
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute confusion matrix.

    Element (i, j) is the count of samples with true class i predicted as class j.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        num_classes: Number of classes (inferred if None)

    Returns:
        Confusion matrix of shape (num_classes, num_classes)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 2, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 2, 0])
        >>> confusion_matrix(y_true, y_pred)
        array([[1, 1, 0],   # Class 0: 1 correct, 1 predicted as class 1
               [0, 2, 0],   # Class 1: 2 correct
               [1, 0, 1]])  # Class 2: 1 correct, 1 predicted as class 0
    """
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = 'binary',
                        pos_label: int = 1) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    For binary classification:
    - Precision: TP / (TP + FP) - accuracy of positive predictions
    - Recall: TP / (TP + FN) - coverage of actual positives
    - F1: 2 * P * R / (P + R) - harmonic mean

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary' for binary classification,
                'macro' for unweighted mean across classes,
                'micro' for global TP/FP/FN,
                'weighted' for weighted mean by support
        pos_label: Positive class label for binary

    Returns:
        Tuple of (precision, recall, f1)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 1, 0, 0])
        >>> precision_recall_f1(y_true, y_pred)
        (0.667, 0.5, 0.571)  # P=2/3, R=2/4, F1=2*P*R/(P+R)

    Trade-offs:
        - High precision, low recall: Conservative, few false positives
        - Low precision, high recall: Liberal, few false negatives
        - F1 balances both equally
    """
    eps = 1e-10

    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return precision, recall, f1

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    precisions = np.zeros(n_classes)
    recalls = np.zeros(n_classes)
    f1s = np.zeros(n_classes)
    supports = np.zeros(n_classes)

    for i, cls in enumerate(classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precisions[i] = tp / (tp + fp + eps)
        recalls[i] = tp / (tp + fn + eps)
        f1s[i] = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i] + eps)
        supports[i] = np.sum(y_true == cls)

    if average == 'macro':
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)
    elif average == 'micro':
        total_tp = np.sum([(y_true == cls) & (y_pred == cls) for cls in classes])
        total_fp = np.sum([(y_true != cls) & (y_pred == cls) for cls in classes])
        total_fn = np.sum([(y_true == cls) & (y_pred != cls) for cls in classes])
        precision = total_tp / (total_tp + total_fp + eps)
        recall = total_tp / (total_tp + total_fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return precision, recall, f1
    elif average == 'weighted':
        total_support = np.sum(supports)
        weights = supports / total_support
        return np.sum(precisions * weights), np.sum(recalls * weights), np.sum(f1s * weights)
    else:
        raise ValueError(f"Unknown average type: {average}")


def roc_curve(y_true: np.ndarray, y_scores: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.

    ROC = Receiver Operating Characteristic
    Plots True Positive Rate vs False Positive Rate at various thresholds.

    Args:
        y_true: Binary labels (0 or 1)
        y_scores: Predicted probabilities for positive class

    Returns:
        Tuple of (fpr, tpr, thresholds)
        - fpr: False positive rates
        - tpr: True positive rates (recall)
        - thresholds: Decision thresholds

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    """
    # Sort by decreasing score
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    y_scores_sorted = y_scores[order]

    # Compute TPR and FPR at each threshold
    tps = np.cumsum(y_true_sorted)  # True positives at each threshold
    fps = np.cumsum(1 - y_true_sorted)  # False positives

    total_positives = tps[-1]
    total_negatives = fps[-1]

    # Handle edge case where all samples are one class
    if total_positives == 0:
        tpr = np.zeros_like(tps, dtype=float)
    else:
        tpr = tps / total_positives

    if total_negatives == 0:
        fpr = np.zeros_like(fps, dtype=float)
    else:
        fpr = fps / total_negatives

    # Add (0, 0) point at the beginning
    fpr = np.concatenate([[0], fpr])
    tpr = np.concatenate([[0], tpr])
    thresholds = np.concatenate([[y_scores_sorted[0] + 1], y_scores_sorted])

    return fpr, tpr, thresholds


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC-ROC).

    AUC measures the probability that a randomly chosen positive example
    is ranked higher than a randomly chosen negative example.

    - AUC = 1.0: Perfect classifier
    - AUC = 0.5: Random classifier
    - AUC < 0.5: Worse than random (flip predictions)

    Args:
        y_true: Binary labels
        y_scores: Predicted probabilities for positive class

    Returns:
        AUC score in [0, 1]

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> auc_roc(y_true, y_scores)
        0.75
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Trapezoidal integration
    return np.trapz(tpr, fpr)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error for regression.

    MSE = (1/n) * Σ(y_i - ŷ_i)²

    Sensitive to outliers due to squared term.
    Commonly used loss function for regression.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE value (non-negative)

    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> mse(y_true, y_pred)
        0.375
    """
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error for regression.

    MAE = (1/n) * Σ|y_i - ŷ_i|

    More robust to outliers than MSE.
    Gradient is constant (±1), which can help optimization.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value (non-negative)
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = √MSE

    Has the same units as the target variable, making it more interpretable.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value (non-negative)
    """
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination (R² score).

    R² = 1 - SS_res / SS_tot
       = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²

    Interpretation:
    - R² = 1: Perfect predictions
    - R² = 0: Model predicts the mean (baseline)
    - R² < 0: Model is worse than predicting the mean

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² score

    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> r2_score(y_true, y_pred)
        0.948...
    """
    eps = 1e-10
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + eps)


def top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int = 5) -> float:
    """
    Top-k accuracy: correct if true label is in top-k predictions.

    Common metric for ImageNet (top-5 accuracy) and retrieval tasks.

    Args:
        y_true: True labels, shape (n_samples,)
        y_probs: Predicted probabilities, shape (n_samples, num_classes)
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy in [0, 1]

    Example:
        >>> y_true = np.array([0, 1, 2])
        >>> y_probs = np.array([[0.8, 0.1, 0.1],   # Top-1: class 0 ✓
        ...                     [0.3, 0.4, 0.3],   # Top-1: class 1 ✓
        ...                     [0.4, 0.4, 0.2]])  # Top-1: class 0, but true is 2
        >>> top_k_accuracy(y_true, y_probs, k=1)
        0.667
        >>> top_k_accuracy(y_true, y_probs, k=2)  # class 2 in top-2
        1.0
    """
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]  # Top-k indices
    correct = np.any(top_k_preds == y_true[:, None], axis=1)
    return np.mean(correct)


def log_loss(y_true: np.ndarray, y_probs: np.ndarray, eps: float = 1e-15) -> float:
    """
    Log loss (cross-entropy loss) for probabilistic predictions.

    Log loss = -(1/n) * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]

    For multi-class:
    Log loss = -(1/n) * Σ_i Σ_c y_ic * log(p_ic)

    Heavily penalizes confident wrong predictions.

    Args:
        y_true: True labels (integers for multi-class, 0/1 for binary)
        y_probs: Predicted probabilities
        eps: Clip probabilities to [eps, 1-eps] for stability

    Returns:
        Log loss value (non-negative, lower is better)
    """
    y_probs = np.clip(y_probs, eps, 1 - eps)

    # Handle multi-class case (y_probs is 2D)
    if y_probs.ndim == 2:
        n_samples = len(y_true)
        # Get probability of true class for each sample
        log_probs = np.log(y_probs[np.arange(n_samples), y_true])
        return -np.mean(log_probs)
    else:
        # Binary case
        return -np.mean(y_true * np.log(y_probs) + (1 - y_true) * np.log(1 - y_probs))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None) -> Dict:
    """
    Generate a detailed classification report.

    Computes per-class precision, recall, F1, and support (count).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional names for classes

    Returns:
        Dictionary with per-class and overall metrics

    Example output:
        {
            'class_0': {'precision': 0.8, 'recall': 0.9, 'f1': 0.85, 'support': 100},
            'class_1': {'precision': 0.7, 'recall': 0.6, 'f1': 0.65, 'support': 50},
            'macro_avg': {'precision': 0.75, 'recall': 0.75, 'f1': 0.75},
            'weighted_avg': {'precision': 0.77, 'recall': 0.8, 'f1': 0.78},
            'accuracy': 0.8
        }
    """
    eps = 1e-10
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    if class_names is None:
        class_names = [f'class_{i}' for i in classes]

    report = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, cls in enumerate(classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        support = np.sum(y_true == cls)

        report[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(support)
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports)

    # Macro average (unweighted mean)
    report['macro_avg'] = {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }

    # Weighted average (weighted by support)
    total_support = np.sum(supports)
    weights = supports / total_support
    report['weighted_avg'] = {
        'precision': np.sum(precisions * weights),
        'recall': np.sum(recalls * weights),
        'f1': np.sum(f1s * weights)
    }

    report['accuracy'] = accuracy(y_true, y_pred)

    return report


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Mean squared error
    """
    raise NotImplementedError(
        "TODO: Implement MSE\n"
        "Hint: return np.mean((y_true - y_pred) ** 2)"
    )


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        R-squared score in (-inf, 1], where 1 is perfect fit
    """
    raise NotImplementedError(
        "TODO: Implement R-squared\n"
        "Hint: ss_res = sum((y_true - y_pred)^2)\n"
        "      ss_tot = sum((y_true - mean(y_true))^2)\n"
        "      return 1 - ss_res / ss_tot"
    )
