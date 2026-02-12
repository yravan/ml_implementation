
import numpy as np
import time
import os
import json
from typing import Tuple, List, Dict, Optional, Callable, Union
from pathlib import Path
from collections import OrderedDict

from python.utils.data_utils import DataLoader
# Our framework imports
from python.foundations import Tensor
from python.nn_core import Module
# =============================================================================
# Training Utilities
# =============================================================================

def top1_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute top-1 accuracy."""
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == labels)


def top5_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute top-5 accuracy."""
    top5_preds = np.argsort(logits, axis=1)[:, -5:]
    correct = np.any(top5_preds == labels[:, None], axis=1)
    return np.mean(correct)

# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: Module,
    train_loader: DataLoader,
    criterion: Module,
    optimizer,
    metrics: Optional[Dict[str, Callable]] = None,
    log_interval: int = 50,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: DataLoader yielding (inputs, targets, ...)
        criterion: Loss function
        optimizer: Optimizer
        metrics: Dict of {name: fn(logits, labels) -> float}
        log_interval: Print every N batches

    Returns:
        Dict with 'loss' and any metric names as keys
    """
    if metrics is None:
        metrics = {'top1': top1_accuracy}

    model.train()
    total_loss = 0.0
    total_samples = 0
    batch_losses = []

    # Accumulators for metrics (store running correct counts)
    metric_totals = {name: 0.0 for name in metrics}

    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()

        inputs, targets = batch[0], batch[1]

        x = Tensor(inputs, requires_grad=True)
        y = Tensor(targets)

        # Forward
        start = time.time()
        logits = model(x)
        # print(f"Forward:  {time.time() - start:.3f}s")
        loss = criterion(logits, y, reduction='mean')

        # Backward
        optimizer.zero_grad()
        start = time.time()
        loss.backward()
        # print(f"Backward: {time.time() - start:.3f}s")
        start = time.time()
        optimizer.step()
        # print(f"Optimization: {time.time() - start:.3f}s")

        # Track loss
        batch_loss = loss.data.item() if loss.data.ndim == 0 else loss.data.mean()
        batch_losses.append(batch_loss)
        n = len(inputs)
        total_loss += batch_loss * n
        total_samples += n

        # Track metrics
        for name, fn in metrics.items():
            metric_totals[name] += fn(logits.data, targets) * n

        # Log
        if log_interval and (batch_idx + 1) % log_interval == 0:
            batch_time = time.time() - batch_start
            avg_loss = np.mean(batch_losses[-log_interval:])
            parts = [f"Batch {batch_idx + 1:5d}/{len(train_loader)}",
                     f"Loss: {avg_loss:.4f}"]
            for name in metrics:
                val = metric_totals[name] / total_samples
                parts.append(f"{name}: {val*100:.2f}%")
            parts.append(f"{n / batch_time:.1f} img/s")
            print(f"    {' | '.join(parts)}")

    results = {'loss': total_loss / total_samples}
    for name in metrics:
        results[name] = metric_totals[name] / total_samples
    return results


def evaluate(
    model: Module,
    val_loader: DataLoader,
    criterion: Module,
    metrics: Optional[Dict[str, Callable]] = None,
    log_interval: int = 100,
    collect_predictions: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate model.

    Args:
        model: The model to evaluate
        val_loader: DataLoader yielding (inputs, targets, ...)
        criterion: Loss function
        metrics: Dict of {name: fn(logits, labels) -> float}
        log_interval: Print progress every N batches (0 to disable)

    Returns:
        Dict with 'loss' and any metric names as keys
    """
    if metrics is None:
        metrics = {'top1': top1_accuracy}

    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_totals = {name: 0.0 for name in metrics}
    all_logits = [] if collect_predictions else None
    all_labels = [] if collect_predictions else None

    for batch_idx, batch in enumerate(val_loader):
        inputs, targets = batch[0], batch[1]

        x = Tensor(inputs, requires_grad=False)
        y = Tensor(targets)

        logits = model(x)
        loss = criterion(logits, y, reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else loss.data.mean()
        n = len(inputs)
        total_loss += batch_loss * n
        total_samples += n

        if collect_predictions:
            all_logits.append(logits.data)
            all_labels.append(targets)

        for name, fn in metrics.items():
            metric_totals[name] += fn(logits.data, targets) * n

        if log_interval and (batch_idx + 1) % log_interval == 0:
            print(f"    Eval batch {batch_idx + 1}/{len(val_loader)}")

    results = {'loss': total_loss / total_samples}
    if collect_predictions:
        results['logits'] = np.concatenate(all_logits, axis=0)
        results['labels'] = np.concatenate(all_labels, axis=0)

    for name in metrics:
        results[name] = metric_totals[name] / total_samples
    return results


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def print_header(text: str, char: str = '=', width: int = 70):
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_metrics(results: Dict[str, float], prefix: str = ""):
    """Print a results dict nicely."""
    parts = []
    for name, val in results.items():
        if name in ('logits', 'labels'):
            continue  # skip collected arrays
        if name == 'loss':
            parts.append(f"Loss: {val:.4f}")
        else:
            parts.append(f"{name}: {val*100:.2f}%")
    print(f"  {prefix}{' | '.join(parts)}")


# =============================================================================
# Plotting
# =============================================================================

def plot_training_history(history: Dict[str, List[float]],
                          save_path: Optional[str] = None):
    """
    Plot training curves. Automatically detects metrics from history keys.

    Expects keys like 'train_loss', 'val_loss', 'train_top1', 'val_top1', etc.
    """
    try:
        import matplotlib.pyplot as plt

        # Find all metric names (everything except 'loss' and 'lr')
        metric_names = set()
        for key in history:
            name = key.replace('train_', '').replace('val_', '')
            if name not in ('loss', 'lr'):
                metric_names.add(name)
        metric_names = sorted(metric_names)

        n_plots = 1 + len(metric_names)  # loss + each metric
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Each metric
        for i, name in enumerate(metric_names):
            ax = axes[i + 1]
            train_key = f'train_{name}'
            val_key = f'val_{name}'
            if train_key in history:
                ax.plot(epochs, [v * 100 for v in history[train_key]],
                        'b-', label='Train', linewidth=2)
            if val_key in history:
                ax.plot(epochs, [v * 100 for v in history[val_key]],
                        'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{name} (%)')
            ax.set_title(name, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved training curves to: {save_path}")
        plt.show()

    except ImportError:
        print("  [Warning] matplotlib not available, skipping plots")

def confusion_matrix(logits: np.ndarray, labels: np.ndarray,
                     num_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute confusion matrix from logits and labels.

    Returns:
        (C, C) matrix where [i, j] = count of true class i predicted as class j
    """
    preds = np.argmax(logits, axis=1)
    if num_classes is None:
        num_classes = max(int(labels.max()), int(preds.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          normalize: bool = True,
                          save_path: Optional[str] = None,
                          title: str = 'Confusion Matrix',
                          figsize: Optional[Tuple[int, int]] = None):
    """Plot confusion matrix heatmap. Takes a precomputed (C,C) matrix."""
    try:
        import matplotlib.pyplot as plt

        num_classes = cm.shape[0]

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.where(row_sums > 0, cm / row_sums, 0.0)
        else:
            cm_display = cm.astype(float)

        if figsize is None:
            size = max(6, num_classes * 0.5)
            figsize = (size, size)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm_display, cmap='Blues', aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046)

        if class_names is not None:
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
        else:
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))

        # Cell text (skip if too many classes)
        if num_classes <= 25:
            thresh = cm_display.max() / 2.0
            for i in range(num_classes):
                for j in range(num_classes):
                    text = f'{cm_display[i, j]:.1%}' if normalize else str(cm[i, j])
                    color = 'white' if cm_display[i, j] > thresh else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            color=color, fontsize=7)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved confusion matrix to: {save_path}")
        plt.show()

    except ImportError:
        print("  [Warning] matplotlib not available, skipping plots")


def print_classification_report(cm: np.ndarray,
                                 class_names: Optional[List[str]] = None):
    """Print per-class precision, recall, F1 from confusion matrix."""
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(precision + recall > 0,
                  2 * precision * recall / (precision + recall), 0.0)

    max_name = max(len(n) for n in class_names)
    header = f"  {'Class':<{max_name}}  Precision  Recall     F1  Support"
    print(header)
    print(f"  {'─' * len(header.strip())}")

    for i in range(num_classes):
        print(f"  {class_names[i]:<{max_name}}  "
              f"{precision[i]:9.2%}  {recall[i]:6.2%}  {f1[i]:6.2%}  {support[i]:7d}")

    print(f"  {'─' * len(header.strip())}")
    total = support.sum()
    print(f"  {'macro avg':<{max_name}}  {precision.mean():9.2%}  {recall.mean():6.2%}  {f1.mean():6.2%}  {total:7d}")
    w_p = np.average(precision, weights=support) if total > 0 else 0
    w_r = np.average(recall, weights=support) if total > 0 else 0
    w_f1 = np.average(f1, weights=support) if total > 0 else 0
    print(f"  {'weighted avg':<{max_name}}  {w_p:9.2%}  {w_r:6.2%}  {w_f1:6.2%}  {total:7d}")
    print(f"  {'accuracy':<{max_name}}  {'':>9}  {'':>6}  {tp.sum()/total:6.2%}  {total:7d}")



def adjust_learning_rate(optimizer, epoch: int, initial_lr: float,
                         schedule: str = 'step'):
    """
    Adjust learning rate based on epoch.

    Standard ImageNet schedule: decay by 10x at epochs 30, 60, 80.
    """
    if schedule == 'step':
        if epoch < 30:
            lr = initial_lr
        elif epoch < 60:
            lr = initial_lr * 0.1
        elif epoch < 80:
            lr = initial_lr * 0.01
        else:
            lr = initial_lr * 0.001
    elif schedule == 'cosine':
        lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / 90))
    else:
        lr = initial_lr

    optimizer.lr = lr
    return lr
