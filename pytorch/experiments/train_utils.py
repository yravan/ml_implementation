"""
Training Utilities — framework-agnostic.

Works with both PyTorch and our custom numpy autograd framework.
The only difference is in how tensors are wrapped and loss values extracted,
handled by detecting the backend automatically.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Callable, Union
from pathlib import Path

# =============================================================================
# Backend Detection
# =============================================================================

def _is_torch_model(model):
    """Check if model is a PyTorch nn.Module (without importing torch at top level)."""
    try:
        import torch.nn as nn
        return isinstance(model, nn.Module)
    except ImportError:
        return False


def _is_torch_tensor(x):
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


# =============================================================================
# Metrics (pure numpy — work for both backends)
# =============================================================================

def top1_accuracy(logits, labels) -> float:
    """Compute top-1 accuracy. Accepts numpy arrays or torch tensors."""
    if _is_torch_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if _is_torch_tensor(labels):
        labels = labels.detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == labels)


def top5_accuracy(logits, labels) -> float:
    """Compute top-5 accuracy. Accepts numpy arrays or torch tensors."""
    if _is_torch_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if _is_torch_tensor(labels):
        labels = labels.detach().cpu().numpy()
    top5_preds = np.argsort(logits, axis=1)[:, -5:]
    correct = np.any(top5_preds == labels[:, None], axis=1)
    return np.mean(correct)


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    metrics: Optional[Dict[str, Callable]] = None,
    log_interval: int = 50,
    profile: bool = False,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Train for one epoch. Auto-detects PyTorch vs custom numpy framework.

    Args:
        model: nn.Module (PyTorch or custom)
        train_loader: DataLoader yielding (inputs, targets)
        criterion: Loss function — torch.nn.functional.cross_entropy or custom
        optimizer: Optimizer (torch or custom)
        metrics: Dict of {name: fn(logits, labels) -> float}
        log_interval: Print every N batches
        profile: Print per-batch timing
        debug: Print per-layer activations and gradients

    Returns:
        Dict with 'loss' and any metric names as keys
    """
    if metrics is None:
        metrics = {'top1': top1_accuracy}

    use_torch = _is_torch_model(model)

    model.train()
    total_loss = 0.0
    total_samples = 0
    batch_losses = []
    metric_totals = {name: 0.0 for name in metrics}

    pre_load = time.time()
    for batch_idx, batch in enumerate(train_loader):
        if profile:
            print(f"Data Loading: {time.time() - pre_load:.3f}s")
        batch_start = time.time()

        inputs, targets = batch[0], batch[1]

        if use_torch:
            import torch
            # PyTorch path: DataLoader may already yield tensors
            if not _is_torch_tensor(inputs):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if not _is_torch_tensor(targets):
                targets = torch.tensor(targets, dtype=torch.long)

            # GPU support
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward
            start = time.time()
            logits = model(inputs)
            if profile:
                print(f"Forward:  {time.time() - start:.3f}s")
            loss = criterion(logits, targets)

            # Backward
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            if profile:
                print(f"Backward: {time.time() - start:.3f}s")
            start = time.time()
            optimizer.step()
            if profile:
                print(f"Optimization: {time.time() - start:.3f}s")

            # Extract values
            batch_loss = loss.item()
            logits_np = logits.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

        else:
            # Custom numpy framework path
            from python.foundations.computational_graph import Tensor
            x = Tensor(inputs, requires_grad=True)
            y = Tensor(targets)

            # Forward
            start = time.time()
            logits = model(x)
            if profile:
                print(f"Forward:  {time.time() - start:.3f}s")
            loss = criterion(logits, y, reduction='mean')

            # Backward
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            if profile:
                print(f"Backward: {time.time() - start:.3f}s")
            start = time.time()
            optimizer.step()
            if profile:
                print(f"Optimization: {time.time() - start:.3f}s")

            # Extract values
            batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
            logits_np = logits.data
            targets_np = targets if isinstance(targets, np.ndarray) else targets.data

        if debug:
            _print_debug_info(model, use_torch)

        # Track loss
        batch_losses.append(batch_loss)
        n = len(inputs)
        total_loss += batch_loss * n
        total_samples += n

        # Track metrics
        for name, fn in metrics.items():
            metric_totals[name] += fn(logits_np, targets_np) * n

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
        pre_load = time.time()

    results = {'loss': total_loss / total_samples}
    for name in metrics:
        results[name] = metric_totals[name] / total_samples
    return results


def evaluate(
    model,
    val_loader,
    criterion,
    metrics: Optional[Dict[str, Callable]] = None,
    log_interval: int = 100,
    collect_predictions: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate model. Auto-detects PyTorch vs custom numpy framework.

    Returns:
        Dict with 'loss' and any metric names as keys.
        If collect_predictions, also includes 'logits' and 'labels'.
    """
    if metrics is None:
        metrics = {'top1': top1_accuracy}

    use_torch = _is_torch_model(model)

    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_totals = {name: 0.0 for name in metrics}
    all_logits = [] if collect_predictions else None
    all_labels = [] if collect_predictions else None

    # Disable gradients for evaluation
    if use_torch:
        import torch
        ctx = torch.no_grad()
        ctx.__enter__()

    for batch_idx, batch in enumerate(val_loader):
        inputs, targets = batch[0], batch[1]

        if use_torch:
            import torch
            if not _is_torch_tensor(inputs):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if not _is_torch_tensor(targets):
                targets = torch.tensor(targets, dtype=torch.long)

            device = next(model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_loss = loss.item()
            logits_np = logits.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

        else:
            from python.foundations.computational_graph import Tensor
            x = Tensor(inputs, requires_grad=False)
            y = Tensor(targets)

            logits = model(x)
            loss = criterion(logits, y, reduction='mean')

            batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
            logits_np = logits.data
            targets_np = targets if isinstance(targets, np.ndarray) else targets.data

        n = len(inputs)
        total_loss += batch_loss * n
        total_samples += n

        if collect_predictions:
            all_logits.append(logits_np)
            all_labels.append(targets_np)

        for name, fn in metrics.items():
            metric_totals[name] += fn(logits_np, targets_np) * n

        if log_interval and (batch_idx + 1) % log_interval == 0:
            print(f"    Eval batch {batch_idx + 1}/{len(val_loader)}")

    if use_torch:
        ctx.__exit__(None, None, None)

    results = {'loss': total_loss / total_samples}
    if collect_predictions:
        results['logits'] = np.concatenate(all_logits, axis=0)
        results['labels'] = np.concatenate(all_labels, axis=0)

    for name in metrics:
        results[name] = metric_totals[name] / total_samples
    return results


def _print_debug_info(model, use_torch: bool):
    """Print gradient norms for debugging."""
    for name, p in model.named_parameters():
        if use_torch:
            g = p.grad
            if g is not None:
                g_np = g.detach().cpu().numpy()
                print(f"{name:40s}  |grad|={np.abs(g_np).mean():.2e}  max={np.abs(g_np).max():.2e}")
            else:
                print(f"{name:40s}  grad=None")
        else:
            g = p.grad
            if g is not None:
                print(f"{name:40s}  |grad|={np.abs(g.data).mean():.2e}  max={np.abs(g.data).max():.2e}")
            else:
                print(f"{name:40s}  grad=None")


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def adjust_learning_rate(optimizer, epoch: int, initial_lr: float,
                         schedule: str = 'step', total_epochs: int = 90):
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
        lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    else:
        lr = initial_lr

    # Works for both PyTorch and custom optimizers
    if hasattr(optimizer, 'param_groups'):
        # PyTorch optimizer
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    else:
        # Custom optimizer
        optimizer.lr = lr
    return lr


# =============================================================================
# Checkpointing
# =============================================================================

def clean_checkpoints(output_path, keep=1):
    """Remove old checkpoints, keeping the most recent `keep`."""
    ckpts = sorted(output_path.glob('checkpoint_*.npy') if not _is_torch_available()
                   else list(output_path.glob('checkpoint_*.npy')) + list(output_path.glob('checkpoint_*.pt')),
                   key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep]:
        old.unlink()


def save(model, output_path, epoch, tag=None):
    """Save model checkpoint. Auto-detects PyTorch vs custom framework."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if _is_torch_model(model):
        import torch
        if tag is None:
            ckpt_path = output_path / f'checkpoint_{epoch}.pt'
            clean_checkpoints(output_path, keep=1)
        else:
            ckpt_path = output_path / f'{tag}.pt'
        torch.save(model.state_dict(), ckpt_path)
    else:
        model_state = model.state_dict()
        if tag is None:
            ckpt_path = output_path / f'checkpoint_{epoch}.npy'
            clean_checkpoints(output_path, keep=1)
        else:
            ckpt_path = output_path / f'{tag}.npy'
        np.save(str(ckpt_path), model_state)

    print(f"  Saved checkpoint to {ckpt_path}")


def load(model, ckpt_path):
    """Load model checkpoint. Auto-detects PyTorch vs custom framework."""
    ckpt_path = Path(ckpt_path)

    if _is_torch_model(model):
        import torch
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    else:
        state = np.load(str(ckpt_path), allow_pickle=True).item()
        model.load_state_dict(state)

    print(f"  Loaded checkpoint from {ckpt_path}")
    return model


def _is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


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
            continue
        if name == 'loss':
            parts.append(f"Loss: {val:.4f}")
        else:
            parts.append(f"{name}: {val*100:.2f}%")
    print(f"  {prefix}{' | '.join(parts)}")


# =============================================================================
# Plotting (pure matplotlib, no framework dependency)
# =============================================================================

def plot_training_history(history: Dict[str, List[float]],
                          save_path: Optional[str] = None):
    """
    Plot training curves. Automatically detects metrics from history keys.
    Expects keys like 'train_loss', 'val_loss', 'train_top1', 'val_top1', etc.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        metric_names = set()
        for key in history:
            name = key.replace('train_', '').replace('val_', '')
            if name not in ('loss', 'lr'):
                metric_names.add(name)
        metric_names = sorted(metric_names)

        n_plots = 1 + len(metric_names)
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
        plt.close()

    except ImportError:
        print("  [Warning] matplotlib not available, skipping plots")


def confusion_matrix(logits, labels, num_classes: Optional[int] = None) -> np.ndarray:
    """Compute confusion matrix from logits and labels."""
    if _is_torch_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if _is_torch_tensor(labels):
        labels = labels.detach().cpu().numpy()

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
    """Plot confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use('Agg')
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
        plt.close()

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