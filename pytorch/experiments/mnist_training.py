"""
MNIST Training — PyTorch version.

Uses the shared train_utils for training loop, metrics, plotting.
Swap between MLP and CNN with model_type argument.

Usage:
    python -m pytorch.experiments.mnist_training
"""

import numpy as np
import time
from typing import List, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.experiments.mnist_data import load_mnist_datasets
from pytorch.experiments.train_utils import (
    print_header, print_metrics, format_time,
    top1_accuracy, train_epoch, evaluate,
    plot_training_history, plot_confusion_matrix,
    confusion_matrix, print_classification_report,
    save,
)


# =============================================================================
# Models
# =============================================================================

class MLP(nn.Module):
    """MLP for MNIST: Input(784) -> FC -> ReLU -> ... -> FC(10)"""

    def __init__(self, input_size: int = 784,
                 hidden_sizes: List[int] = [512, 256],
                 num_classes: int = 10):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.reshape(x.shape[0], -1))


class CNN(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self, in_channels: int = 1, conv_channels: int = 16,
                 kernel_size: int = 3, num_classes: int = 10):
        super().__init__()
        mid = conv_channels // 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, conv_channels, kernel_size, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(conv_channels * 28 * 28, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.reshape(x.shape[0], -1))


# =============================================================================
# Training
# =============================================================================

def train(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    optimizer_type: str = "adam",
    optimizer_params: Optional[dict] = None,
    model_type: str = "mlp",
    model_args: Optional[dict] = None,
    log_interval: int = 100,
    val_split: float = 0.1,
    seed: int = 42,
    save_plots: bool = True,
    output_dir: str = './outputs/mnist',
):
    """Full MNIST training pipeline."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if optimizer_params is None:
        optimizer_params = {}
    if model_args is None:
        model_args = {}

    # ── Config ───────────────────────────────────────────────────────
    print_header("MNIST Training (PyTorch)")
    print_header("Configuration", '-')
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer:     {optimizer_type}")
    print(f"  Model:         {model_type}")

    # ── Data ─────────────────────────────────────────────────────────
    print_header("Loading Data", '-')
    channels = (model_type != 'mlp')  # CNNs need channel dim
    train_loader, val_loader, test_loader = load_mnist_datasets(
        val_split=val_split, batch_size=batch_size, channels=channels
    )

    # ── Model ────────────────────────────────────────────────────────
    print_header("Model", '-')
    if model_type == 'mlp':
        model = MLP(**model_args)
    elif model_type == 'cnn':
        model = CNN(**model_args)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    model = model.to(device)
    print(f"  Device: {device}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(model)

    # ── Optimizer ────────────────────────────────────────────────────
    if optimizer_type == 'sgd':
        defaults = {'momentum': 0.9, 'weight_decay': 1e-4}
        defaults.update(optimizer_params)
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, **defaults)
    elif optimizer_type == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, **optimizer_params)
    elif optimizer_type == 'adamw':
        defaults = {'weight_decay': 1e-4}
        defaults.update(optimizer_params)
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, **defaults)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    criterion = F.cross_entropy

    # ── Train ────────────────────────────────────────────────────────
    print_header("Training", '-')
    metrics = {'acc': top1_accuracy}
    history = {f'{split}_{k}': [] for split in ('train', 'val') for k in ['loss'] + list(metrics)}

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n  Epoch {epoch}/{epochs}")
        print(f"  {'─' * 50}")

        train_results = train_epoch(
            model, train_loader, criterion, opt,
            metrics=metrics, log_interval=log_interval,
        )
        val_results = evaluate(model, val_loader, criterion, metrics=metrics)

        for key, val in train_results.items():
            history[f'train_{key}'].append(val)
        for key, val in val_results.items():
            history[f'val_{key}'].append(val)

        is_best = val_results['acc'] > best_val_acc
        if is_best:
            best_val_acc = val_results['acc']

        epoch_time = time.time() - epoch_start

        print_metrics(train_results, prefix="Train | ")
        print_metrics(val_results, prefix="Val   | ")
        if is_best:
            print(f"  ★ New Best!")
            save(model, output_path, epoch, tag='best')
        print(f"  Time: {format_time(epoch_time)}")

    total_time = time.time() - start_time

    # ── Test ──────────────────────────────────────────────────────────
    print_header("Test Evaluation", '-')
    test_results = evaluate(
        model, test_loader, criterion, metrics=metrics, collect_predictions=True,
    )
    cm = confusion_matrix(test_results['logits'], test_results['labels'])

    print_metrics(test_results, prefix="Test | ")

    # ── Summary ──────────────────────────────────────────────────────
    print_header("Training Complete!", '=')
    print(f"  Total time:          {format_time(total_time)}")
    print(f"  Best val accuracy:   {best_val_acc * 100:.2f}%")
    print(f"  Final test accuracy: {test_results['acc'] * 100:.2f}%")

    # ── Plots ────────────────────────────────────────────────────────
    if save_plots:
        plot_training_history(history, save_path=str(output_path / 'training_curves.png'))
        plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)],
                              save_path=str(output_path / 'confusion_matrix.png'))
    print_classification_report(cm, class_names=[str(i) for i in range(10)])

    print_header("Done!", '=')
    return {'model': model, 'history': history}


if __name__ == '__main__':
    train(
        epochs=10,
        batch_size=64,
        learning_rate=0.001,
        optimizer_type='adam',
        optimizer_params={'betas': (0.99, 0.999), 'weight_decay': 1e-4},
        model_type='mlp',
        log_interval=100,
        save_plots=True,
        output_dir='./outputs/mnist_cnn',
    )