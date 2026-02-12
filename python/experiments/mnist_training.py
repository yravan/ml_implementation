"""
MNIST MLP Training Experiment
=============================

Train a simple Multi-Layer Perceptron (MLP) on MNIST using our custom
deep learning framework with numpy-based autograd.

Architecture:
    Input (784) -> Linear(512) -> ReLU -> Linear(256) -> ReLU -> Linear(10)

Training:
    - SGD optimizer with momentum
    - Cross-entropy loss
    - Mini-batch training

This script demonstrates the full training loop including:
1. Data loading and preprocessing
2. Model definition using our Module API
3. Training with validation monitoring
4. Test evaluation with comprehensive metrics
5. Visualization of results using matplotlib

Usage:
    python -m python.experiments.mnist_mlp_training
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import urllib.request
import gzip
import struct

from python.experiments.mnist_data import load_mnist, split_validation
from python.experiments.train_utils import (
    print_header,
    top1_accuracy,
    train_epoch,
    evaluate,
    print_metrics,
    format_time,
    plot_training_history,
    plot_confusion_matrix,
    confusion_matrix,
    print_classification_report,
)
# Our framework imports
from python.foundations import Tensor
from python.foundations.computational_graph import print_graph
from python.nn_core import Module, Sequential, Linear, ReLU, normal_, Conv2d
from python.optimization import SGD, CrossEntropyLoss, Adam, AdamW


class DataLoader:
    """Simple mini-batch data loader."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 batch_size: int = 64, shuffle: bool = True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(images)

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]

            yield (
                self.images[batch_indices][:, None, :, :],
                self.labels[batch_indices]
            )

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


# =============================================================================
# Model Definition
# =============================================================================

class MLP(Module):
    """
    Multi-Layer Perceptron for MNIST classification.

    Architecture: Input(784) -> FC(512) -> ReLU -> FC(256) -> ReLU -> FC(10)
    """

    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [512, 256],
                 num_classes: int = 10):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(Linear(prev_size, hidden_size))
            layers.append(ReLU())
            prev_size = hidden_size

        layers.append(Linear(prev_size, num_classes))

        self.network = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        x = x.reshape(x.shape[0], -1)
        return self.network(x)


class CNN(Module):
    """
    Simple Convolutional Neural Network for MNIST classification.

    Architecture: Input(1, 28, 28) -> Conv2d(16, 3x3, pad=1) -> ReLU -> Flatten -> FC(10)
    """

    def __init__(self, in_channels: int = 1, conv_channels: int = 16,
                 kernel_size: int = 3, num_classes: int = 10):
        super().__init__()

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.num_classes = num_classes

        # Conv2d with padding=1 keeps spatial size the same for kernel_size=3
        # Output: (batch, conv_channels, 28, 28)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=conv_channels // 2,
            kernel_size=kernel_size,
            padding=1,
            bias=True
        )
        self.relu = ReLU()
        self.conv2 = Conv2d(
            in_channels=conv_channels // 2,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=True
        )

        # After flattening: conv_channels * 28 * 28
        self.fc = Linear(conv_channels * 28 * 28, num_classes)
        self._init_parameters(lambda x: normal_(x, mean=0, std=0.01))


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        # x shape: (batch, 1, 28, 28)
        B = x.shape[0]
        out = self.conv(x)       # -> (batch, conv_channels, 28, 28)
        out = self.relu(out)  # -> (batch, conv_channels, 28, 28)
        out = self.conv2(out)
        out = out.reshape(B, -1)
        out = self.fc(out)       # -> (batch, num_classes)
        return out


def load_data(val_split: float = 0.1, batch_size: int = 64):
    print_header("Loading Data", '-')

    data = load_mnist()
    train_images, train_labels, val_images, val_labels = split_validation(
        data["train_images"], data["train_labels"], val_split
    )
    test_images, test_labels = data["test_images"], data["test_labels"]

    print(f"  Training samples:   {len(train_images):,}")
    print(f"  Validation samples: {len(val_images):,}")
    print(f"  Test samples:       {len(test_images):,}")
    print(f"  Input shape:        (batch, 784)")
    print(f"  Num classes:        10")

    # Create data loaders
    train_loader = DataLoader(train_images, train_labels, batch_size, shuffle=True)
    val_loader = DataLoader(val_images, val_labels, batch_size, shuffle=False)
    test_loader = DataLoader(test_images, test_labels, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def load_model(model_type, **kwargs):
    print_header("Model Architecture", "-")

    if model_type == "mlp":
        model = MLP(input_size=784, **kwargs)
    elif model_type == "cnn":
        model = CNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"  Model: {model_type}")
    print(model)

    total_params = sum(np.prod(p.data.shape) for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    return model

def load_optimizer_and_loss(model, optimizer_type, optimizer_params, learning_rate, lr_schedule):
    params = list(model.parameters())

    if optimizer_type == "sgd":
        default_sgd = {"momentum": 0.9, "weight_decay": 1e-4}
        default_sgd.update(optimizer_params)
        opt = SGD(params, lr=learning_rate, **default_sgd)
    elif optimizer_type == "adam":
        opt = Adam(params, lr=learning_rate, **optimizer_params)
    elif optimizer_type == "adamw":
        default_adamw = {"weight_decay": 1e-4}
        default_adamw.update(optimizer_params)
        opt = AdamW(params, lr=learning_rate, **default_adamw)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    criterion = CrossEntropyLoss()

    print(f"  Optimizer: {optimizer_type} (lr={learning_rate})")
    print(f"  LR schedule: {lr_schedule}")
    print(f"  Loss: CrossEntropyLoss")
    return opt, criterion


def plot_example_predictions(images: np.ndarray, labels: np.ndarray,
                             logits: np.ndarray, num_examples: int = 16,
                             save_path: Optional[str] = None):
    """Plot example predictions (correct and incorrect)."""
    try:
        import matplotlib.pyplot as plt

        predictions = np.argmax(logits, axis=1)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        confidences = np.max(probs, axis=1)

        # Get correct and incorrect examples
        correct_mask = predictions == labels
        incorrect_mask = ~correct_mask

        n_show = min(num_examples // 2, 8)

        fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 5))

        # Correct predictions
        correct_indices = np.where(correct_mask)[0][:n_show]
        for i, idx in enumerate(correct_indices):
            ax = axes[0, i]
            img = images[idx].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f'✓ {predictions[idx]}\n({confidences[idx]*100:.1f}%)',
                        fontsize=10, color='green')

        axes[0, 0].set_ylabel('Correct', fontsize=12, color='green', fontweight='bold')

        # Incorrect predictions
        incorrect_indices = np.where(incorrect_mask)[0][:n_show]
        for i, idx in enumerate(incorrect_indices):
            ax = axes[1, i]
            img = images[idx].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f'✗ {predictions[idx]} (→{labels[idx]})\n({confidences[idx]*100:.1f}%)',
                        fontsize=10, color='red')

        axes[1, 0].set_ylabel('Incorrect', fontsize=12, color='red', fontweight='bold')

        plt.suptitle('Example Predictions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved example predictions to: {save_path}")

        plt.show()

    except ImportError:
        print("  [Warning] matplotlib not available, skipping plots")


# =============================================================================
# Main Training Script
# =============================================================================

def train(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.1,
    optimizer: str = "adam",
    optimizer_params: dict = None,
    model_type: str = "mlp",
    model_args: dict = {},
    log_interval: int = 10,
    val_split: float = 0.1,
    seed: int = 42,
    save_plots: bool = True,
    output_dir: str = './outputs'
):
    """
    Full MNIST training pipeline.

    Args:
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Initial learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization strength
        hidden_sizes: List of hidden layer sizes
        log_interval: Batches between log messages
        val_split: Fraction of training data for validation
        seed: Random seed for reproducibility
        save_plots: Whether to save plots to files
        output_dir: Directory for output files
    """
    # Set random seed
    np.random.seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print_header("MNIST MLP Training Experiment")

    # ==========================================================================
    # Configuration
    # ==========================================================================
    print_header("Configuration", '-')
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Optimizer_params: {optimizer_params}")
    print(f"  Model type: {model_type}")
    print(f"  Model args:  {model_args}")
    print(f"  Val split:     {val_split}")
    print(f"  Random seed:   {seed}")

    # ==========================================================================
    # Data Loading
    # ==========================================================================
    train_loader, val_loader, test_loader = load_data(val_split, batch_size)

    # ==========================================================================
    # Model Setup
    # ==========================================================================
    model = load_model(model_type, **model_args)

    # ==========================================================================
    # Optimizer and Loss
    # ==========================================================================
    opt, criterion = load_optimizer_and_loss(model, optimizer_type=optimizer, optimizer_params=optimizer_params, learning_rate=learning_rate, lr_schedule=None)

    metrics = {"acc": top1_accuracy}
    history = {f"train_{k}": [] for k in ["loss"] + list(metrics)}
    history.update({f"val_{k}": [] for k in ["loss"] + list(metrics)})

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print_header("Training", '-')

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        print(f"\n  Epoch {epoch}/{epochs}")
        print(f"  {'─' * 50}")

        # Train
        train_results = train_epoch(
            model, train_loader, criterion, opt,
            metrics=metrics, log_interval=log_interval,
        )

        # Validate
        val_results = evaluate(
            model, val_loader, criterion, metrics=metrics,
        )

        # Record history
        for key, val in train_results.items():
            history[f'train_{key}'].append(val)
        for key, val in val_results.items():
            history[f'val_{key}'].append(val)

        # Track best
        if val_results['acc'] > best_val_acc:
            best_val_acc = val_results['acc']
            best_marker = " ★ New Best!"
        else:
            best_marker = ""

        epoch_time = time.time() - epoch_start

        print_metrics(train_results, prefix="Train | ")
        print_metrics(val_results, prefix="Val   | ")
        if best_marker:
            print(f"  {best_marker}")
        print(f"  Time: {format_time(epoch_time)}")

    total_time = time.time() - start_time

    # ==========================================================================
    # Test Evaluation
    # ==========================================================================
    print_header("Test Evaluation", '-')

    test_results = evaluate(
        model, test_loader, criterion, metrics=metrics, collect_predictions=True,
    )
    cm = confusion_matrix(test_results["logits"], test_results["labels"])

    print_metrics(test_results, prefix="Test | ")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print_header("Training Complete!", '=')

    print(f"  Total training time: {format_time(total_time)}")
    print(f"  Best validation accuracy: { best_val_acc * 100:.2f}%")
    print(f"  Final test accuracy: {test_results['acc'] * 100:.2f}%")
    print()

    # ==========================================================================
    # Visualization
    # ==========================================================================
    print_header("Generating Visualizations", '-')

    if save_plots:
        plot_training_history(
            history,
            save_path=str(output_path / 'training_curves.png')
        )

        plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)],
                      save_path=str(output_path / 'confusion_matrix.png'))

        # plot_example_predictions(
        #     test_images, test_labels_arr, test_logits,
        #     save_path=str(output_path / 'example_predictions.png')
        # )
    else:
        plot_training_history(history)
        plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
        # plot_example_predictions(test_images, test_labels_arr, test_logits)
    print_classification_report(cm, class_names=[str(i) for i in range(10)])

    print_header("Done!", '=')

    return {
        'model': model,
        'history': history,
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    # Run with default hyperparameters
    results = train(
        epochs=10,
        batch_size=64,
        learning_rate=0.001,
        optimizer='adam',
        optimizer_params=dict(betas=(0.99, 0.999),weight_decay=1e-4,),
        # optimizer='sgd',
        # optimizer_params=dict(momentum=0.9,weight_decay=1e-4,),
        model_type='mlp',
        model_args=dict(hidden_sizes=[512, 256], num_classes=10),
        log_interval=100,
        val_split=0.1,
        seed=42,
        save_plots=True,
        output_dir='./outputs/mnist_cnn'
    )
