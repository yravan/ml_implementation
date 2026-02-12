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

# Our framework imports
from python.foundations import Tensor
from python.foundations.computational_graph import print_graph
from python.nn_core import Module, Sequential, Linear, ReLU
from python.optimization import SGD, CrossEntropyLoss


# =============================================================================
# Data Loading Utilities
# =============================================================================

def download_mnist(data_dir: str = './data') -> None:
    """Download MNIST dataset if not already present."""
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for filename in files:
        filepath = data_path / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            url = base_url + filename
            urllib.request.urlretrieve(url, filepath)
            print(f"  Downloaded to {filepath}")


def load_mnist_images(filepath: str) -> np.ndarray:
    """Load MNIST images from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return images.astype(np.float64) / 255.0  # Normalize to [0, 1]


def load_mnist_labels(filepath: str) -> np.ndarray:
    """Load MNIST labels from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir: str = './data') -> Dict[str, np.ndarray]:
    """Load full MNIST dataset."""
    download_mnist(data_dir)

    data_path = Path(data_dir)

    train_images = load_mnist_images(data_path / 'train-images-idx3-ubyte.gz')
    train_labels = load_mnist_labels(data_path / 'train-labels-idx1-ubyte.gz')
    test_images = load_mnist_images(data_path / 't10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels(data_path / 't10k-labels-idx1-ubyte.gz')

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }


def split_validation(train_images: np.ndarray, train_labels: np.ndarray,
                     val_split: float = 0.1) -> Tuple[np.ndarray, ...]:
    """Split training data into train and validation sets."""
    n_train = len(train_images)
    n_val = int(n_train * val_split)

    # Shuffle indices
    indices = np.random.permutation(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return (
        train_images[train_indices],
        train_labels[train_indices],
        train_images[val_indices],
        train_labels[val_indices]
    )


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
                self.images[batch_indices],
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
        return self.network(x)


# =============================================================================
# Training Utilities
# =============================================================================

def compute_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy."""
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == labels)


def compute_per_class_accuracy(logits: np.ndarray, labels: np.ndarray,
                                num_classes: int = 10) -> np.ndarray:
    """Compute per-class accuracy."""
    predictions = np.argmax(logits, axis=1)
    per_class_acc = np.zeros(num_classes)

    for c in range(num_classes):
        mask = labels == c
        if np.sum(mask) > 0:
            per_class_acc[c] = np.mean(predictions[mask] == labels[mask])

    return per_class_acc


def format_time(seconds: float) -> str:
    """Format time duration nicely."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_header(text: str, char: str = '=', width: int = 70):
    """Print a formatted header."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_progress_bar(current: int, total: int, prefix: str = '',
                       suffix: str = '', length: int = 30):
    """Print a progress bar."""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    print(f'\r{prefix} |{bar}| {percent*100:.1f}% {suffix}', end='', flush=True)


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model: Module, train_loader: DataLoader,
                criterion: Module, optimizer: SGD,
                log_interval: int = 100) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_losses = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Convert to Tensors
        x = Tensor(images, requires_grad=True)
        y = Tensor(labels)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, y, reduction='mean')

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        batch_loss = loss.data.item() if loss.data.ndim == 0 else loss.data.mean()
        batch_losses.append(batch_loss)
        total_loss += batch_loss * len(images)

        predictions = np.argmax(logits.data, axis=1)
        total_correct += np.sum(predictions == labels)
        total_samples += len(images)

        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = np.mean(batch_losses[-log_interval:])
            running_acc = total_correct / total_samples
            print(f"    Batch {batch_idx + 1:4d}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {running_acc*100:.2f}%")

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model: Module, data_loader: DataLoader,
             criterion: Module) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.

    Returns:
        Tuple of (loss, accuracy, all_logits, all_labels)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_logits = []
    all_labels = []

    for images, labels in data_loader:
        x = Tensor(images, requires_grad=False)
        y = Tensor(labels)

        logits = model(x)
        loss = criterion(logits, y, reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else loss.data.mean()
        total_loss += batch_loss * len(images)

        predictions = np.argmax(logits.data, axis=1)
        total_correct += np.sum(predictions == labels)
        total_samples += len(images)

        all_logits.append(logits.data)
        all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return (total_loss / total_samples,
            total_correct / total_samples,
            all_logits, all_labels)


# =============================================================================
# Visualization
# =============================================================================

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training history (loss and accuracy curves)."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1 = axes[0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2 = axes[1]
        ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-',
                 label='Train Acc', linewidth=2)
        ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r-',
                 label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved training curves to: {save_path}")

        plt.show()

    except ImportError:
        print("  [Warning] matplotlib not available, skipping plots")


def plot_confusion_matrix(logits: np.ndarray, labels: np.ndarray,
                          save_path: Optional[str] = None):
    """Plot confusion matrix."""
    try:
        import matplotlib.pyplot as plt

        predictions = np.argmax(logits, axis=1)
        num_classes = 10

        # Compute confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for pred, true in zip(predictions, labels):
            cm[true, pred] += 1

        # Normalize
        cm_normalized = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(num_classes),
               yticks=np.arange(num_classes),
               xticklabels=list(range(10)),
               yticklabels=list(range(10)),
               xlabel='Predicted Label',
               ylabel='True Label')

        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha='center', va='center',
                       color='white' if cm_normalized[i, j] > thresh else 'black',
                       fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved confusion matrix to: {save_path}")

        plt.show()

    except ImportError:
        print("  [Warning] matplotlib not available, skipping plots")


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
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    hidden_sizes: List[int] = [512, 256],
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
    print(f"  Momentum:      {momentum}")
    print(f"  Weight decay:  {weight_decay}")
    print(f"  Hidden sizes:  {hidden_sizes}")
    print(f"  Val split:     {val_split}")
    print(f"  Random seed:   {seed}")

    # ==========================================================================
    # Data Loading
    # ==========================================================================
    print_header("Loading Data", '-')

    data = load_mnist()
    train_images, train_labels, val_images, val_labels = split_validation(
        data['train_images'], data['train_labels'], val_split
    )
    test_images, test_labels = data['test_images'], data['test_labels']

    print(f"  Training samples:   {len(train_images):,}")
    print(f"  Validation samples: {len(val_images):,}")
    print(f"  Test samples:       {len(test_images):,}")
    print(f"  Input shape:        (batch, 784)")
    print(f"  Num classes:        10")

    # Create data loaders
    train_loader = DataLoader(train_images, train_labels, batch_size, shuffle=True)
    val_loader = DataLoader(val_images, val_labels, batch_size, shuffle=False)
    test_loader = DataLoader(test_images, test_labels, batch_size, shuffle=False)

    # ==========================================================================
    # Model Setup
    # ==========================================================================
    print_header("Model Architecture", '-')

    model = MLP(input_size=784, hidden_sizes=hidden_sizes, num_classes=10)

    # Count parameters
    total_params = 0
    for name, param in model.named_parameters():
        param_count = np.prod(param.data.shape)
        total_params += param_count
        print(f"  {name}: {param.data.shape} ({param_count:,} params)")
    print(f"  {'─' * 40}")
    print(f"  Total parameters: {total_params:,}")

    # ==========================================================================
    # Optimizer and Loss
    # ==========================================================================
    print_header("Training Setup", '-')

    # Get parameters from model
    params = list(model.parameters())
    optimizer = SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    print(f"  Optimizer:  SGD (lr={learning_rate}, momentum={momentum})")
    print(f"  Loss:       CrossEntropyLoss")
    print(f"  Batches per epoch: {len(train_loader)}")

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print_header("Training", '-')

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        print(f"\n  Epoch {epoch}/{epochs}")
        print(f"  {'─' * 50}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, log_interval
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_marker = " ★ New Best!"
        else:
            best_marker = ""

        epoch_time = time.time() - epoch_start

        print(f"\n  Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"           Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%{best_marker}")
        print(f"           Time: {format_time(epoch_time)}")

    total_time = time.time() - start_time

    # ==========================================================================
    # Test Evaluation
    # ==========================================================================
    print_header("Test Evaluation", '-')

    test_loss, test_acc, test_logits, test_labels_arr = evaluate(
        model, test_loader, criterion
    )

    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")

    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(test_logits, test_labels_arr)

    print(f"\n  Per-Class Accuracy:")
    for digit in range(10):
        bar_len = int(per_class_acc[digit] * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        print(f"    Digit {digit}: |{bar}| {per_class_acc[digit]*100:.1f}%")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print_header("Training Complete!", '=')

    print(f"  Total training time: {format_time(total_time)}")
    print(f"  Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Final test accuracy: {test_acc * 100:.2f}%")
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

        plot_confusion_matrix(
            test_logits, test_labels_arr,
            save_path=str(output_path / 'confusion_matrix.png')
        )

        plot_example_predictions(
            test_images, test_labels_arr, test_logits,
            save_path=str(output_path / 'example_predictions.png')
        )
    else:
        plot_training_history(history)
        plot_confusion_matrix(test_logits, test_labels_arr)
        plot_example_predictions(test_images, test_labels_arr, test_logits)

    print_header("Done!", '=')

    return {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_logits': test_logits,
        'test_labels': test_labels_arr
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
        momentum=0.9,
        weight_decay=1e-4,
        hidden_sizes=[512, 256],
        log_interval=100,
        val_split=0.1,
        seed=42,
        save_plots=True,
        output_dir='./outputs/mnist_mlp'
    )
