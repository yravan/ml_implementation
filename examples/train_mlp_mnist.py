#!/usr/bin/env python3
"""
Example: Train MLP on MNIST

This script demonstrates training a multi-layer perceptron
on MNIST digit classification using only NumPy.

Usage:
    python examples/train_mlp_mnist.py

Prerequisites to implement first:
    - foundations/autograd.py (Variable, grad)
    - nn_core/layers/linear.py (Linear)
    - nn_core/activations/relu.py (ReLU)
    - nn_core/activations/softmax.py (Softmax)
    - optimization/losses/cross_entropy.py (CrossEntropyLoss)
    - optimization/optimizers/adam.py (Adam)
    - architectures/mlp.py (MLP)
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from python.architectures.mlp import MLP
from python.optimization.losses.cross_entropy import CrossEntropyLoss
from python.optimization.optimizers.adam import Adam


def load_mnist():
    """
    Load MNIST dataset.

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Try to load from sklearn
    try:
        from sklearn.datasets import fetch_openml

        print("Loading MNIST from sklearn...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int32)

        # Normalize to [0, 1]
        X = X / 255.0

        # Split
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        return X_train, y_train, X_test, y_test

    except ImportError:
        print("sklearn not found. Generating synthetic data...")
        return generate_synthetic_data()


def generate_synthetic_data(n_train=10000, n_test=1000):
    """Generate synthetic digit-like data for testing."""
    np.random.seed(42)

    n_classes = 10
    n_features = 784  # 28x28

    X_train = np.random.randn(n_train, n_features).astype(np.float32) * 0.3
    y_train = np.random.randint(0, n_classes, n_train).astype(np.int32)

    X_test = np.random.randn(n_test, n_features).astype(np.float32) * 0.3
    y_test = np.random.randint(0, n_classes, n_test).astype(np.int32)

    # Add some class-dependent signal
    for i in range(n_classes):
        mask_train = y_train == i
        mask_test = y_test == i
        signal = np.zeros(n_features)
        signal[i * 78:(i + 1) * 78] = 1.0
        X_train[mask_train] += signal
        X_test[mask_test] += signal

    return X_train, y_train, X_test, y_test


def batch_iterator(X, y, batch_size, shuffle=True):
    """Iterate over mini-batches."""
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def compute_accuracy(model, X, y, batch_size=256):
    """Compute accuracy on dataset."""
    correct = 0
    total = 0

    for X_batch, y_batch in batch_iterator(X, y, batch_size, shuffle=False):
        logits = model.forward(X_batch)
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == y_batch)
        total += len(y_batch)

    return correct / total


def train_epoch(model, optimizer, loss_fn, X, y, batch_size):
    """Train for one epoch."""
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in batch_iterator(X, y, batch_size):
        # Forward pass
        logits = model.forward(X_batch)
        loss = loss_fn.forward(logits, y_batch)

        # Backward pass
        grad_logits = loss_fn.backward()
        model.backward(grad_logits)

        # Update weights
        optimizer.step(model.parameters(), model.gradients())

        # Zero gradients
        model.zero_grad()

        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


def main():
    print("=" * 50)
    print("MLP on MNIST")
    print("=" * 50)

    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")

    # Create model
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    model = MLP(
        input_dim=input_dim,
        hidden_dims=[256, 128],
        output_dim=n_classes,
        activation='relu',
        dropout_rate=0.2
    )

    print(f"\nModel: MLP")
    print(f"  Architecture: {input_dim} -> 256 -> 128 -> {n_classes}")
    print(f"  Total parameters: {model.count_parameters():,}")

    # Loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(learning_rate=1e-3)

    # Training settings
    n_epochs = 10
    batch_size = 64

    print(f"\nTraining:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Optimizer: Adam (lr=1e-3)")

    # Train
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    best_acc = 0
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(
            model, optimizer, loss_fn,
            X_train, y_train, batch_size
        )

        # Evaluate
        train_acc = compute_accuracy(model, X_train, y_train)
        test_acc = compute_accuracy(model, X_test, y_test)

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch {epoch + 1}/{n_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

    # Final results
    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Best test accuracy: {best_acc:.4f}")

    # Show some predictions
    print("\nSample predictions:")
    sample_idx = np.random.choice(len(X_test), 5, replace=False)
    logits = model.forward(X_test[sample_idx])
    preds = np.argmax(logits, axis=1)

    for i, idx in enumerate(sample_idx):
        print(f"  Sample {i+1}: Predicted={preds[i]}, Actual={y_test[idx]}")


if __name__ == "__main__":
    main()
