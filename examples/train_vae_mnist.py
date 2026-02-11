#!/usr/bin/env python3
"""
Example: Train VAE on MNIST

This script demonstrates training a Variational Autoencoder
for generative modeling of MNIST digits.

Usage:
    python examples/train_vae_mnist.py

Prerequisites to implement first:
    - generative/autoencoders/vae.py (VAE class)
    - foundations/autograd.py
    - nn_core/layers/linear.py

External dependencies:
    pip install matplotlib scikit-learn
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from python.generative.autoencoders.vae import VAE


def load_mnist():
    """Load and preprocess MNIST."""
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data.astype(np.float32) / 255.0
        return X[:60000], X[60000:]
    except ImportError:
        print("Using synthetic data (install sklearn for real MNIST)")
        X_train = np.random.rand(10000, 784).astype(np.float32)
        X_test = np.random.rand(1000, 784).astype(np.float32)
        return X_train, X_test


def train_vae(
    model,
    X_train,
    n_epochs: int = 20,
    batch_size: int = 128,
    verbose: bool = True
):
    """Train VAE."""
    n_samples = len(X_train)
    n_batches = n_samples // batch_size

    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(n_epochs):
        # Shuffle data
        perm = np.random.permutation(n_samples)
        X_shuffled = X_train[perm]

        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]

            # Forward pass
            loss, recon_loss, kl_loss = model.train_step(X_batch)

            epoch_loss += loss
            epoch_recon += recon_loss
            epoch_kl += kl_loss

        # Average losses
        epoch_loss /= n_batches
        epoch_recon /= n_batches
        epoch_kl /= n_batches

        history['loss'].append(epoch_loss)
        history['recon_loss'].append(epoch_recon)
        history['kl_loss'].append(epoch_kl)

        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Recon: {epoch_recon:.4f} | "
                  f"KL: {epoch_kl:.4f}")

    return history


def visualize_results(model, X_test, n_samples: int = 10):
    """Visualize reconstructions and samples."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, n_samples, figsize=(15, 6))

        # Row 1: Original images
        for i in range(n_samples):
            axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')

        # Row 2: Reconstructions
        recons = model.reconstruct(X_test[:n_samples])
        for i in range(n_samples):
            axes[1, i].imshow(recons[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstruction')

        # Row 3: Random samples
        samples = model.sample(n_samples)
        for i in range(n_samples):
            axes[2, i].imshow(samples[i].reshape(28, 28), cmap='gray')
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('Random Sample')

        # Row 4: Interpolations
        z1 = model.encode(X_test[0:1])[0]  # Get mean
        z2 = model.encode(X_test[1:2])[0]
        alphas = np.linspace(0, 1, n_samples)
        for i, alpha in enumerate(alphas):
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = model.decode(z_interp)
            axes[3, i].imshow(img.reshape(28, 28), cmap='gray')
            axes[3, i].axis('off')
            if i == 0:
                axes[3, i].set_title('Interpolation')

        plt.tight_layout()
        plt.savefig('vae_results.png')
        print("\nVisualization saved to vae_results.png")

    except ImportError:
        print("\nInstall matplotlib for visualization")


def main():
    print("=" * 50)
    print("VAE on MNIST")
    print("=" * 50)

    # Load data
    X_train, X_test = load_mnist()
    print(f"\nData: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Create VAE
    model = VAE(
        input_dim=784,
        hidden_dims=[512, 256],
        latent_dim=32,
        learning_rate=1e-3
    )

    print(f"\nModel: VAE")
    print(f"  Encoder: 784 -> 512 -> 256 -> 32 (latent)")
    print(f"  Decoder: 32 -> 256 -> 512 -> 784")

    # Train
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    history = train_vae(model, X_train, n_epochs=20)

    # Evaluate
    print("\n" + "=" * 50)
    print("Evaluation")
    print("=" * 50)

    test_loss, test_recon, test_kl = model.evaluate(X_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"  Reconstruction: {test_recon:.4f}")
    print(f"  KL Divergence: {test_kl:.4f}")

    # Visualize
    visualize_results(model, X_test)

    # Plot training curves
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['loss'], label='Total Loss')
        ax.plot(history['recon_loss'], label='Reconstruction')
        ax.plot(history['kl_loss'], label='KL Divergence')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('VAE Training')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('vae_training.png')
        print("Training curve saved to vae_training.png")

    except ImportError:
        pass


if __name__ == "__main__":
    main()
