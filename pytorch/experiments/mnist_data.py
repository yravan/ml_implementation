"""
MNIST Data Loading — PyTorch version.

Uses torchvision for transforms, torch DataLoader for batching.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict
from pathlib import Path
import urllib.request
import gzip
import struct

import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Raw MNIST Loading (shared between frameworks)
# =============================================================================

def download_mnist(data_dir: str = '../../data') -> None:
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
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images.astype(np.float32) / 255.0


def load_mnist_labels(filepath: str) -> np.ndarray:
    """Load MNIST labels from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_mnist(data_dir: str = '../../data') -> Dict[str, np.ndarray]:
    """Load full MNIST dataset as numpy arrays."""
    download_mnist(data_dir)
    data_path = Path(data_dir)

    return {
        'train_images': load_mnist_images(data_path / 'train-images-idx3-ubyte.gz'),
        'train_labels': load_mnist_labels(data_path / 'train-labels-idx1-ubyte.gz'),
        'test_images': load_mnist_images(data_path / 't10k-images-idx3-ubyte.gz'),
        'test_labels': load_mnist_labels(data_path / 't10k-labels-idx1-ubyte.gz'),
    }


def split_validation(images: np.ndarray, labels: np.ndarray,
                     val_split: float = 0.1) -> Tuple[np.ndarray, ...]:
    """Split into train and validation sets."""
    n = len(images)
    n_val = int(n * val_split)
    indices = np.random.permutation(n)
    return (
        images[indices[n_val:]],
        labels[indices[n_val:]],
        images[indices[:n_val]],
        labels[indices[:n_val]],
    )


# =============================================================================
# PyTorch Dataset
# =============================================================================

class MNISTDataset(Dataset):
    """PyTorch Dataset for MNIST."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 channels: bool = True):
        """
        Args:
            images: (N, 28, 28) float32 numpy array
            labels: (N,) int64 numpy array
            channels: If True, add channel dim → (N, 1, 28, 28) for CNNs
        """
        if channels and images.ndim == 3:
            images = images[:, np.newaxis, :, :]  # (N, 1, 28, 28)
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    # Compatibility with custom DataLoader
    def load_sample(self, idx):
        return self.images[idx].numpy(), self.labels[idx].item()


def load_mnist_datasets(val_split: float = 0.1, batch_size: int = 64,
                        channels: bool = True, num_workers: int = 0):
    """
    Load MNIST and return PyTorch DataLoaders.

    Args:
        val_split: Fraction for validation
        batch_size: Batch size
        channels: Add channel dim for CNNs
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data = load_mnist()
    train_images, train_labels, val_images, val_labels = split_validation(
        data['train_images'], data['train_labels'], val_split
    )
    test_images, test_labels = data['test_images'], data['test_labels']

    print(f"  Training samples:   {len(train_images):,}")
    print(f"  Validation samples: {len(val_images):,}")
    print(f"  Test samples:       {len(test_images):,}")

    train_ds = MNISTDataset(train_images, train_labels, channels=channels)
    val_ds = MNISTDataset(val_images, val_labels, channels=channels)
    test_ds = MNISTDataset(test_images, test_labels, channels=channels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader