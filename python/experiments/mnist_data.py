import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from pathlib import Path
import urllib.request
import gzip
import struct

from python.utils.data_utils import Dataset
from python.vision import transforms


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
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
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

def load_mnist_datasets(val_split: float = 0.1, transform: Optional[Callable] = transforms.ToTensor()) :
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

    train_dataset = MNistDataset(train_images, train_labels, transform=transform)
    val_dataset = MNistDataset(val_images, val_labels, transform=transform)
    test_dataset = MNistDataset(test_images, test_labels, transform=transform)

    return train_dataset, val_dataset, test_dataset


class MNistDataset(Dataset):
    """
    MNIST dataset loader.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Optional[Callable] = None):
        """
        Args:
            root: Root directory of ImageNet (contains train/ and val/)
            train: If True, use training set; else validation set
            subset: If set, only use this many samples (for debugging)
        """
        self.images = images
        self.labels = labels
        self.transform = transform


    def load_sample(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load a single sample by index."""
        img, label = self.images[idx], self.labels[idx]
        if self.transform is not None:
            return self.transform(img), label
        else:
            return img, label


