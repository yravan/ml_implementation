"""
Data Utilities
==============

Utilities for data loading, batching, splitting, and preprocessing.

Theory
------
Efficient data handling is crucial for training neural networks:

1. **Batching**: Neural networks process data in batches for efficiency and
   regularization (batch gradient has lower variance than single-sample).

2. **Shuffling**: Random shuffling each epoch prevents the model from learning
   spurious patterns based on data order.

3. **Train/Val/Test Split**: Proper splitting prevents data leakage and enables
   unbiased evaluation. Common splits: 80/10/10 or 70/15/15.

4. **Stratified Splitting**: For classification, maintain class proportions in
   all splits to avoid biased evaluation.

5. **Mini-batch Gradient Descent**: Balance between:
   - Batch GD (full dataset): Stable but slow, needs lots of memory
   - SGD (single sample): Noisy but fast, good regularization
   - Mini-batch (32-512 samples): Best of both worlds

Math
----
# Mini-batch gradient estimate:
# g_batch ≈ (1/|B|) * Σ_{i∈B} ∇L(x_i, θ)
#
# Variance of gradient estimate:
# Var(g_batch) = Var(g_single) / |B|
# Larger batches → lower variance → smoother optimization

# Stratified split for class k:
# |train_k| / |train| = |val_k| / |val| = |test_k| / |test| = |class_k| / |total|

References
----------
- CS231n: Data and Preprocessing
  https://cs231n.github.io/neural-networks-2/#datapre
- "Practical Recommendations for Gradient-Based Training" - Bengio
  https://arxiv.org/abs/1206.5533
- PyTorch DataLoader documentation
  https://pytorch.org/docs/stable/data.html

Implementation Notes
--------------------
- Always shuffle training data (but not validation/test)
- Drop last incomplete batch during training if batch norm is used
- Normalize features based on training set statistics only
- Be careful with time series: no future data in training
"""
from abc import abstractmethod
from pathlib import Path

# Implementation Status: NOT STARTED
# Complexity: Easy
# Prerequisites: seeding (for reproducible shuffling)

import numpy as np
from typing import Iterator, Tuple, List, Optional, Union, Callable

from python.utils.seeding import set_seed


def batch_iterator(X: np.ndarray, y: Optional[np.ndarray] = None,
                   batch_size: int = 32, shuffle: bool = True,
                   drop_last: bool = False,
                   seed: Optional[int] = None) -> Iterator[Tuple[np.ndarray, ...]]:
    """
    Iterate over data in mini-batches.

    This is the core data loading primitive. Yields batches of (X,) or (X, y)
    depending on whether labels are provided.

    Args:
        X: Features array of shape (n_samples, ...)
        y: Optional labels array of shape (n_samples,) or (n_samples, ...)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data before iterating
        drop_last: If True, drop last batch if incomplete
        seed: Random seed for shuffling

    Yields:
        Tuple of (X_batch,) or (X_batch, y_batch)

    Example:
        >>> X = np.arange(100).reshape(10, 10)
        >>> y = np.arange(10)
        >>> for X_batch, y_batch in batch_iterator(X, y, batch_size=3):
        ...     print(X_batch.shape, y_batch.shape)
        (3, 10) (3,)
        (3, 10) (3,)
        (3, 10) (3,)
        (1, 10) (1,)  # Last batch (unless drop_last=True)

    Training Loop Pattern:
        for epoch in range(num_epochs):
            for X_batch, y_batch in batch_iterator(X_train, y_train, shuffle=True):
                loss, grads = compute_loss_and_grads(model, X_batch, y_batch)
                optimizer.step(grads)
    """
    rng = np.random.default_rng(seed)
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    if shuffle:
        indices = np.arange(num_samples)
        rng.shuffle(indices)
        X = X[indices]
        if y is not None:
            y = y[indices]
    for batch in range(num_batches):
        start = batch * batch_size
        if y is not None:
            yield X[start:start + batch_size], y[start:start + batch_size]
        else:
            yield X[start:start + batch_size],
    if not drop_last:
        if y is not None:
            yield X[num_batches * batch_size:], y[num_batches * batch_size:]
        else:
            yield X[num_batches * batch_size:],


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     shuffle: bool = True,
                     stratify: bool = False,
                     seed: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Args:
        X: Features array of shape (n_samples, ...)
        y: Labels array of shape (n_samples,)
        test_size: Fraction of data for test set (0 < test_size < 1)
        shuffle: Whether to shuffle before splitting
        stratify: If True, maintain class proportions in splits
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Example:
        >>> X = np.arange(100).reshape(50, 2)
        >>> y = np.array([0]*25 + [1]*25)
        >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=42)
        >>> len(X_tr), len(X_te)
        (40, 10)

    With stratification:
        >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=True)
        >>> np.mean(y_tr == 0), np.mean(y_te == 0)  # Both ≈ 0.5
        (0.5, 0.5)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        if stratify:
            _, counts = np.unique(y, return_counts=True)
            test_sizes = (counts * test_size).astype(int)

            class_start_indices = np.append([0], np.cumsum(counts))[:-1]
            class_position_indices = np.arange(num_samples) - np.repeat(class_start_indices, counts)

            test_mask = class_position_indices < np.repeat(test_sizes, counts)

            random_vals = rng.random(num_samples)
            sorted_order = np.lexsort((random_vals, y))
            test_indices = sorted_order[test_mask]
            train_indices = sorted_order[~test_mask]
        else:
            rng.shuffle(indices)
            test_indices = indices[:int(num_samples * test_size)]
            train_indices = indices[int(num_samples * test_size):]
    else:
        test_indices = indices[:int(num_samples * test_size)]
        train_indices = indices[int(num_samples * test_size):]
    return X[train_indices, :], X[test_indices, :], y[train_indices, :], y[test_indices, :]


def train_val_test_split(X: np.ndarray, y: np.ndarray,
                         val_size: float = 0.1,
                         test_size: float = 0.1,
                         shuffle: bool = True,
                         stratify: bool = False,
                         seed: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets.

    Validation set is used for hyperparameter tuning and early stopping.
    Test set is only used for final evaluation.

    Args:
        X: Features array
        y: Labels array
        val_size: Fraction for validation
        test_size: Fraction for test
        shuffle: Whether to shuffle
        stratify: Maintain class proportions
        seed: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    Example:
        >>> X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(
        ...     X, y, val_size=0.1, test_size=0.1, seed=42
        ... )
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle,stratify=stratify, seed=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(val_size) / (1 - test_size), shuffle=shuffle,stratify=stratify, seed=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def k_fold_split(n_samples: int, k: int = 5, shuffle: bool = True,
                 seed: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate indices for k-fold cross-validation.

    K-fold CV provides more robust evaluation than a single train/test split
    by training and evaluating on all data.

    Args:
        n_samples: Total number of samples
        k: Number of folds
        shuffle: Whether to shuffle before splitting
        seed: Random seed

    Yields:
        Tuple of (train_indices, val_indices) for each fold

    Example:
        >>> for fold, (train_idx, val_idx) in enumerate(k_fold_split(100, k=5)):
        ...     print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
        Fold 0: train=80, val=20
        Fold 1: train=80, val=20
        ...
        Fold 4: train=80, val=20

    Cross-Validation Pattern:
        scores = []
        for train_idx, val_idx in k_fold_split(len(X), k=5):
            model = train(X[train_idx], y[train_idx])
            score = evaluate(model, X[val_idx], y[val_idx])
            scores.append(score)
        print(f"CV Score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    """
    indices = np.arange(n_samples)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        yield train_idx, val_idx
        current += fold_size


def normalize(X: np.ndarray, axis: int = 0,
              mean: Optional[np.ndarray] = None,
              std: Optional[np.ndarray] = None,
              eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize data: (X - mean) / std.

    Normalization helps training by:
    - Putting all features on the same scale
    - Centering data around 0 (better for gradient descent)
    - Preventing features with large values from dominating

    Args:
        X: Data array
        axis: Axis along which to compute statistics
        mean: Pre-computed mean (use training set mean for test data)
        std: Pre-computed std (use training set std for test data)
        eps: Small constant to avoid division by zero

    Returns:
        Tuple of (normalized_X, mean, std)

    Example:
        >>> X_train = np.random.randn(100, 10) * 5 + 3  # Mean≈3, std≈5
        >>> X_train_norm, mean, std = normalize(X_train)
        >>> # For test data, use training statistics:
        >>> X_test_norm, _, _ = normalize(X_test, mean=mean, std=std)

    Important:
        ALWAYS compute mean/std on training data only!
        Using test data statistics causes data leakage.
    """
    if mean is None:
        mean = np.mean(X, axis=axis, keepdims=True)
    else:
        mean = np.expand_dims(mean, axis=axis)
    if std is None:
        std = np.std(X, axis=axis, keepdims=True)
    else:
        std = np.expand_dims(std, axis=axis)
    return (X - mean) / (std + eps), np.squeeze(mean, axis=axis), np.squeeze(std, axis=axis)


def to_categorical(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded vectors.

    Args:
        y: Integer labels of shape (n_samples,)
        num_classes: Total number of classes (inferred if None)

    Returns:
        One-hot encoded array of shape (n_samples, num_classes)

    Example:
        >>> y = np.array([0, 1, 2, 1, 0])
        >>> to_categorical(y)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.],
               [0., 1., 0.],
               [1., 0., 0.]])
    """
    if num_classes is None:
        num_classes = y.max() + 1
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


class Dataset:
    """
    General dataset loader.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        """
        Args:
            root: Root directory of ImageNet (contains train/ and val/)
            train: If True, use training set; else validation set
            subset: If set, only use this many samples (for debugging)
        """
        self.root = Path(root)
        self.train = train
        self.samples = []
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    @abstractmethod
    def load_sample(self, idx: int):
        """Load a single sample by index."""
        raise NotImplementedError


class DataLoader:
    """
    A simple data loader class for batch iteration.

    More feature-rich than batch_iterator, supports:
    - Stateful iteration (can track position)
    - Length computation
    - Reset for new epochs

    Attributes:
        X: Features array
        y: Labels array (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle each epoch

    Example:
        >>> loader = DataLoader(X_train, y_train, batch_size=32, shuffle=True)
        >>> for epoch in range(10):
        ...     for X_batch, y_batch in loader:
        ...         # Training step
        ...         pass
        >>> len(loader)  # Number of batches
        32
    """

    def __init__(self, dataset: Dataset,
                 batch_size: int = 32, shuffle: bool = True,
                 drop_last: bool = False, seed: Optional[int] = None, num_workers: int = 10,):
        """
        Initialize DataLoader.

        Args:
            X: Features array
            y: Labels array (optional)
            batch_size: Samples per batch
            shuffle: Shuffle each epoch
            drop_last: Drop incomplete final batch
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.n_samples = len(dataset)
        self._rng = np.random.default_rng(seed)
        self._indices = np.arange(self.n_samples)
        self._current = 0
        self.num_workers = num_workers

    def __iter__(self) -> "DataLoader":
        """Reset and return iterator."""
        self._current = 0
        if self.shuffle:
            self._rng.shuffle(self._indices)
        return self

    def __next__(self):
        """Return next batch."""
        if self._current >= self.n_samples:
            raise StopIteration

        end = min(self._current + self.batch_size, self.n_samples)
        batch_indices = self._indices[self._current:end]

        # Check if this is an incomplete last batch
        if len(batch_indices) < self.batch_size and self.drop_last:
            raise StopIteration

        self._current = end

        data_samples = []
        for idx in batch_indices:
            try:
                data = self.dataset.load_sample(idx)
                data_samples.append(data)
            except Exception as e:
                continue
        num_data_types = len(data_samples[0])
        batched_data = []
        for i in range(num_data_types):
            batched_data.append(
                np.stack(tuple(sample[i] for sample in data_samples), axis=0)
            )

        return tuple(batched_data)

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size


def normalize_data(X: np.ndarray, axis: int = 0) -> tuple:
    """
    Normalize data to zero mean and unit variance.

    Args:
        X: Data array
        axis: Axis along which to normalize

    Returns:
        Tuple of (normalized_data, mean, std)
    """
    raise NotImplementedError(
        "TODO: Implement data normalization\n"
        "Hint: mean = X.mean(axis); std = X.std(axis)\n"
        "      return (X - mean) / std, mean, std"
    )


def one_hot_encode(labels: np.ndarray, num_classes: int = None) -> np.ndarray:
    """
    One-hot encode class labels.

    Args:
        labels: Integer class labels
        num_classes: Number of classes (inferred if None)

    Returns:
        One-hot encoded array
    """
    raise NotImplementedError(
        "TODO: Implement one-hot encoding\n"
        "Hint: Use np.eye(num_classes)[labels]"
    )
