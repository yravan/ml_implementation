"""
ImageNet Training Experiment
============================

Train a ResNet on ImageNet (ILSVRC2012) using our custom
deep learning framework with numpy-based autograd.

ImageNet Dataset:
    - ~1.2M training images, ~50K validation images
    - 1000 classes
    - Variable resolution, resized to 224x224

Prerequisites:
    - Download ImageNet from https://image-net.org (requires account)
    - Extract to directory structure:
        data/imagenet/
            train/
                n01440764/
                    *.JPEG
                n01443537/
                    *.JPEG
                ...
            val/
                n01440764/
                    *.JPEG
                ...

    - Or use the included script to reorganize val if needed.

Usage:
    python -m python.experiments.imagenet_training

    # Quick sanity check with subset
    python -m python.experiments.imagenet_training --subset 5000

    # Full training
    python -m python.experiments.imagenet_training --epochs 90
"""

import numpy as np
import time
import os
import json
from typing import Tuple, List, Dict, Optional, Callable
from pathlib import Path

from python.utils.data_utils import Dataset
# Our framework imports
from torchvision import transforms

# =============================================================================
# ImageNet Constants
# =============================================================================

# ImageNet channel-wise mean and std (RGB, 0-1 scale)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Standard image size
IMAGE_SIZE = 224


# =============================================================================
# Image Loading Utilities
# =============================================================================


def download_imagenet(data_dir: str = '/Users/yajvanravan/mit/ml_implementation/data', variant: str = 'imagenette',
                      size: str = '320') -> str:
    """
    Download/prepare ImageNet or ImageNette dataset.

    Args:
        data_dir: Root directory for data storage
        variant: 'imagenette' (auto-download, 10 classes) or
                 'imagenet' (manual download required, 1000 classes)
        size: For imagenette only - 'full', '320', or '160'

    Returns:
        Path to dataset root (contains train/ and val/)
    """
    import urllib.request
    import tarfile

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # ImageNette (auto-download)
    # =================================================================
    if variant == 'imagenette':
        urls = {
            'full': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz',
            '320': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
            '160': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
        }
        folder_names = {
            'full': 'imagenette2',
            '320': 'imagenette2-320',
            '160': 'imagenette2-160',
        }

        if size not in urls:
            raise ValueError(f"Invalid size '{size}'. Choose from: {list(urls.keys())}")

        url = urls[size]
        folder_name = folder_names[size]
        dataset_dir = data_path / folder_name
        tar_file = data_path / f'{folder_name}.tgz'

        if dataset_dir.exists() and (dataset_dir / 'train').exists():
            print(f"  ImageNette already exists at {dataset_dir}")
            return str(dataset_dir)

        if not tar_file.exists():
            print(f"  Downloading ImageNette ({size})...")
            print(f"  URL: {url}")

            def progress_hook(count, block_size, total_size):
                pct = count * block_size * 100 / total_size
                mb = count * block_size / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)", end='', flush=True)

            urllib.request.urlretrieve(url, tar_file, reporthook=progress_hook)
            print(f"\n  Downloaded to {tar_file}")

        print("  Extracting...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(data_path)
        print(f"  Extracted to {dataset_dir}")

        tar_file.unlink()
        print(f"  Removed {tar_file}")

        _verify_dataset(dataset_dir)
        return str(dataset_dir)

    # =================================================================
    # Full ImageNet (manual download, auto-extract)
    # =================================================================
    elif variant == 'imagenet':
        dataset_dir = data_path / 'imagenet' / 'ILSVRC' / 'Data' / 'CLS-LOC'
        train_dir = dataset_dir / 'train'
        val_dir = dataset_dir / 'val'
        test_dir = dataset_dir / 'test'
        if test_dir.exists():
            flat_images = list(test_dir.glob("*.JPEG"))
            if len(flat_images) > 0:
                csv_path = _find_file(data_path / "imagenet", ["LOC_test_solution.csv"])
                _reorganize_test(test_dir, csv_path=csv_path)

        # Check if already fully set up
        if train_dir.exists() and val_dir.exists():
            n_train_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
            n_val_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
            if n_train_classes >= 1000 and n_val_classes >= 1000:
                print(f"  ImageNet already exists at {dataset_dir} ({n_train_classes} train classes, {n_val_classes} val classes)")
                _verify_dataset(dataset_dir)
                return str(dataset_dir)

        # Check if Kaggle zip was extracted (train exists but val is flat)
        if train_dir.exists():
            n_train_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
            if n_train_classes >= 1000:
                print(f"  Found Kaggle-extracted training set: {n_train_classes} classes")

                # Check if val needs reorganizing
                flat_images = list(val_dir.glob('*.JPEG'))
                if len(flat_images) > 0:
                    print(f"  Val directory is flat ({len(flat_images)} images). Reorganizing...")
                    csv_path = _find_file(data_path / 'imagenet', ['LOC_val_solution.csv'])
                    _reorganize_val(val_dir, csv_path=csv_path)

                _verify_dataset(dataset_dir)
                return str(dataset_dir)

        # Check if already extracted
        if (dataset_dir / 'train').exists() and (dataset_dir / 'val').exists():
            n_train_classes = len([d for d in (dataset_dir / 'train').iterdir() if d.is_dir()])
            if n_train_classes >= 1000:
                print(f"  ImageNet already exists at {dataset_dir} ({n_train_classes} classes)")
                return str(dataset_dir)

        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Look for downloaded tar files
        train_tar = _find_file(data_path, ['ILSVRC2012_img_train.tar'])
        val_tar = _find_file(data_path, ['ILSVRC2012_img_val.tar'])
        devkit_tar = _find_file(data_path, ['ILSVRC2012_devkit_t12.tar.gz',
                                             'ILSVRC2012_devkit_t12.tar'])

        if train_tar is None or val_tar is None:
            print("=" * 70)
            print("  Full ImageNet requires manual download.")
            print()
            print("  1. Register at https://image-net.org/download-images.php")
            print("  2. Download these files:")
            print(f"     - ILSVRC2012_img_train.tar (~138 GB)")
            print(f"     - ILSVRC2012_img_val.tar (~6.3 GB)")
            print(f"     - ILSVRC2012_devkit_t12.tar.gz (optional, for labels)")
            print(f"  3. Place them in: {data_path}")
            print(f"  4. Run this again.")
            print()
            print("  For quick testing, use variant='imagenette' instead.")
            print("=" * 70)
            raise FileNotFoundError(
                f"ImageNet tar files not found in {data_path}. "
                f"See above for download instructions."
            )

        # Extract training set
        train_dir = dataset_dir / 'train'
        if not train_dir.exists() or len(list(train_dir.iterdir())) < 1000:
            print(f"  Extracting training set from {train_tar}...")
            print(f"  This will take a while (~138 GB)...")
            train_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(train_tar, 'r:') as tar:
                tar.extractall(train_dir)

            # Each class is a nested tar: n01440764.tar, etc.
            print("  Extracting per-class tar files...")
            class_tars = sorted(train_dir.glob('*.tar'))
            for i, class_tar in enumerate(class_tars):
                class_name = class_tar.stem
                class_dir = train_dir / class_name
                class_dir.mkdir(exist_ok=True)

                with tarfile.open(class_tar, 'r:') as tar:
                    tar.extractall(class_dir)
                class_tar.unlink()

                if (i + 1) % 100 == 0:
                    print(f"    {i + 1}/{len(class_tars)} classes extracted")

            print(f"  Training set extracted: {len(class_tars)} classes")

        # Extract validation set
        val_dir = dataset_dir / 'val'
        if not val_dir.exists() or not any(val_dir.iterdir()):
            print(f"  Extracting validation set from {val_tar}...")
            val_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(val_tar, 'r:') as tar:
                tar.extractall(val_dir)

            # Reorganize val into class folders
            _reorganize_val(val_dir, devkit_tar)

        _verify_dataset(dataset_dir)
        return str(dataset_dir)

    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose 'imagenette' or 'imagenet'.")


def _find_file(search_dir: Path, filenames: list) -> Optional[Path]:
    """Search for a file in directory and subdirectories."""
    for name in filenames:
        path = search_dir / name
        if path.exists():
            return path
        # Also check one level deeper
        for child in search_dir.iterdir():
            if child.is_dir():
                path = child / name
                if path.exists():
                    return path
    return None


def _reorganize_val(val_dir: Path, devkit_tar: Optional[Path] = None, csv_path: Optional[Path] = None):
    """
    Reorganize flat val directory into class subdirectories.
    Supports Kaggle CSV (LOC_val_solution.csv) or devkit tar.
    """
    val_images = sorted(val_dir.glob('*.JPEG'))
    if len(val_images) == 0:
        print("  No flat images found in val directory to reorganize")
        return

    synsets = None

    # Method 1: Kaggle CSV
    if csv_path is not None and csv_path.exists():
        import csv
        print(f"  Using {csv_path.name} for val labels...")
        synsets = {}
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                fname = row[0]  # ILSVRC2012_val_00000001
                synset = row[1].split()[0]  # n01751748
                synsets[fname] = synset

        for img_path in val_images:
            fname = img_path.stem  # ILSVRC2012_val_00000001
            if fname in synsets:
                class_dir = val_dir / synsets[fname]
                class_dir.mkdir(exist_ok=True)
                img_path.rename(class_dir / img_path.name)

        n_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
        print(f"  Reorganized {len(val_images)} images into {n_classes} class folders")
        return

    # Method 2: devkit tar
    if devkit_tar is not None:
        synset_list = _load_val_labels_from_devkit(devkit_tar)
        if synset_list is not None:
            for img_path, synset in zip(val_images, synset_list):
                class_dir = val_dir / synset.strip()
                class_dir.mkdir(exist_ok=True)
                img_path.rename(class_dir / img_path.name)
            n_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
            print(f"  Reorganized {len(val_images)} images into {n_classes} class folders")
            return

    # Method 3: download mapping
    print("  Downloading validation label mapping...")
    mapping_url = (
        "https://raw.githubusercontent.com/tensorflow/models/master/"
        "research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
    )
    try:
        import urllib.request
        response = urllib.request.urlopen(mapping_url)
        synset_list = response.read().decode().strip().split('\n')
    except Exception as e:
        print(f"  Warning: Could not download label mapping: {e}")
        return

    if len(synset_list) != len(val_images):
        print(f"  Warning: Label count ({len(synset_list)}) != image count ({len(val_images)})")
        return

    for img_path, synset in zip(val_images, synset_list):
        class_dir = val_dir / synset.strip()
        class_dir.mkdir(exist_ok=True)
        img_path.rename(class_dir / img_path.name)

    n_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
    print(f"  Reorganized {len(val_images)} images into {n_classes} class folders")


def _load_val_labels_from_devkit(devkit_tar: Path) -> Optional[list]:
    """Extract validation labels from the devkit tar."""
    import tarfile
    import io

    try:
        with tarfile.open(devkit_tar) as tar:
            # Read ground truth file
            gt_member = None
            for member in tar.getmembers():
                if 'ILSVRC2012_validation_ground_truth.txt' in member.name:
                    gt_member = member
                    break

            meta_member = None
            for member in tar.getmembers():
                if 'meta.mat' in member.name:
                    meta_member = member
                    break

            if gt_member is None:
                return None

            gt_file = tar.extractfile(gt_member)
            gt_labels = [int(line.strip()) for line in gt_file.readlines()]

            # Try to map class indices to synsets via meta.mat
            if meta_member is not None:
                try:
                    from scipy.io import loadmat
                    meta_file = tar.extractfile(meta_member)
                    meta = loadmat(io.BytesIO(meta_file.read()))
                    synsets_meta = meta['synsets']

                    # Build mapping: ILSVRC2012_ID -> synset string
                    id_to_synset = {}
                    for entry in synsets_meta:
                        ilsvrc_id = int(entry[0][0][0][0])
                        synset = str(entry[0][1][0])
                        id_to_synset[ilsvrc_id] = synset

                    synsets = [id_to_synset[label] for label in gt_labels]
                    return synsets
                except Exception:
                    pass

            return None

    except Exception:
        return None

def _reorganize_test(test_dir: Path, csv_path: Optional[Path] = None):
    """
    Reorganize flat test directory into class subdirectories if labels are available.
    Only possible with Kaggle's LOC_test_solution.csv (released post-competition).
    """
    flat_images = sorted(test_dir.glob('*.JPEG'))
    if len(flat_images) == 0:
        return False

    if csv_path is None or not csv_path.exists():
        print(f"  Test directory is flat ({len(flat_images)} images), no labels available.")
        print(f"  Provide LOC_test_solution.csv to reorganize into class folders.")
        return False

    import csv
    print(f"  Using {csv_path.name} for test labels...")
    synsets = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            fname = row[0]
            synset = row[1].split()[0]
            synsets[fname] = synset

    for img_path in flat_images:
        fname = img_path.stem
        if fname in synsets:
            class_dir = test_dir / synsets[fname]
            class_dir.mkdir(exist_ok=True)
            img_path.rename(class_dir / img_path.name)

    n_classes = len([d for d in test_dir.iterdir() if d.is_dir()])
    print(f"  Reorganized {len(flat_images)} images into {n_classes} class folders")
    return True

def _verify_dataset(dataset_dir: Path):
    """Print dataset statistics."""
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'

    if train_dir.exists():
        n_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
        n_images = sum(1 for _ in train_dir.rglob('*.JPEG'))
        print(f"  Train: {n_images:,} images, {n_classes} classes")

    if val_dir.exists():
        n_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
        n_images = sum(1 for _ in val_dir.rglob('*.JPEG'))
        print(f"  Val:   {n_images:,} images, {n_classes} classes")


# ImageNette class labels
IMAGENETTE_CLASSES = {
    'n01440764': 'tench',
    'n02102040': 'English springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute',
}

# =============================================================================
# Dataset Classes
# =============================================================================

class ImageNetDataset(Dataset):
    """
    ImageNet dataset loader.

    Expects standard torchvision-style directory structure:
        root/class_folder/image.JPEG

    Images are loaded on-the-fly (not cached in memory).
    """

    def __init__(self, root: str, train: bool = True, subset: Optional[int] = None, transform: Optional[Callable] = None):
        """
        Args:
            root: Root directory of ImageNet (contains train/ and val/)
            train: If True, use training set; else validation set
            subset: If set, only use this many samples (for debugging)
        """
        self.root = Path(root)
        self.train = train
        self.split_dir = self.root / ('train' if train else 'val')
        self.transform = transform

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"ImageNet directory not found: {self.split_dir}\n"
                f"Download ImageNet from https://image-net.org and extract to {root}"
            )

        # Discover classes (sorted for deterministic ordering)
        self.classes = sorted([
            d.name for d in self.split_dir.iterdir()
            if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Build file list
        self.samples = []  # List of (path, label)
        for cls_name in self.classes:
            cls_dir = self.split_dir / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in ('.jpeg', '.jpg', '.png'):
                    self.samples.append((str(img_path), cls_idx))

        # Optional subset for quick testing
        if subset is not None and subset < len(self.samples):
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), subset, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]

        print(f"  {'Train' if train else 'Val'} set: {len(self.samples):,} images, "
              f"{self.num_classes} classes")

        if train:
            print(f"  Num classes:      {self.num_classes}")
            print(f"  Image size:       {IMAGE_SIZE}x{IMAGE_SIZE}")
            print(f"  Normalization:    mean={IMAGENET_MEAN.flatten()}, std={IMAGENET_STD.flatten()}")
            print(f"  Train augmentation: RandomResizedCrop + HorizontalFlip")
            print(f"  Val preprocessing:  Resize + CenterCrop")


    def load_sample(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load a single sample by index."""
        path, label = self.samples[idx]
        img = transforms.load_image(path, size=IMAGE_SIZE)
        if self.transform is not None:
            return self.transform(img), label
        else:
            return img, label


# =============================================================================
# Class Label Utilities
# =============================================================================

def load_imagenet_labels(path: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Load human-readable ImageNet class labels.

    Args:
        path: Path to imagenet_class_index.json or LOC_synset_mapping.txt

    Returns:
        Dict mapping synset ID (e.g., 'n01440764') to human-readable name
    """
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        return None

    labels = {}
    if path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
        for idx, (synset, name) in data.items():
            labels[synset] = name
    elif path.suffix == '.txt':
        with open(path) as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    labels[parts[0]] = parts[1]
    return labels

