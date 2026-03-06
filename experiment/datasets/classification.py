"""Classification datasets: MNIST, CIFAR-10, ImageNette, ImageNet (+FFCV)."""

import os
import numpy as np
from typing import TYPE_CHECKING

from experiment.registry import register_dataset, _maybe_distributed_sampler

if TYPE_CHECKING:
    from experiment.config import Config


# =============================================================================
# Module-level Dataset Wrappers (must be at module scope for pickling)
# =============================================================================

class _PtArrayDataset:
    """Wraps numpy arrays as a PyTorch-compatible dataset."""
    def __init__(self, images, labels):
        import torch
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class _NpArrayDataset:
    """Wraps numpy arrays as a numpy-backend-compatible dataset."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.samples = [(i, int(labels[i])) for i in range(len(labels))]

    def __len__(self):
        return len(self.labels)

    def load_sample(self, idx):
        return self.images[idx], int(self.labels[idx])


class _NpNormalize:
    """Channel-wise normalize for CHW numpy arrays (ImageNet stats)."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, img):
        return (img - self.mean) / self.std


# =============================================================================
# PyTorch Datasets
# =============================================================================

@register_dataset('mnist', 'pytorch')
def _pt_mnist(config):
    import numpy as np, torch
    from torch.utils.data import DataLoader
    from pathlib import Path
    import urllib.request, gzip, struct
    data_path = Path(config.data_dir); data_path.mkdir(parents=True, exist_ok=True)
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = {'train_img': 'train-images-idx3-ubyte.gz', 'train_lbl': 'train-labels-idx1-ubyte.gz',
             'test_img': 't10k-images-idx3-ubyte.gz', 'test_lbl': 't10k-labels-idx1-ubyte.gz'}
    for fname in files.values():
        fp = data_path / fname
        if not fp.exists(): urllib.request.urlretrieve(base_url + fname, fp)
    def load_imgs(p):
        with gzip.open(p, 'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, r, c).astype(np.float32) / 255.0
    def load_lbls(p):
        with gzip.open(p, 'rb') as f:
            struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    train_imgs = load_imgs(data_path / files['train_img']); train_lbls = load_lbls(data_path / files['train_lbl'])
    test_imgs = load_imgs(data_path / files['test_img']); test_lbls = load_lbls(data_path / files['test_lbl'])
    np.random.seed(config.seed); n = len(train_imgs); n_val = int(n * config.val_split); idx = np.random.permutation(n)
    val_imgs, val_lbls = train_imgs[idx[:n_val]], train_lbls[idx[:n_val]]
    train_imgs, train_lbls = train_imgs[idx[n_val:]], train_lbls[idx[n_val:]]
    if config.subset: train_imgs, train_lbls = train_imgs[:config.subset], train_lbls[:config.subset]
    print(f"  MNIST: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None)
    train_ds = _PtArrayDataset(train_imgs, train_lbls)
    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
            DataLoader(_PtArrayDataset(val_imgs, val_lbls), config.batch_size, shuffle=False, **kw),
            DataLoader(_PtArrayDataset(test_imgs, test_lbls), config.batch_size, shuffle=False, **kw))


@register_dataset('cifar10', 'pytorch')
def _pt_cifar10(config):
    import torch; from torch.utils.data import DataLoader
    from torchvision import datasets, transforms as T
    t_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                         T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    t_test = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_full = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=t_train)
    test_ds = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=t_test)
    n = len(train_full); n_val = int(n * config.val_split)
    train_ds, val_ds = torch.utils.data.random_split(train_full, [n - n_val, n_val],
                                                      generator=torch.Generator().manual_seed(config.seed))
    if config.subset: train_ds = torch.utils.data.Subset(train_ds, range(config.subset))
    print(f"  CIFAR-10: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None)
    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
            DataLoader(test_ds, config.batch_size, shuffle=False, **kw))


@register_dataset('imagenette', 'pytorch')
def _pt_imagenette(config):
    import torch; from torch.utils.data import DataLoader
    from torchvision import datasets, transforms as T; from pathlib import Path
    sz = config.model_args.get('img_size', 224)
    t_train = T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), T.ToTensor(),
                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    t_val = T.Compose([T.Resize(int(sz * 256/224)), T.CenterCrop(sz), T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    root = Path(config.data_dir).expanduser() / 'imagenette2-320'
    if not root.exists():
        import urllib.request, tarfile
        tar = Path(config.data_dir) / 'imagenette2-320.tgz'
        if not tar.exists():
            print("  Downloading ImageNette..."); urllib.request.urlretrieve(
                'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz', tar)
        with tarfile.open(tar, 'r:gz') as t: t.extractall(config.data_dir)
        tar.unlink()
    train_ds = datasets.ImageFolder(str(root / 'train'), transform=t_train)
    val_ds = datasets.ImageFolder(str(root / 'val'), transform=t_val)
    if config.subset: train_ds = torch.utils.data.Subset(train_ds, range(config.subset))
    print(f"  ImageNette: {len(train_ds)} train, {len(val_ds)} val")
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None)
    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw))


@register_dataset('imagenet', 'pytorch')
def _pt_imagenet(config):
    if config.ffcv:
        return _pt_imagenet_ffcv(config)
    import torch; from torch.utils.data import DataLoader
    from torchvision import datasets, transforms as T; from pathlib import Path
    sz = config.model_args.get('img_size', 224)
    t_train = T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), T.ToTensor(),
                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    t_val = T.Compose([T.Resize(256), T.CenterCrop(sz), T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    root = Path(config.data_dir)
    train_ds = datasets.ImageFolder(str(root / 'train'), transform=t_train)
    val_ds = datasets.ImageFolder(str(root / 'val'), transform=t_val)
    if config.subset: train_ds = torch.utils.data.Subset(train_ds, range(config.subset))
    print(f"  ImageNet: {len(train_ds)} train, {len(val_ds)} val")
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=2 if config.num_workers > 0 else None,
              drop_last=True)
    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw))


def _write_beton(split_dir, beton_path, max_resolution=256, num_workers=16,
                 jpeg_quality=90):
    """Convert an ImageFolder split to FFCV .beton format."""
    from ffcv.writer import DatasetWriter
    from ffcv.fields import RGBImageField, IntField
    from torchvision.datasets import ImageFolder

    print(f"  Writing {beton_path} (this is a one-time cost)...")
    ds = ImageFolder(str(split_dir))
    writer = DatasetWriter(str(beton_path), {
        'image': RGBImageField(
            write_mode='jpg',
            max_resolution=max_resolution,
            jpeg_quality=jpeg_quality,
        ),
        'label': IntField(),
    }, num_workers=num_workers)
    writer.from_indexed_dataset(ds)
    print(f"  Done: {beton_path} ({beton_path.stat().st_size / 1e9:.1f} GB)")


def _pt_imagenet_ffcv(config):
    """ImageNet via FFCV — fast data loading with .beton files."""
    import torch
    import numpy as np
    from pathlib import Path
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import (
        RandomResizedCropRGBImageDecoder,
        CenterCropRGBImageDecoder,
        IntDecoder,
    )
    from ffcv.transforms import (
        ToTensor, ToDevice, ToTorchImage, NormalizeImage,
        RandomHorizontalFlip, Squeeze,
    )

    root = Path(config.data_dir).expanduser()
    ddp = getattr(config, 'ddp', False)

    beton_dir = Path(getattr(config, 'beton_dir', None) or '') if getattr(config, 'beton_dir', None) else root / 'ffcv'
    beton_dir.mkdir(parents=True, exist_ok=True)

    train_beton = beton_dir / 'train.beton'
    val_beton = beton_dir / 'val.beton'

    is_main = True
    if ddp:
        import torch.distributed as dist
        is_main = dist.get_rank() == 0

    beton_source = getattr(config, 'beton_source_dir', None)
    if beton_source:
        beton_source = str(Path(beton_source).expanduser())
    if is_main:
        for name in ('train.beton', 'val.beton'):
            dst = beton_dir / name
            if not dst.exists() and beton_source:
                src = Path(beton_source).expanduser() / name
                if src.exists():
                    import shutil
                    print(f"  Copying {src} → {dst} ...")
                    shutil.copy2(str(src), str(dst))
                    print(f"  Done ({dst.stat().st_size / 1e9:.1f} GB)")

        if not train_beton.exists():
            _write_beton(root / 'train', train_beton, max_resolution=256,
                         num_workers=config.num_workers)
        else:
            print(f"  FFCV train beton exists: {train_beton}")

        if not val_beton.exists():
            _write_beton(root / 'val', val_beton, max_resolution=256,
                         num_workers=config.num_workers)
        else:
            print(f"  FFCV val beton exists: {val_beton}")

    if ddp:
        dist.barrier()

    MEAN = np.array([0.485, 0.456, 0.406]) * 255
    STD = np.array([0.229, 0.224, 0.225]) * 255

    sz = config.model_args.get('img_size', 224)

    if ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_pipeline = {
        'image': [
            RandomResizedCropRGBImageDecoder((sz, sz)),
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(MEAN, STD, np.float32),
        ],
        'label': [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True),
        ],
    }
    val_pipeline = {
        'image': [
            CenterCropRGBImageDecoder((sz, sz), ratio=sz / 256),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(MEAN, STD, np.float32),
        ],
        'label': [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True),
        ],
    }

    train_order = OrderOption.RANDOM

    train_loader = Loader(
        str(train_beton),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        order=train_order,
        distributed=ddp,
        os_cache=True,
        drop_last=True,
        pipelines=train_pipeline,
        seed=config.seed,
    )

    val_loader = Loader(
        str(val_beton),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        order=OrderOption.SEQUENTIAL,
        distributed=ddp,
        os_cache=True,
        drop_last=False,
        pipelines=val_pipeline,
    )

    print(f"  ImageNet FFCV: {len(train_loader) * config.batch_size} train, "
          f"{len(val_loader) * config.batch_size} val")

    return train_loader, val_loader, val_loader


# =============================================================================
# Numpy Datasets
# =============================================================================

@register_dataset('mnist', 'numpy')
def _np_mnist(config):
    import numpy as np
    from python.utils.data_utils import DataLoader as NpLoader
    from pathlib import Path
    import urllib.request, gzip, struct
    data_path = Path(config.data_dir); data_path.mkdir(parents=True, exist_ok=True)
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = {'train_img': 'train-images-idx3-ubyte.gz', 'train_lbl': 'train-labels-idx1-ubyte.gz',
             'test_img': 't10k-images-idx3-ubyte.gz', 'test_lbl': 't10k-labels-idx1-ubyte.gz'}
    for fname in files.values():
        fp = data_path / fname
        if not fp.exists(): urllib.request.urlretrieve(base_url + fname, fp)
    def load_imgs(p):
        with gzip.open(p, 'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, r, c).astype(np.float32) / 255.0
    def load_lbls(p):
        with gzip.open(p, 'rb') as f:
            struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    train_imgs = load_imgs(data_path / files['train_img']); train_lbls = load_lbls(data_path / files['train_lbl'])
    test_imgs = load_imgs(data_path / files['test_img']); test_lbls = load_lbls(data_path / files['test_lbl'])
    np.random.seed(config.seed); n = len(train_imgs); n_val = int(n * config.val_split); idx = np.random.permutation(n)
    val_imgs, val_lbls = train_imgs[idx[:n_val]], train_lbls[idx[:n_val]]
    train_imgs, train_lbls = train_imgs[idx[n_val:]], train_lbls[idx[n_val:]]
    if config.subset: train_imgs, train_lbls = train_imgs[:config.subset], train_lbls[:config.subset]
    print(f"  MNIST: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    return (NpLoader(_NpArrayDataset(train_imgs, train_lbls), batch_size=config.batch_size, shuffle=True),
            NpLoader(_NpArrayDataset(val_imgs, val_lbls), batch_size=config.batch_size, shuffle=False),
            NpLoader(_NpArrayDataset(test_imgs, test_lbls), batch_size=config.batch_size, shuffle=False))


@register_dataset('imagenette', 'numpy')
def _np_imagenette(config):
    from python.experiments.image_net_data import download_imagenet, ImageNetDataset
    from python.utils.data_utils import DataLoader as NpLoader
    from python.vision import transforms
    root = download_imagenet(data_dir=config.data_dir, variant='imagenette', size='320')
    norm = getattr(transforms, 'Normalize', None)
    if norm is not None:
        norm_t = norm((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        norm_t = _NpNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    t_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), norm_t])
    t_val = transforms.Compose([norm_t])
    train_ds = ImageNetDataset(root, train=True, subset=config.subset, transform=t_train)
    val_ds = ImageNetDataset(root, train=False, transform=t_val)
    return (NpLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8),
            NpLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8),
            NpLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8))


@register_dataset('imagenet', 'numpy')
def _np_imagenet(config):
    from python.experiments.image_net_data import download_imagenet, ImageNetDataset
    from python.utils.data_utils import DataLoader as NpLoader
    from python.vision import transforms
    root = download_imagenet(data_dir=config.data_dir, variant='imagenet')
    norm = getattr(transforms, 'Normalize', None)
    if norm is not None:
        norm_t = norm((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        norm_t = _NpNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    t_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), norm_t])
    t_val = transforms.Compose([norm_t])
    train_ds = ImageNetDataset(root, train=True, subset=config.subset, transform=t_train)
    val_ds = ImageNetDataset(root, train=False, transform=t_val)
    return (NpLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8),
            NpLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8),
            NpLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8))
