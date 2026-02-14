"""
Registry for models, datasets, and optimizers — backend-aware.

Register components with optional backend specifier:

    @register_model('resnet18')              # available for all backends
    @register_model('resnet18', 'pytorch')   # pytorch only
    @register_model('resnet18', 'numpy')     # numpy only

Then build by name — registry picks the right one based on config.backend:
    model = build_model(config)
"""

from typing import Tuple, Callable, Dict, Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from .config import Config

# =============================================================================
# Registry Machinery
# =============================================================================

_MODEL_REGISTRY: Dict[Tuple[str, Optional[str]], Callable] = {}
_DATASET_REGISTRY: Dict[Tuple[str, Optional[str]], Callable] = {}


def register_model(name: str, backend: Optional[str] = None):
    def decorator(fn):
        _MODEL_REGISTRY[(name.lower(), backend)] = fn
        return fn
    return decorator


def register_dataset(name: str, backend: Optional[str] = None):
    def decorator(fn):
        _DATASET_REGISTRY[(name.lower(), backend)] = fn
        return fn
    return decorator


def _lookup(registry, name, backend):
    name = name.lower()
    backend = backend.lower() if backend else None
    if (name, backend) in registry:
        return registry[(name, backend)]
    if (name, None) in registry:
        return registry[(name, None)]
    available = sorted(set(n for n, _ in registry.keys()))
    raise ValueError(f"Unknown '{name}' for backend '{backend}'. Available: {available}")


def list_models(backend=None):
    return sorted(set(n for n, b in _MODEL_REGISTRY if backend is None or b is None or b == backend))


def list_datasets(backend=None):
    return sorted(set(n for n, b in _DATASET_REGISTRY if backend is None or b is None or b == backend))


# =============================================================================
# Builders
# =============================================================================

def build_model(config):
    fn = _lookup(_MODEL_REGISTRY, config.model, config.backend)
    model = fn(config)
    if config.resume:
        if config.backend == 'pytorch':
            import torch
            state = torch.load(config.resume, map_location='cpu', weights_only=True)
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state)
        else:
            import numpy as np
            state = np.load(config.resume, allow_pickle=True).item()
            model.load_state_dict(state)
        print(f"  Resumed from {config.resume}")
    return model


def build_dataloaders(config):
    fn = _lookup(_DATASET_REGISTRY, config.dataset, config.backend)
    return fn(config)


def build_optimizer(model, config):
    name = config.optimizer.lower()
    if config.backend == 'pytorch':
        import torch
        params = model.parameters()
        if name == 'sgd':
            return torch.optim.SGD(params, lr=config.lr, momentum=config.momentum,
                                   weight_decay=config.weight_decay, **config.optimizer_args)
        elif name == 'adam':
            return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay,
                                    **config.optimizer_args)
        elif name == 'adamw':
            return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay,
                                     **config.optimizer_args)
        else:
            raise ValueError(f"Unknown optimizer '{name}'")
    else:
        from python.optimization.optimizers import SGD, Adam, AdamW
        params = list(model.parameters())
        if name == 'sgd':
            return SGD(params, lr=config.lr, momentum=config.momentum,
                       weight_decay=config.weight_decay, **config.optimizer_args)
        elif name == 'adam':
            return Adam(params, lr=config.lr, weight_decay=config.weight_decay,
                        **config.optimizer_args)
        elif name == 'adamw':
            return AdamW(params, lr=config.lr, weight_decay=config.weight_decay,
                         **config.optimizer_args)
        else:
            raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer, config):
    if config.backend != 'pytorch':
        return None
    import torch
    name = config.scheduler.lower()
    if name == 'step':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
    elif name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs)
    elif name in ('none', 'constant'):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
    else:
        raise ValueError(f"Unknown scheduler '{name}'")


def build_warmup_scheduler(optimizer, config):
    if config.backend != 'pytorch' or config.warmup_epochs <= 0:
        return None
    import torch
    return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                              total_iters=config.warmup_epochs)


# =============================================================================
# Built-in Models — PyTorch
# =============================================================================

@register_model('mlp', 'pytorch')
def _pt_mlp(config):
    import torch.nn as nn
    hidden = config.model_args.get('hidden_sizes', [512, 256])
    nc = config.model_args.get('num_classes', 10)
    inp = config.model_args.get('input_size', 784)
    layers = []
    prev = inp
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, nc))
    class MLP(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = nn.Sequential(*net)
        def forward(self, x):
            return self.net(x.reshape(x.shape[0], -1))
    return MLP(layers)


@register_model('cnn', 'pytorch')
def _pt_cnn(config):
    import torch.nn as nn
    ch = config.model_args.get('conv_channels', 32)
    nc = config.model_args.get('num_classes', 10)
    in_ch = config.model_args.get('in_channels', 1)
    img = config.model_args.get('img_size', 28)
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_ch, ch // 2, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch // 2, ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.fc = nn.Linear(ch * (img // 2) ** 2, nc)
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None: m.bias.data.zero_()
        def forward(self, x):
            return self.fc(self.features(x).reshape(x.shape[0], -1))
    return CNN()


@register_model('resnet18', 'pytorch')
def _pt_r18(config):
    import torch.nn as nn; from torchvision.models import resnet18
    nc = config.model_args.get('num_classes', 1000)
    m = resnet18(pretrained=config.pretrained)
    if nc != 1000: m.fc = nn.Linear(512, nc)
    return m

@register_model('resnet34', 'pytorch')
def _pt_r34(config):
    import torch.nn as nn; from torchvision.models import resnet34
    nc = config.model_args.get('num_classes', 1000)
    m = resnet34(pretrained=config.pretrained)
    if nc != 1000: m.fc = nn.Linear(512, nc)
    return m

@register_model('resnet50', 'pytorch')
def _pt_r50(config):
    import torch.nn as nn; from torchvision.models import resnet50
    nc = config.model_args.get('num_classes', 1000)
    m = resnet50(pretrained=config.pretrained)
    if nc != 1000: m.fc = nn.Linear(2048, nc)
    return m


# =============================================================================
# Built-in Models — Numpy
# =============================================================================

@register_model('mlp', 'numpy')
def _np_mlp(config):
    from python.nn_core.module import Module, Sequential
    from python.nn_core.linear import Linear
    from python.nn_core.activations import ReLU
    hidden = config.model_args.get('hidden_sizes', [512, 256])
    nc = config.model_args.get('num_classes', 10)
    inp = config.model_args.get('input_size', 784)
    layers = []
    prev = inp
    for h in hidden:
        layers += [Linear(prev, h), ReLU()]
        prev = h
    layers.append(Linear(prev, nc))
    class MLP(Module):
        def __init__(self, net):
            super().__init__()
            self.net = Sequential(*net)
        def forward(self, x):
            B = x.shape[0] if hasattr(x, 'shape') else x.data.shape[0]
            x = x.reshape(B, -1)
            return self.net(x)
    return MLP(layers)

@register_model('cnn', 'numpy')
def _pt_cnn(config):
    import python.nn_core as nn
    ch = config.model_args.get('conv_channels', 32)
    nc = config.model_args.get('num_classes', 10)
    in_ch = config.model_args.get('in_channels', 1)
    img = config.model_args.get('img_size', 28)
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_ch, ch // 2, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch // 2, ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.fc = nn.Linear(ch * (img // 2) ** 2, nc)
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
        def forward(self, x):
            return self.fc(self.features(x).reshape(x.shape[0], -1))
    return CNN()



@register_model('resnet18', 'numpy')
def _np_r18(config):
    from python.vision.models.resnet import resnet18
    return resnet18(num_classes=config.model_args.get('num_classes', 1000))

@register_model('resnet34', 'numpy')
def _np_r34(config):
    from python.vision.models.resnet import resnet34
    return resnet34(num_classes=config.model_args.get('num_classes', 1000))

@register_model('resnet50', 'numpy')
def _np_r50(config):
    from python.vision.models.resnet import resnet50
    return resnet50(num_classes=config.model_args.get('num_classes', 1000))

@register_model('alexnet', 'numpy')
def _np_alexnet(config):
    from python.vision.models.alexnet import alexnet
    return alexnet(num_classes=config.model_args.get('num_classes', 1000))


# =============================================================================
# Built-in Datasets — PyTorch
# =============================================================================

@register_dataset('mnist', 'pytorch')
def _pt_mnist(config):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms as T

    t = T.Compose([T.ToTensor()])
    train_full = datasets.MNIST(config.data_dir, train=True, download=True, transform=t)
    test_ds = datasets.MNIST(config.data_dir, train=False, download=True, transform=t)

    n = len(train_full)
    n_val = int(n * config.val_split)
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [n - n_val, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )
    if config.subset:
        train_ds = torch.utils.data.Subset(train_ds, range(config.subset))

    print(f"  MNIST: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory)
    return (
        DataLoader(train_ds, config.batch_size, shuffle=True, **kw),
        DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
        DataLoader(test_ds, config.batch_size, shuffle=False, **kw),
    )


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
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory)
    return (DataLoader(train_ds, config.batch_size, shuffle=True, **kw),
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
    root = Path(config.data_dir) / 'imagenette2-320'
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
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory)
    return (DataLoader(train_ds, config.batch_size, shuffle=True, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw))


@register_dataset('imagenet', 'pytorch')
def _pt_imagenet(config):
    import torch; from torch.utils.data import DataLoader
    from torchvision import datasets, transforms as T; from pathlib import Path
    from pytorch.experiments.image_net_data import download_imagenet
    sz = config.model_args.get('img_size', 224)
    t_train = T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), T.ToTensor(),
                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    t_val = T.Compose([T.Resize(256), T.CenterCrop(sz), T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    data_dir = Path(download_imagenet(variant="imagenet"))
    train_ds = datasets.ImageFolder(str(data_dir / 'train'), transform=t_train)
    val_ds = datasets.ImageFolder(str(data_dir / 'val'), transform=t_val)
    if config.subset: train_ds = torch.utils.data.Subset(train_ds, range(config.subset))
    print(f"  ImageNet: {len(train_ds)} train, {len(val_ds)} val")
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory)
    return (DataLoader(train_ds, config.batch_size, shuffle=True, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
            DataLoader(val_ds, config.batch_size, shuffle=False, **kw))


# =============================================================================
# Built-in Datasets — Numpy
# =============================================================================
from python.utils.data_utils import Dataset as NpDataset
class AD_Np(NpDataset):
    def __init__(s, i, l): s.imgs, s.lbls = i, l; s.samples = [(j, int(l[j])) for j in range(len(l))]
    def __len__(s): return len(s.lbls)
    def load_sample(s, i): return s.imgs[i], int(s.lbls[i])

@register_dataset('mnist', 'numpy')
def _np_mnist(config):
    import numpy as np
    from python.utils.data_utils import DataLoader as NpLoader, Dataset as NpDataset
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
    return (NpLoader(AD_Np(train_imgs, train_lbls), batch_size=config.batch_size, shuffle=True),
            NpLoader(AD_Np(val_imgs, val_lbls), batch_size=config.batch_size, shuffle=False),
            NpLoader(AD_Np(test_imgs, test_lbls), batch_size=config.batch_size, shuffle=False))


@register_dataset('imagenette', 'numpy')
def _np_imagenette(config):
    from python.experiments.image_net_data import download_imagenet, ImageNetDataset
    from python.utils.data_utils import DataLoader as NpLoader
    from python.vision import transforms
    root = download_imagenet(data_dir=config.data_dir, variant='imagenette', size='320')
    t_train = transforms.Compose([transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    t_val = transforms.Compose([transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    train_ds = ImageNetDataset(root, type='train', subset=config.subset, transform=t_train)
    val_ds = ImageNetDataset(root, type='val', transform=t_val)
    test_ds = ImageNetDataset(root, type='val', transform=t_val)
    return (NpLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8),
            NpLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8),
            NpLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=8))


@register_dataset('imagenet', 'numpy')
def _np_imagenet(config):
    from python.experiments.image_net_data import download_imagenet, ImageNetDataset
    from python.utils.data_utils import DataLoader as NpLoader
    from python.vision import transforms
    root = download_imagenet(data_dir=config.data_dir, variant='imagenet')
    t_train = transforms.Compose([transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    t_val = transforms.Compose([transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    train_ds = ImageNetDataset(root, type='train', subset=config.subset, transform=t_train)
    val_ds = ImageNetDataset(root, type='val', transform=t_val)
    test_ds = ImageNetDataset(root, type='test', transform=t_val)
    return (NpLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8),
            NpLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8),
            NpLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=8))