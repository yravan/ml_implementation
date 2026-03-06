"""Classification models: MLP, CNN, ResNet."""

from experiment.registry import register_model


# =============================================================================
# PyTorch
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
    import torch.nn as nn; from pytorch.vision.models import resnet18
    nc = config.model_args.get('num_classes', 1000)
    m = resnet18(num_classes=nc)
    if nc != 1000: m.fc = nn.Linear(512, nc)
    return m

@register_model('resnet34', 'pytorch')
def _pt_r34(config):
    import torch.nn as nn; from pytorch.vision.models import resnet34
    nc = config.model_args.get('num_classes', 1000)
    m = resnet34(num_classes=nc)
    if nc != 1000: m.fc = nn.Linear(512, nc)
    return m

@register_model('resnet50', 'pytorch')
def _pt_r50(config):
    import torch.nn as nn; from pytorch.vision.models import resnet50
    nc = config.model_args.get('num_classes', 1000)
    m = resnet50(num_classes=nc)
    if nc != 1000: m.fc = nn.Linear(2048, nc)
    return m


# =============================================================================
# Numpy
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
def _np_cnn(config):
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
