"""
Experiment Framework.

Quick start:
    from experiment import Config, run
    run(Config(dataset='mnist', model='cnn', epochs=10))

CLI:
    python -m experiment --dataset cifar10 --model resnet18 --epochs=50 --logger wandb

DDP (multi-GPU):
    torchrun --nproc_per_node=4 -m experiment --config configs/imagenet_resnet18_ddp.yaml

Register custom models/datasets:
    from experiment import register_model, register_dataset

    @register_model('my_vit')
    def build_vit(config):
        return MyViT(**config.model_args)
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from .config import Config
from .runner import run
from .logger import Logger
from .registry import (
    register_model, register_dataset,
    build_model, build_dataloaders, build_optimizer,
    list_models, list_datasets,
)

__all__ = [
    'Config', 'run', 'Logger',
    'register_model', 'register_dataset',
    'build_model', 'build_dataloaders', 'build_optimizer',
    'list_models', 'list_datasets',
]