"""
CLI entry point for experiments.

Usage:
    # MNIST quick test
    python -m experiment --dataset mnist --model cnn --epochs 5

    # CIFAR-10 with ResNet
    python -m experiment --dataset cifar10 --model resnet18 --epochs 50 --lr 0.01 --optimizer sgd

    # ImageNet with W&B logging
    python -m experiment --dataset imagenet --model resnet50 --epochs 90 \
        --lr 0.1 --optimizer sgd --scheduler step --batch_size 256 \
        --logger wandb --wandb_project imagenet-runs

    # From YAML config
    python -m experiment --config configs/imagenet_resnet50.yaml

    # YAML + CLI overrides
    python -m experiment --config configs/imagenet_resnet50.yaml --lr 0.01 --epochs 30

    # Quick sanity check (100 samples)
    python -m experiment --dataset imagenet --model resnet50 --subset 100 --epochs 1

    # List available models/datasets
    python -m experiment --list
"""

import sys

if '--list' in sys.argv:
    from experiment.registry import list_models, list_datasets
    print(f"Models:   {', '.join(list_models())}")
    print(f"Datasets: {', '.join(list_datasets())}")
    sys.exit(0)

from experiment import Config, run

config = Config.from_cli()
run(config)
