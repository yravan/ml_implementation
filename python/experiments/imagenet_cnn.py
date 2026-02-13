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
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from collections import OrderedDict

from python.utils.data_utils import DataLoader
from python.experiments.image_net_data import ImageNetDataset, download_imagenet
from python.experiments.train_utils import (
    print_header,
    adjust_learning_rate,
    train_epoch,
    evaluate,
    format_time,
    plot_training_history,
    top1_accuracy,
    top5_accuracy,
    print_metrics,
)
# Our framework imports
from python.optimization import SGD, CrossEntropyLoss, AdamW, Adam
from python.vision import transforms
from python.vision.models import resnet18, resnet34, resnet50, alexnet


def load_data(subset, batch_size):

    print_header("Loading Data", '-')

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = download_imagenet(variant='imagenette', size='160')
    train_dataset = ImageNetDataset(data_dir, train=True, subset=subset, transform=train_transform)
    val_dataset = ImageNetDataset(data_dir, train=False, subset=subset // 10 if subset else None, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Batches/epoch:    train={len(train_loader)}, val={len(val_loader)}")
    return train_loader, val_loader

def load_model(model_type, train_loader: DataLoader):
    print_header("Model Architecture", "-")

    if model_type == "resnet18":
        model = resnet18(num_classes=train_loader.dataset.num_classes)
    elif model_type == "resnet34":
        model = resnet34(num_classes=train_loader.dataset.num_classes)
    elif model_type == "resnet50":
        model = resnet50(num_classes=train_loader.dataset.num_classes)
    elif model_type == "alexnet":
        model = alexnet(num_classes=train_loader.dataset.num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"  Model: {model_type}")
    print(model)

    total_params = sum(np.prod(p.data.shape) for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    return model

def load_optimizer_and_loss(model, optimizer_type, optimizer_params, learning_rate, lr_schedule):
    print_header("Training Setup", "-")

    params = list(model.parameters())

    if optimizer_type == "sgd":
        default_sgd = {"momentum": 0.9, "weight_decay": 1e-4}
        default_sgd.update(optimizer_params)
        opt = SGD(params, lr=learning_rate, **default_sgd)
    elif optimizer_type == "adam":
        opt = Adam(params, lr=learning_rate, **optimizer_params)
    elif optimizer_type == "adamw":
        default_adamw = {"weight_decay": 1e-4}
        default_adamw.update(optimizer_params)
        opt = AdamW(params, lr=learning_rate, **default_adamw)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    criterion = CrossEntropyLoss()

    print(f"  Optimizer: {optimizer_type} (lr={learning_rate})")
    print(f"  LR schedule: {lr_schedule}")
    print(f"  Loss: CrossEntropyLoss")
    return opt, criterion

# =============================================================================
# Main Training Script
# =============================================================================

def train(
    data_dir: str = './data/imagenet',
    epochs: int = 90,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    optimizer_type: str = 'sgd',
    optimizer_params: Optional[dict] = None,
    model_type: str = 'resnet18',
    lr_schedule: str = 'step',
    log_interval: int = 100,
    seed: int = 42,
    save_plots: bool = True,
    output_dir: str = './outputs/imagenet',
    subset: Optional[int] = None,
    label_file: Optional[str] = None,
):
    """
    Full ImageNet training pipeline.

    Args:
        data_dir: Path to ImageNet root (contains train/ and val/)
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Initial learning rate
        optimizer_type: 'sgd', 'adam', or 'adamw'
        optimizer_params: Extra optimizer kwargs
        model_type: 'resnet18', 'resnet34', or 'resnet50'
        lr_schedule: 'step' (30/60/80 decay) or 'cosine'
        log_interval: Print every N batches
        seed: Random seed
        save_plots: Whether to save training curves
        output_dir: Directory for outputs
        subset: If set, use only this many samples per split (for debugging)
        label_file: Path to class label mapping file
    """
    np.random.seed(seed)

    if optimizer_params is None:
        optimizer_params = {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print_header("ImageNet Training Experiment")

    # =========================================================================
    # Configuration
    # =========================================================================
    print_header("Configuration", '-')
    print(f"  Data dir:      {data_dir}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LR schedule:   {lr_schedule}")
    print(f"  Optimizer:     {optimizer_type}")
    print(f"  Optimizer params: {optimizer_params}")
    print(f"  Model:         {model_type}")
    print(f"  Subset:        {subset or 'full'}")
    print(f"  Seed:          {seed}")

    # =========================================================================
    # Data Loading
    # =========================================================================
    train_loader, val_loader = load_data(subset, batch_size)

    # =========================================================================
    # Model Setup
    # =========================================================================
    model = load_model(model_type, train_loader)
    # for i, layer in enumerate(model.conv_layers._modules.values()):
    #     if hasattr(layer, 'weight'):
    #         w = layer.weight.data
    #         b = layer.bias.data if layer.bias is not None else None
    #         fan_in = w.shape[1] * (w.shape[2] * w.shape[3] if w.ndim == 4 else 1)
    #         expected_std = np.sqrt(2.0 / fan_in)  # kaiming
    #         print(f"Layer {i:2d}  w.std={w.std():.4f} (expected ~{expected_std:.4f})  "
    #               f"w.mean={w.mean():.6f}  bias.mean={b.mean():.6f}  bias.max={np.abs(b).max():.6f}")

    # =========================================================================
    # Optimizer and Loss
    # =========================================================================
    opt, criterion = load_optimizer_and_loss(model, optimizer_type, optimizer_params, learning_rate, lr_schedule)

    # =========================================================================
    # Metrics — adapt to number of classes
    # =========================================================================
    metrics = {"top1": top1_accuracy}
    if train_loader.dataset.num_classes > 10:
        metrics["top5"] = top5_accuracy
    history = {f"train_{k}": [] for k in ["loss"] + list(metrics)}
    history.update({f"val_{k}": [] for k in ["loss"] + list(metrics)})
    history["lr"] = []

    # =========================================================================
    # Training Loop
    # =========================================================================
    print_header("Training", '-')

    best_val_top1 = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Adjust learning rate
        current_lr = adjust_learning_rate(opt, epoch - 1, learning_rate, lr_schedule)

        print(f"\n  Epoch {epoch}/{epochs} (lr={current_lr:.6f})")
        print(f"  {'─' * 60}")

        # Train
        train_results = train_epoch(
            model, train_loader, criterion, opt,
            metrics=metrics, log_interval=log_interval,
            # profile=True,
            # debug=True,
        )

        # Validate
        val_results = evaluate(
            model, val_loader, criterion, metrics=metrics,
        )

        # Record history
        for key, val in train_results.items():
            history[f'train_{key}'].append(val)
        for key, val in val_results.items():
            history[f'val_{key}'].append(val)
        history['lr'].append(current_lr)

        # Best tracking
        best_marker = ""
        if val_results["top1"] > best_val_top1:
            best_val_top1 = val_results["top1"]
            best_marker = " ★ New Best!"
            best_state = model.state_dict()

        epoch_time = time.time() - epoch_start

        print_metrics(train_results, prefix="Train | ")
        print_metrics(val_results, prefix="Val   | ")
        if best_marker:
            print(f"  {best_marker}")
        print(f"  Time: {format_time(epoch_time)}")

        # Save checkpoint periodically
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'history': history,
                'best_val_top1': best_val_top1,
            }
            ckpt_path = output_path / f'checkpoint_epoch{epoch}.npy'
            np.save(str(ckpt_path), checkpoint)
            print(f"  Saved checkpoint to {ckpt_path}")

    total_time = time.time() - start_time

    # =========================================================================
    # Final Summary
    # =========================================================================
    print_header("Training Complete!", "=")

    print(f"  Total training time: {format_time(total_time)}")
    print(f"  Best validation Top-1: {best_val_top1 * 100:.2f}%")
    print_metrics(val_results, prefix="Final | ")

    # =========================================================================
    # Visualization
    # =========================================================================
    if save_plots and len(history["train_loss"]) > 0:
        print_header("Generating Visualizations", "-")
        plot_training_history(
            history, save_path=str(output_path / "imagenet_training_curves.png")
        )

    print_header("Done!", '=')

    return {
        'model': model,
        'history': history,
        'best_val_top1': best_val_top1,
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ImageNet Training')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet',
                        help='Path to ImageNet dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--lr-schedule', type=str, default='step',
                        choices=['step', 'cosine'])
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of data for debugging')
    parser.add_argument('--output-dir', type=str, default='./outputs/imagenet')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--label-file', type=str, default=None)

    args = parser.parse_args()

    optimizer_params = {}
    if args.optimizer == 'sgd':
        optimizer_params = {'momentum': args.momentum, 'weight_decay': args.weight_decay}
    elif args.optimizer in ('adam', 'adamw'):
        optimizer_params = {'weight_decay': args.weight_decay, 'betas': (0.9, 0.999)}

    results = train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer_type=args.optimizer,
        optimizer_params=optimizer_params,
        model_type='resnet18',
        lr_schedule=args.lr_schedule,
        log_interval=10,
        seed=args.seed,
        save_plots=True,
        output_dir=args.output_dir,
        subset=args.subset,
        label_file=args.label_file,
    )