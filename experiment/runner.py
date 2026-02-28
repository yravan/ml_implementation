"""
Experiment Runner — dual-backend (PyTorch + custom numpy framework).

Usage:
    from experiment import Config, run

    # PyTorch
    run(Config(backend='pytorch', dataset='mnist', model='cnn', epochs=10))

    # Your custom numpy framework
    run(Config(backend='numpy', dataset='mnist', model='mlp', epochs=10))

    # Same config, swap backend
    cfg = Config(dataset='cifar10', model='resnet18', epochs=50)
    run(Config(**{**cfg.to_dict(), 'backend': 'pytorch'}))
    run(Config(**{**cfg.to_dict(), 'backend': 'numpy'}))
"""

import os
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional

from .config import Config
from .logger import Logger
from .registry import (
    build_model, build_dataloaders, build_optimizer,
    build_scheduler, build_warmup_scheduler,
)


# =============================================================================
# Metrics — pure numpy, work for both backends
# =============================================================================

def _to_numpy(x):
    """Convert torch tensor or custom Tensor to numpy."""
    if hasattr(x, 'detach'):  # torch tensor
        return x.detach().cpu().numpy()
    if hasattr(x, 'data') and isinstance(x.data, np.ndarray):  # custom Tensor
        return x.data
    return np.asarray(x)


def _is_torch(x):
    """Check if x is a PyTorch tensor."""
    return hasattr(x, 'detach') and hasattr(x, 'argmax')


def top1_accuracy(logits, targets) -> float:
    if _is_torch(logits):
        return (logits.argmax(1) == targets).sum()  # stays on GPU
    logits, targets = _to_numpy(logits), _to_numpy(targets)
    return float((logits.argmax(axis=1) == targets).sum())


def top5_accuracy(logits, targets) -> float:
    if _is_torch(logits):
        _, top5 = logits.topk(5, dim=1)
        return (top5 == targets.unsqueeze(1)).any(1).sum()  # stays on GPU
    logits, targets = _to_numpy(logits), _to_numpy(targets)
    top5 = np.argsort(logits, axis=1)[:, -5:]
    return float(np.any(top5 == targets[:, None], axis=1).sum())


METRIC_FNS = {'top1': top1_accuracy, 'top5': top5_accuracy}


# =============================================================================
# Backend helpers
# =============================================================================

class _PyTorchBackend:
    """All PyTorch-specific logic in one place."""

    def __init__(self, config=None):
        import torch
        import torch.nn.functional as F
        self.torch = torch
        self.F = F

        # ── DDP state ───────────────────────────────────────────────
        self.ddp = config.ddp if config else False
        if self.ddp:
            import torch.distributed as dist
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = dist.get_world_size()
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device(
                'cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
            )

        self.is_main = (self.rank == 0)

        # AMP (automatic mixed precision)
        use_amp = config.amp if config else False
        # AMP only benefits CUDA (tensor cores); skip on MPS/CPU
        self._use_amp = use_amp and self.device.type == 'cuda'
        self._scaler = torch.amp.GradScaler('cuda') if self._use_amp else None

    def to_device(self, model):
        return model.to(self.device)

    def wrap_ddp(self, model, config):
        """Wrap model with DistributedDataParallel if DDP is enabled."""
        if not self.ddp:
            return model
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=config.ddp_find_unused,
            gradient_as_bucket_view=config.ddp_gradient_as_bucket,
        )
        return model

    def train_step(self, model, batch, criterion, optimizer, config):
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(self.device, non_blocking=True, memory_format=self.torch.channels_last)
        targets = targets.to(self.device, non_blocking=True)

        if self._use_amp:
            with self.torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            self._scaler.scale(loss).backward()
            if config.grad_clip:
                self._scaler.unscale_(optimizer)
                self.torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                self.torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        return logits, loss.detach(), inputs.shape[0], targets

    def eval_step(self, model, batch, criterion):
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(self.device, non_blocking=True, memory_format=self.torch.channels_last)
        targets = targets.to(self.device, non_blocking=True)

        if self._use_amp:
            with self.torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, targets)
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
        return logits, loss.detach(), inputs.shape[0], targets

    def no_grad_context(self):
        return self.torch.no_grad()

    def criterion(self, config=None):
        ls = config.label_smoothing if config else 0.0
        if ls > 0:
            return lambda logits, targets: self.F.cross_entropy(logits, targets, label_smoothing=ls)
        return self.F.cross_entropy

    def get_lr(self, optimizer):
        return optimizer.param_groups[0]['lr']

    def param_count(self, model):
        return sum(p.numel() for p in model.parameters())

    def _unwrap(self, model):
        """Unwrap DDP module to get the underlying model."""
        return model.module if hasattr(model, 'module') else model

    def save_checkpoint(self, model, optimizer, epoch, config, path):
        # Only save on main process in DDP
        if not self.is_main:
            return
        self.torch.save({
            'epoch': epoch,
            'model_state_dict': self._unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.to_dict(),
        }, path)

    def load_best(self, model, path):
        state = self.torch.load(path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        self._unwrap(model).load_state_dict(state)


class _NumpyBackend:
    """All custom numpy framework logic in one place."""

    def __init__(self):
        # Lazy imports — only when numpy backend is actually used
        self.is_main = True  # numpy backend is always single-process

    @property
    def device(self):
        return 'cpu'

    def to_device(self, model):
        return model  # numpy is always CPU

    def train_step(self, model, batch, criterion, optimizer, config):
        from python.foundations.computational_graph import Tensor

        inputs, targets = batch[0], batch[1]
        # Ensure numpy arrays
        if hasattr(inputs, 'numpy'):
            inputs = inputs.numpy()
        if hasattr(targets, 'numpy'):
            targets = targets.numpy()

        x = Tensor(inputs, requires_grad=True)
        y = Tensor(targets)

        logits = model(x)
        loss = criterion(logits, y, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, len(inputs), targets

    def eval_step(self, model, batch, criterion):
        from python.foundations.computational_graph import Tensor

        inputs, targets = batch[0], batch[1]
        if hasattr(inputs, 'numpy'):
            inputs = inputs.numpy()
        if hasattr(targets, 'numpy'):
            targets = targets.numpy()

        x = Tensor(inputs, requires_grad=False)
        y = Tensor(targets)

        logits = model(x)
        loss = criterion(logits, y, reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, len(inputs), targets

    def no_grad_context(self):
        """Context manager that sets _no_grad flag in custom framework."""
        import python.foundations as F
        class _NoGrad:
            def __enter__(self_):
                self_._prev = F._no_grad
                F._no_grad = True
            def __exit__(self_, *args):
                F._no_grad = self_._prev
        return _NoGrad()

    def criterion(self):
        from python.optimization.losses import CrossEntropyLoss
        return CrossEntropyLoss()

    def get_lr(self, optimizer):
        return optimizer.defaults['lr']

    def param_count(self, model):
        return sum(p.data.size for p in model.parameters())

    def save_checkpoint(self, model, optimizer, epoch, config, path):
        np.save(str(path), {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
        })

    def load_best(self, model, path):
        state = np.load(str(path), allow_pickle=True).item()
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)


def _get_backend(config):
    if config.backend == 'pytorch':
        return _PyTorchBackend(config)
    else:
        return _NumpyBackend()


# =============================================================================
# Train / Eval — backend-agnostic
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, config, logger,
                    metric_fns, backend, epoch):
    """Train for one epoch. Returns dict of averaged metrics."""
    model.train()
    total_loss = 0.0       # accumulator (GPU tensor for pytorch, float for numpy)
    total_samples = 0
    metric_totals = {name: 0.0 for name in metric_fns}
    epoch_start = time.time()
    interval_start = epoch_start
    interval_samples = 0

    for batch_idx, batch in enumerate(loader):
        logits, batch_loss, n, targets_d = backend.train_step(
            model, batch, criterion, optimizer, config
        )

        # Accumulate — stays on GPU for pytorch (no sync)
        total_loss = total_loss + batch_loss * n
        total_samples += n
        interval_samples += n

        logits_d = logits.detach() if hasattr(logits, 'detach') else logits
        for name, fn in metric_fns.items():
            metric_totals[name] = metric_totals[name] + fn(logits_d, targets_d)

        # Log batch — THIS is where we sync (only every log_interval batches)
        global_step = (epoch - 1) * len(loader) + batch_idx
        if config.log_interval and (batch_idx + 1) % config.log_interval == 0:
            avg_loss = float(total_loss) / total_samples  # single sync here
            now = time.time()
            interval_elapsed = now - interval_start
            ips = interval_samples / interval_elapsed if interval_elapsed > 0 else 0.0
            interval_start = now
            interval_samples = 0
            parts = [f"Batch {batch_idx+1:5d}/{len(loader)}", f"Loss: {avg_loss:.4f}"]
            for name in metric_fns:
                parts.append(f"{name}: {float(metric_totals[name])/total_samples*100:.2f}%")
            parts.append(f"{ips:.0f} img/s")
            if backend.is_main:
                print(f"    {' | '.join(parts)}")
            logger.log_scalar('train/batch_loss', avg_loss, step=global_step)
            logger.log_scalar('train/throughput', ips, step=global_step)

    # Final sync at end of epoch
    epoch_elapsed = time.time() - epoch_start
    throughput = total_samples / epoch_elapsed if epoch_elapsed > 0 else 0.0

    results = {'loss': float(total_loss) / total_samples}
    for name in metric_fns:
        results[name] = float(metric_totals[name]) / total_samples
    results['_throughput'] = throughput
    return results


def evaluate(model, loader, criterion, metric_fns, backend,
             collect_predictions=False):
    """Evaluate model. Returns dict of averaged metrics."""
    model.eval()
    total_loss = 0.0       # GPU tensor for pytorch, float for numpy
    total_samples = 0
    metric_totals = {name: 0.0 for name in metric_fns}
    all_logits = [] if collect_predictions else None
    all_labels = [] if collect_predictions else None

    with backend.no_grad_context():
        for batch in loader:
            logits, batch_loss, n, targets_d = backend.eval_step(model, batch, criterion)

            total_loss = total_loss + batch_loss * n
            total_samples += n

            logits_d = logits.detach() if hasattr(logits, 'detach') else logits
            for name, fn in metric_fns.items():
                metric_totals[name] = metric_totals[name] + fn(logits_d, targets_d)

            if collect_predictions:
                all_logits.append(_to_numpy(logits))
                all_labels.append(_to_numpy(targets_d))

    # Single sync at end of eval
    results = {'loss': float(total_loss) / total_samples}
    for name in metric_fns:
        results[name] = float(metric_totals[name]) / total_samples
    if collect_predictions:
        results['logits'] = np.concatenate(all_logits)
        results['labels'] = np.concatenate(all_labels)
    return results


# =============================================================================
# Main Runner
# =============================================================================

def _setup_ddp(config):
    """Initialize the DDP process group. Must be called before anything else."""
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend=config.ddp_backend)


def _cleanup_ddp():
    """Destroy the DDP process group."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def _set_epoch_samplers(loader, epoch):
    """Set epoch on DistributedSampler so shuffling differs per epoch."""
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)


def run(config: Config) -> Dict:
    """
    Run a complete experiment. Works with both PyTorch and numpy backends.

    For DDP, launch via torchrun:
        torchrun --nproc_per_node=N -m experiment --config cfg.yaml --ddp

    Returns dict with model, history, and test results.
    """
    # ── DDP Init ─────────────────────────────────────────────────────
    if config.ddp:
        _setup_ddp(config)

    # ── Setup ────────────────────────────────────────────────────────
    np.random.seed(config.seed)
    backend = _get_backend(config)
    is_main = backend.is_main  # True for rank 0 or non-DDP

    # Helper: only print on main process
    def mprint(*args, **kwargs):
        if is_main:
            print(*args, **kwargs)

    if config.backend == 'pytorch':
        import torch
        torch.manual_seed(config.seed + backend.rank)  # different seed per rank
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + backend.rank)
            if config.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            # Use tensor cores for FP32 matmuls (TF32: same range, slightly reduced precision)
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # pin_memory is not supported on MPS — silently disable to
        # avoid the noisy UserWarning from DataLoader.
        if config.pin_memory and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config = Config(**{**config.to_dict(), 'pin_memory': False})

    mprint(f"\n{'='*70}")
    mprint(f" {config.name}  [{config.backend}]")
    mprint(f"{'='*70}")
    mprint(config)
    mprint(f"  Device: {backend.device}")
    if config.ddp:
        mprint(f"  DDP:    {backend.world_size} processes, backend={config.ddp_backend}")
    if config.backend == 'pytorch':
        parts = []
        if getattr(backend, '_use_amp', False):
            parts.append("AMP")
        if config.cudnn_benchmark and backend.device.type == 'cuda':
            parts.append("cuDNN benchmark")
        if config.compile:
            parts.append("torch.compile")
        if config.ddp:
            parts.append("DDP")
        if parts:
            mprint(f"  Perf:   {', '.join(parts)}")

    # ── Data ─────────────────────────────────────────────────────────
    mprint(f"\n{'─'*70}\n Data\n{'─'*70}")
    train_loader, val_loader, test_loader = build_dataloaders(config)

    # ── Model ────────────────────────────────────────────────────────
    mprint(f"\n{'─'*70}\n Model\n{'─'*70}")
    model = build_model(config)
    model = backend.to_device(model)
    if config.backend == 'pytorch':
        import torch
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)
            mprint("  memory_format: channels_last (NHWC)")
    if config.compile and config.backend == 'pytorch':
        import torch
        model = torch.compile(model, mode=config.compile_mode)
        mprint("  torch.compile: enabled (max-autotune)")

    # ── DDP wrap (after compile, after to_device) ────────────────────
    if config.ddp:
        model = backend.wrap_ddp(model, config)
        mprint(f"  DDP:    wrapped with DistributedDataParallel")

    mprint(model)
    mprint(f"  Parameters: {backend.param_count(backend._unwrap(model) if config.ddp else model):,}")

    # ── Optimizer / Scheduler ────────────────────────────────────────
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    warmup = build_warmup_scheduler(optimizer, config)
    criterion = backend.criterion(config)

    # ── Logger (only on main process for DDP) ────────────────────────
    mprint(f"\n{'─'*70}\n Logger\n{'─'*70}")
    logger = Logger(config) if is_main else None
    if logger and config.backend == 'pytorch':
        logger.watch_model(model)

    # ── Metrics ──────────────────────────────────────────────────────
    metric_fns = {name: METRIC_FNS[name] for name in config.metrics
                  if name in METRIC_FNS}

    # ── Training ─────────────────────────────────────────────────────
    mprint(f"\n{'─'*70}\n Training\n{'─'*70}")
    if logger:
        logger.begin_training()

    history = {f'{s}_{k}': [] for s in ('train', 'val')
               for k in ['loss'] + list(metric_fns)}
    best_metric = 0.0 if metric_fns else float('inf')
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        if logger:
            logger.begin_epoch(epoch)
        mprint(f"\n  Epoch {epoch}/{config.epochs}")
        mprint(f"  {'─' * 50}")

        # Set epoch on DistributedSampler for proper shuffling
        if config.ddp:
            _set_epoch_samplers(train_loader, epoch)

        # Train
        train_results = train_one_epoch(
            model, train_loader, criterion, optimizer,
            config, logger if is_main else _NullLogger(), metric_fns, backend, epoch,
        )
        throughput = train_results.pop('_throughput', 0.0)

        # Val
        val_results = evaluate(model, val_loader, criterion, metric_fns, backend)

        # LR step (pytorch only — numpy uses manual schedule)
        current_lr = backend.get_lr(optimizer)
        if scheduler is not None:
            if warmup and epoch <= config.warmup_epochs:
                warmup.step()
            else:
                scheduler.step()

        # Record history (all ranks keep history, but only main logs)
        for k, v in train_results.items():
            history[f'train_{k}'].append(v)
        for k, v in val_results.items():
            history[f'val_{k}'].append(v)

        # Log — use end-of-epoch global step so W&B steps are monotonic
        if logger:
            epoch_global_step = epoch * len(train_loader)
            logger.end_epoch(train_results, val_results, lr=current_lr,
                             throughput=throughput, step=epoch_global_step)

        # Best model tracking
        primary_metric = list(metric_fns.keys())[0] if metric_fns else 'loss'
        current = val_results.get(primary_metric, val_results.get('loss', 0))
        is_best = (current > best_metric) if primary_metric != 'loss' else (current < best_metric)

        if is_best:
            best_metric = current
            best_epoch = epoch
            mprint(f"  ★ New best {primary_metric}!")
            ext = '.pt' if config.backend == 'pytorch' else '.npy'
            best_path = config.run_dir / f'best{ext}'
            best_path.parent.mkdir(parents=True, exist_ok=True)
            backend.save_checkpoint(model, optimizer, epoch, config, best_path)

        # Periodic save
        if config.save_every and epoch % config.save_every == 0:
            ext = '.pt' if config.backend == 'pytorch' else '.npy'
            ckpt_path = config.run_dir / f'checkpoint_{epoch}{ext}'
            backend.save_checkpoint(model, optimizer, epoch, config, ckpt_path)

        # Synchronize all processes before next epoch
        if config.ddp:
            import torch.distributed as dist
            dist.barrier()

    # ── Test ─────────────────────────────────────────────────────────
    mprint(f"\n{'─'*70}\n Test Evaluation\n{'─'*70}")

    # Load best model
    ext = '.pt' if config.backend == 'pytorch' else '.npy'
    best_path = config.run_dir / f'best{ext}'
    if best_path.exists():
        backend.load_best(model, best_path)
        mprint(f"  Loaded best model (epoch {best_epoch})")

    test_results = evaluate(
        model, test_loader, criterion, metric_fns, backend,
        collect_predictions=True,
    )

    if logger:
        final_step = config.epochs * len(train_loader) + 1
        logger.log_scalars('test', {k: v for k, v in test_results.items()
                                    if k not in ('logits', 'labels')}, step=final_step)
        logger.flush()

    # Summary
    mprint(f"\n{'='*70}")
    mprint(f" Done!  [{config.backend}]")
    mprint(f"{'='*70}")
    for k, v in test_results.items():
        if k in ('logits', 'labels'):
            continue
        mprint(f"  Test {k}: {v:.4f}" if k == 'loss' else f"  Test {k}: {v*100:.2f}%")
    mprint(f"  Best epoch: {best_epoch}")

    if logger:
        logger.finish()

    # ── DDP Cleanup ──────────────────────────────────────────────────
    if config.ddp:
        _cleanup_ddp()

    return {
        'model': model,
        'history': history,
        'test_results': test_results,
        'config': config,
    }


class _NullLogger:
    """No-op logger for non-main DDP processes."""
    def __getattr__(self, name):
        return lambda *a, **kw: None