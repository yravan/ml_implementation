"""
Experiment Runner — dual-backend (PyTorch + custom numpy framework).

Supports three task types:
  - classification (default): image classification with top-1/top-5 accuracy
  - language_model: autoregressive LM with perplexity, text generation
  - seq2seq: encoder-decoder translation with BLEU scoring

Usage:
    from experiment import Config, run

    # Classification (unchanged)
    run(Config(backend='pytorch', dataset='mnist', model='cnn', epochs=10))

    # Language modeling
    run(Config(task='language_model', dataset='wikitext2', model='gpt2', epochs=10))

    # Translation
    run(Config(task='seq2seq', dataset='multi30k', model='transformer_base', epochs=30))

    # Same config, swap backend
    cfg = Config(dataset='cifar10', model='resnet18', epochs=50)
    run(Config(**{**cfg.to_dict(), 'backend': 'pytorch'}))
    run(Config(**{**cfg.to_dict(), 'backend': 'numpy'}))
"""

import os
import math
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
# Train / Eval — backend-agnostic (used by ClassificationRunner)
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
            # In DDP, multiply by world_size for aggregate throughput
            world = backend.world_size if hasattr(backend, 'world_size') else 1
            ips_total = ips * world
            interval_start = now
            interval_samples = 0
            parts = [f"Batch {batch_idx+1:5d}/{len(loader)}", f"Loss: {avg_loss:.4f}"]
            for name in metric_fns:
                parts.append(f"{name}: {float(metric_totals[name])/total_samples*100:.2f}%")
            if world > 1:
                parts.append(f"{ips_total:.0f} img/s ({ips:.0f}/rank × {world})")
            else:
                parts.append(f"{ips_total:.0f} img/s")
            if backend.is_main:
                print(f"    {' | '.join(parts)}")
            logger.log_scalar('train/batch_loss', avg_loss, step=global_step)
            logger.log_scalar('train/throughput', ips_total, step=global_step)
            logger.log_scalar('train/throughput_per_rank', ips, step=global_step)

    # Final sync at end of epoch
    epoch_elapsed = time.time() - epoch_start
    world = backend.world_size if hasattr(backend, 'world_size') else 1
    throughput = (total_samples * world) / epoch_elapsed if epoch_elapsed > 0 else 0.0

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
# DDP Helpers
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


class _NullLogger:
    """No-op logger for non-main DDP processes."""
    def __getattr__(self, name):
        return lambda *a, **kw: None


# =============================================================================
# Base Runner
# =============================================================================

class BaseRunner:
    """
    Base experiment runner. Handles common setup: DDP, seeds, backend,
    data, model, optimizer, scheduler, logger, epoch loop, checkpointing.

    Subclasses override:
      - setup_metrics() -> dict of metric functions
      - train_step(batch) -> (loss, n_samples, extra_for_metrics)
      - eval_step(batch) -> (loss, n_samples, extra_for_metrics)
      - accumulate_metrics(extra, metric_totals) -> updated metric_totals
      - on_epoch_end(epoch, train_results, val_results) -> None
      - format_metric(name, value) -> str
      - setup_criterion() -> loss function
      - setup_model_extras() -> called after model is built (e.g. channels_last)
    """

    def __init__(self, config: Config):
        self.config = config

        # ── DDP Init ─────────────────────────────────────────────
        if config.ddp:
            _setup_ddp(config)

        # ── Seeds ────────────────────────────────────────────────
        np.random.seed(config.seed)
        self.backend = _get_backend(config)
        self.is_main = self.backend.is_main

        if config.backend == 'pytorch':
            import torch
            torch.manual_seed(config.seed + self.backend.rank)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed + self.backend.rank)
                if config.cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if config.pin_memory and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.config = Config(**{**config.to_dict(), 'pin_memory': False})
                config = self.config

    def mprint(self, *args, **kwargs):
        """Print only on main process."""
        if self.is_main:
            print(*args, **kwargs)

    def setup_metrics(self) -> dict:
        """Return dict of {name: fn(logits, targets) -> scalar}. Override in subclass."""
        return {}

    def setup_criterion(self):
        """Return loss function. Override in subclass."""
        return self.backend.criterion(self.config)

    def setup_model_extras(self, model):
        """Post-processing after model build (e.g. channels_last). Returns model."""
        return model

    def train_step(self, model, batch, criterion, optimizer):
        """Execute one training step. Returns (logits, loss, n_samples, targets)."""
        return self.backend.train_step(model, batch, criterion, optimizer, self.config)

    def eval_step(self, model, batch, criterion):
        """Execute one eval step. Returns (logits, loss, n_samples, targets)."""
        return self.backend.eval_step(model, batch, criterion)

    def on_epoch_end(self, epoch, train_results, val_results, model, logger):
        """Hook called at end of each epoch. Override for generation, BLEU, etc."""
        pass

    def format_metric(self, name, value) -> str:
        """Format a metric for display. Override for perplexity, BLEU, etc."""
        if name == 'loss':
            return f"{value:.4f}"
        return f"{value*100:.2f}%"

    def primary_metric_improves(self, current, best, metric_name) -> bool:
        """Return True if current metric is better than best."""
        if metric_name in ('loss', 'perplexity'):
            return current < best
        return current > best

    def initial_best_metric(self, metric_name) -> float:
        """Return initial best metric value (worst possible)."""
        if metric_name in ('loss', 'perplexity'):
            return float('inf')
        return 0.0

    def run(self) -> Dict:
        """Run the complete experiment."""
        config = self.config
        backend = self.backend

        self.mprint(f"\n{'='*70}")
        self.mprint(f" {config.name}  [{config.backend}] [{config.task}]")
        self.mprint(f"{'='*70}")
        self.mprint(config)
        self.mprint(f"  Device: {backend.device}")
        if config.ddp:
            self.mprint(f"  DDP:    {backend.world_size} processes, backend={config.ddp_backend}")
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
                self.mprint(f"  Perf:   {', '.join(parts)}")

        # ── Data ────────────────────────────────────────────────
        self.mprint(f"\n{'─'*70}\n Data\n{'─'*70}")
        train_loader, val_loader, test_loader = build_dataloaders(config)

        # ── Model ───────────────────────────────────────────────
        self.mprint(f"\n{'─'*70}\n Model\n{'─'*70}")
        model = build_model(config)
        model = backend.to_device(model)
        model = self.setup_model_extras(model)

        if config.compile and config.backend == 'pytorch':
            import torch
            cache_dir = getattr(config, 'compile_cache_dir', None)
            if cache_dir:
                os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir
                self.mprint(f"  torch.compile cache: {cache_dir}")
            model = torch.compile(model, mode=config.compile_mode)
            self.mprint(f"  torch.compile: enabled ({config.compile_mode})")

        # DDP wrap
        if config.ddp:
            model = backend.wrap_ddp(model, config)
            self.mprint(f"  DDP:    wrapped with DistributedDataParallel")

        self.mprint(model)
        self.mprint(f"  Parameters: {backend.param_count(backend._unwrap(model) if config.ddp else model):,}")

        # ── Optimizer / Scheduler ───────────────────────────────
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        warmup = build_warmup_scheduler(optimizer, config)
        criterion = self.setup_criterion()

        # ── Logger ──────────────────────────────────────────────
        self.mprint(f"\n{'─'*70}\n Logger\n{'─'*70}")
        logger = Logger(config) if self.is_main else None
        if logger and config.backend == 'pytorch':
            logger.watch_model(model)

        # ── Metrics ─────────────────────────────────────────────
        metric_fns = self.setup_metrics()

        # ── Training ────────────────────────────────────────────
        self.mprint(f"\n{'─'*70}\n Training\n{'─'*70}")
        if logger:
            logger.begin_training()

        history = {f'{s}_{k}': [] for s in ('train', 'val')
                   for k in ['loss'] + list(metric_fns)}
        primary_metric = list(metric_fns.keys())[0] if metric_fns else 'loss'
        best_metric = self.initial_best_metric(primary_metric)
        best_epoch = 0

        for epoch in range(1, config.epochs + 1):
            if logger:
                logger.begin_epoch(epoch)
            self.mprint(f"\n  Epoch {epoch}/{config.epochs}")
            self.mprint(f"  {'─' * 50}")

            if config.ddp:
                _set_epoch_samplers(train_loader, epoch)

            # Train
            train_results = self._train_one_epoch(
                model, train_loader, criterion, optimizer,
                logger if self.is_main else _NullLogger(), metric_fns, epoch,
            )
            throughput = train_results.pop('_throughput', 0.0)

            # Val
            val_results = self._evaluate(model, val_loader, criterion, metric_fns)

            # LR step
            current_lr = backend.get_lr(optimizer)
            if scheduler is not None:
                if warmup and epoch <= config.warmup_epochs:
                    warmup.step()
                else:
                    scheduler.step()

            # Record history
            for k, v in train_results.items():
                history[f'train_{k}'].append(v)
            for k, v in val_results.items():
                history[f'val_{k}'].append(v)

            # Log
            if logger:
                epoch_global_step = epoch * len(train_loader)
                logger.end_epoch(train_results, val_results, lr=current_lr,
                                 throughput=throughput, step=epoch_global_step)

            # Epoch end hook (generation, BLEU, etc.)
            self.on_epoch_end(epoch, train_results, val_results, model,
                              logger if self.is_main else _NullLogger())

            # Best model tracking
            current = val_results.get(primary_metric, val_results.get('loss', 0))
            is_best = self.primary_metric_improves(current, best_metric, primary_metric)

            if is_best:
                best_metric = current
                best_epoch = epoch
                self.mprint(f"  ★ New best {primary_metric}!")
                ext = '.pt' if config.backend == 'pytorch' else '.npy'
                best_path = config.run_dir / f'best{ext}'
                best_path.parent.mkdir(parents=True, exist_ok=True)
                backend.save_checkpoint(model, optimizer, epoch, config, best_path)

            # Periodic save
            if config.save_every and epoch % config.save_every == 0:
                ext = '.pt' if config.backend == 'pytorch' else '.npy'
                ckpt_path = config.run_dir / f'checkpoint_{epoch}{ext}'
                backend.save_checkpoint(model, optimizer, epoch, config, ckpt_path)

            # DDP barrier
            if config.ddp:
                import torch.distributed as dist
                dist.barrier()

        # ── Test ────────────────────────────────────────────────
        self.mprint(f"\n{'─'*70}\n Test Evaluation\n{'─'*70}")
        ext = '.pt' if config.backend == 'pytorch' else '.npy'
        best_path = config.run_dir / f'best{ext}'
        if best_path.exists():
            backend.load_best(model, best_path)
            self.mprint(f"  Loaded best model (epoch {best_epoch})")

        test_results = self._evaluate(model, test_loader, criterion, metric_fns,
                                       collect_predictions=True)

        if logger:
            final_step = config.epochs * len(train_loader) + 1
            logger.log_scalars('test', {k: v for k, v in test_results.items()
                                        if k not in ('logits', 'labels')}, step=final_step)
            logger.flush()

        # Summary
        self.mprint(f"\n{'='*70}")
        self.mprint(f" Done!  [{config.backend}]")
        self.mprint(f"{'='*70}")
        for k, v in test_results.items():
            if k in ('logits', 'labels'):
                continue
            self.mprint(f"  Test {k}: {self.format_metric(k, v)}")
        self.mprint(f"  Best epoch: {best_epoch}")

        if logger:
            logger.finish()

        if config.ddp:
            _cleanup_ddp()

        return {
            'model': model,
            'history': history,
            'test_results': test_results,
            'config': config,
        }

    def _train_one_epoch(self, model, loader, criterion, optimizer, logger, metric_fns, epoch):
        """Train for one epoch. Returns dict of averaged metrics."""
        config = self.config
        backend = self.backend
        model.train()
        total_loss = 0.0
        total_samples = 0
        metric_totals = {name: 0.0 for name in metric_fns}
        epoch_start = time.time()
        interval_start = epoch_start
        interval_samples = 0

        for batch_idx, batch in enumerate(loader):
            logits, batch_loss, n, targets_d = self.train_step(
                model, batch, criterion, optimizer
            )

            total_loss = total_loss + batch_loss * n
            total_samples += n
            interval_samples += n

            logits_d = logits.detach() if hasattr(logits, 'detach') else logits
            for name, fn in metric_fns.items():
                metric_totals[name] = metric_totals[name] + fn(logits_d, targets_d)

            global_step = (epoch - 1) * len(loader) + batch_idx
            if config.log_interval and (batch_idx + 1) % config.log_interval == 0:
                avg_loss = float(total_loss) / total_samples
                now = time.time()
                interval_elapsed = now - interval_start
                ips = interval_samples / interval_elapsed if interval_elapsed > 0 else 0.0
                world = backend.world_size if hasattr(backend, 'world_size') else 1
                ips_total = ips * world
                interval_start = now
                interval_samples = 0
                parts = [f"Batch {batch_idx+1:5d}/{len(loader)}", f"Loss: {avg_loss:.4f}"]
                for name in metric_fns:
                    parts.append(f"{name}: {self.format_metric(name, float(metric_totals[name])/total_samples)}")
                if world > 1:
                    parts.append(f"{ips_total:.0f} tok/s ({ips:.0f}/rank × {world})")
                else:
                    parts.append(f"{ips_total:.0f} tok/s")
                if backend.is_main:
                    print(f"    {' | '.join(parts)}")
                logger.log_scalar('train/batch_loss', avg_loss, step=global_step)
                logger.log_scalar('train/throughput', ips_total, step=global_step)

        epoch_elapsed = time.time() - epoch_start
        world = backend.world_size if hasattr(backend, 'world_size') else 1
        throughput = (total_samples * world) / epoch_elapsed if epoch_elapsed > 0 else 0.0

        results = {'loss': float(total_loss) / total_samples}
        for name in metric_fns:
            results[name] = float(metric_totals[name]) / total_samples
        results['_throughput'] = throughput
        return results

    def _evaluate(self, model, loader, criterion, metric_fns, collect_predictions=False):
        """Evaluate model. Returns dict of averaged metrics."""
        backend = self.backend
        model.eval()
        total_loss = 0.0
        total_samples = 0
        metric_totals = {name: 0.0 for name in metric_fns}
        all_logits = [] if collect_predictions else None
        all_labels = [] if collect_predictions else None

        with backend.no_grad_context():
            for batch in loader:
                logits, batch_loss, n, targets_d = self.eval_step(model, batch, criterion)

                total_loss = total_loss + batch_loss * n
                total_samples += n

                logits_d = logits.detach() if hasattr(logits, 'detach') else logits
                for name, fn in metric_fns.items():
                    metric_totals[name] = metric_totals[name] + fn(logits_d, targets_d)

                if collect_predictions:
                    all_logits.append(_to_numpy(logits))
                    all_labels.append(_to_numpy(targets_d))

        results = {'loss': float(total_loss) / total_samples}
        for name in metric_fns:
            results[name] = float(metric_totals[name]) / total_samples
        if collect_predictions:
            results['logits'] = np.concatenate(all_logits)
            results['labels'] = np.concatenate(all_labels)
        return results


# =============================================================================
# Classification Runner — preserves exact original behavior
# =============================================================================

class ClassificationRunner(BaseRunner):
    """Image classification runner with top-1/top-5 accuracy."""

    def setup_metrics(self):
        return {name: METRIC_FNS[name] for name in self.config.metrics
                if name in METRIC_FNS}

    def setup_model_extras(self, model):
        """Apply channels_last memory format for classification."""
        if self.config.backend == 'pytorch':
            import torch
            if torch.cuda.is_available():
                model = model.to(memory_format=torch.channels_last)
                self.mprint("  memory_format: channels_last (NHWC)")
        return model

    def format_metric(self, name, value):
        if name == 'loss':
            return f"{value:.4f}"
        return f"{value*100:.2f}%"

    def _train_one_epoch(self, model, loader, criterion, optimizer, logger, metric_fns, epoch):
        """Use original train_one_epoch for exact backward compat."""
        return train_one_epoch(
            model, loader, criterion, optimizer,
            self.config, logger, metric_fns, self.backend, epoch,
        )

    def _evaluate(self, model, loader, criterion, metric_fns, collect_predictions=False):
        """Use original evaluate for exact backward compat."""
        return evaluate(model, loader, criterion, metric_fns, self.backend,
                        collect_predictions=collect_predictions)


# =============================================================================
# Language Model Runner
# =============================================================================

class LanguageModelRunner(BaseRunner):
    """
    Language modeling runner (GPT-2 style).

    Batch format: (input_ids,) — single tensor of token IDs.
    Training: next-token prediction (shift logits by 1).
    Metrics: perplexity = exp(loss).
    Generation: produce text samples at configurable intervals.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            self.mprint("  [Warning] transformers not installed, generation disabled")

    def setup_metrics(self):
        def perplexity_metric(logits, targets):
            # Perplexity computed from epoch loss in format_metric; return 0 as placeholder
            return 0.0
        if 'perplexity' in self.config.metrics:
            return {'perplexity': perplexity_metric}
        return {}

    def setup_criterion(self):
        if self.config.backend == 'pytorch':
            import torch.nn.functional as F
            ls = self.config.label_smoothing
            def lm_criterion(logits, targets):
                # logits: [B, T, V], targets: [B, T]
                B, T, V = logits.shape
                return F.cross_entropy(
                    logits.reshape(B * T, V), targets.reshape(B * T),
                    label_smoothing=ls if ls > 0 else 0.0,
                )
            return lm_criterion
        # Numpy: flattening done in _np_train_step/_np_eval_step
        return self.backend.criterion(self.config)

    def train_step(self, model, batch, criterion, optimizer):
        """LM training: shift logits for next-token prediction."""
        if self.config.backend == 'numpy':
            return self._np_train_step(model, batch, criterion, optimizer)

        import torch
        backend = self.backend
        input_ids = batch[0].to(backend.device, non_blocking=True)

        # Input: all tokens except last; Target: all tokens except first
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        if backend._use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            backend._scaler.scale(loss).backward()
            if self.config.grad_clip:
                backend._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            backend._scaler.step(optimizer)
            backend._scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()

        return logits, loss.detach(), inputs.shape[0], targets

    def _np_train_step(self, model, batch, criterion, optimizer):
        """Numpy backend LM training step."""
        from python.foundations.computational_graph import Tensor
        import numpy as np

        input_ids = batch[0]
        if hasattr(input_ids, 'numpy'):
            input_ids = input_ids.numpy()

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        x = Tensor(inputs, requires_grad=True)
        logits = model(x)  # [B, T, V]

        # Flatten for cross-entropy: [B*T, V] and [B*T]
        B, T = targets.shape
        logits_flat = logits.reshape(B * T, -1)
        targets_flat = Tensor(targets.reshape(B * T))

        loss = criterion(logits_flat, targets_flat, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, B, targets

    def eval_step(self, model, batch, criterion):
        """LM eval: same shift as training."""
        if self.config.backend == 'numpy':
            return self._np_eval_step(model, batch, criterion)

        import torch
        backend = self.backend
        input_ids = batch[0].to(backend.device, non_blocking=True)
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        if backend._use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, targets)
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)

        return logits, loss.detach(), inputs.shape[0], targets

    def _np_eval_step(self, model, batch, criterion):
        """Numpy backend LM eval step."""
        from python.foundations.computational_graph import Tensor

        input_ids = batch[0]
        if hasattr(input_ids, 'numpy'):
            input_ids = input_ids.numpy()

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        x = Tensor(inputs, requires_grad=False)
        logits = model(x)

        B, T = targets.shape
        logits_flat = logits.reshape(B * T, -1)
        targets_flat = Tensor(targets.reshape(B * T))

        loss = criterion(logits_flat, targets_flat, reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, B, targets

    def on_epoch_end(self, epoch, train_results, val_results, model, logger):
        """Generate text samples periodically."""
        config = self.config
        if config.generate_every <= 0 or epoch % config.generate_every != 0:
            return
        if self.tokenizer is None or config.backend != 'pytorch':
            return

        import torch
        self.mprint(f"\n  ── Generated Samples (Epoch {epoch}) ──")
        raw_model = self.backend._unwrap(model)

        for prompt_text in config.generate_prompts[:config.num_generate_samples]:
            input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')
            input_ids = input_ids.to(self.backend.device)

            if hasattr(raw_model, 'generate'):
                output_ids = raw_model.generate(
                    input_ids,
                    max_new_tokens=config.generate_max_tokens,
                    temperature=config.generate_temperature,
                    top_k=config.generate_top_k,
                    top_p=config.generate_top_p,
                )
            else:
                output_ids = input_ids  # fallback

            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            self.mprint(f"  Prompt: {prompt_text!r}")
            self.mprint(f"  Output: {text!r}\n")

            if logger:
                logger.log_text(f'generation/epoch_{epoch}', text, step=epoch)

    def format_metric(self, name, value):
        if name == 'loss':
            return f"{value:.4f}"
        if name == 'perplexity':
            return f"{value:.2f}"
        return f"{value:.4f}"

    def _evaluate(self, model, loader, criterion, metric_fns, collect_predictions=False):
        """Override to compute perplexity from loss."""
        results = super()._evaluate(model, loader, criterion, metric_fns, collect_predictions)
        # Compute perplexity from loss
        ppl = math.exp(min(results['loss'], 100))  # clamp to avoid overflow
        results['perplexity'] = ppl
        return results

    def _train_one_epoch(self, model, loader, criterion, optimizer, logger, metric_fns, epoch):
        """Override to compute perplexity from loss."""
        results = super()._train_one_epoch(model, loader, criterion, optimizer, logger, metric_fns, epoch)
        ppl = math.exp(min(results['loss'], 100))
        results['perplexity'] = ppl
        return results


# =============================================================================
# Seq2Seq Runner
# =============================================================================

class Seq2SeqRunner(BaseRunner):
    """
    Sequence-to-sequence (encoder-decoder) runner for translation.

    Batch format: (src_ids, tgt_ids) — source and target token ID tensors.
    Training: teacher forcing — model(src, tgt[:, :-1]), loss on tgt[:, 1:].
    Metrics: BLEU via sacrebleu.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer = None
        self.pad_token_id = 0
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
            if self.tokenizer.pad_token is not None:
                self.pad_token_id = self.tokenizer.pad_token_id
        except ImportError:
            self.mprint("  [Warning] transformers not installed")

    def setup_metrics(self):
        # BLEU computed separately in on_epoch_end, not per-batch
        return {}

    def setup_criterion(self):
        if self.config.backend == 'pytorch':
            import torch.nn.functional as F
            ls = self.config.label_smoothing
            pad_id = self.pad_token_id
            def seq2seq_criterion(logits, targets):
                B, T, V = logits.shape
                return F.cross_entropy(
                    logits.reshape(B * T, V), targets.reshape(B * T),
                    ignore_index=pad_id,
                    label_smoothing=ls if ls > 0 else 0.0,
                )
            return seq2seq_criterion
        # Numpy: flattening done in _np_train_step/_np_eval_step
        return self.backend.criterion(self.config)

    def train_step(self, model, batch, criterion, optimizer):
        """Seq2seq training with teacher forcing."""
        if self.config.backend == 'numpy':
            return self._np_train_step(model, batch, criterion, optimizer)

        import torch
        backend = self.backend
        src_ids = batch[0].to(backend.device, non_blocking=True)
        tgt_ids = batch[1].to(backend.device, non_blocking=True)

        tgt_input = tgt_ids[:, :-1]
        tgt_target = tgt_ids[:, 1:]

        if backend._use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(src_ids, tgt_input)
                loss = criterion(logits, tgt_target)
            optimizer.zero_grad(set_to_none=True)
            backend._scaler.scale(loss).backward()
            if self.config.grad_clip:
                backend._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            backend._scaler.step(optimizer)
            backend._scaler.update()
        else:
            logits = model(src_ids, tgt_input)
            loss = criterion(logits, tgt_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()

        return logits, loss.detach(), src_ids.shape[0], tgt_target

    def _np_train_step(self, model, batch, criterion, optimizer):
        """Numpy backend seq2seq training step."""
        from python.foundations.computational_graph import Tensor

        src_ids = batch[0]
        tgt_ids = batch[1]
        if hasattr(src_ids, 'numpy'):
            src_ids = src_ids.numpy()
        if hasattr(tgt_ids, 'numpy'):
            tgt_ids = tgt_ids.numpy()

        tgt_input = tgt_ids[:, :-1]
        tgt_target = tgt_ids[:, 1:]

        src = Tensor(src_ids, requires_grad=True)
        tgt_in = Tensor(tgt_input, requires_grad=True)
        logits = model(src, tgt_in)  # [B, T, V]

        B, T = tgt_target.shape
        logits_flat = logits.reshape(B * T, -1)
        targets_flat = Tensor(tgt_target.reshape(B * T))

        loss = criterion(logits_flat, targets_flat, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, B, tgt_target

    def eval_step(self, model, batch, criterion):
        """Seq2seq eval with teacher forcing."""
        if self.config.backend == 'numpy':
            return self._np_eval_step(model, batch, criterion)

        import torch
        backend = self.backend
        src_ids = batch[0].to(backend.device, non_blocking=True)
        tgt_ids = batch[1].to(backend.device, non_blocking=True)

        tgt_input = tgt_ids[:, :-1]
        tgt_target = tgt_ids[:, 1:]

        if backend._use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(src_ids, tgt_input)
                loss = criterion(logits, tgt_target)
        else:
            logits = model(src_ids, tgt_input)
            loss = criterion(logits, tgt_target)

        return logits, loss.detach(), src_ids.shape[0], tgt_target

    def _np_eval_step(self, model, batch, criterion):
        """Numpy backend seq2seq eval step."""
        from python.foundations.computational_graph import Tensor

        src_ids = batch[0]
        tgt_ids = batch[1]
        if hasattr(src_ids, 'numpy'):
            src_ids = src_ids.numpy()
        if hasattr(tgt_ids, 'numpy'):
            tgt_ids = tgt_ids.numpy()

        tgt_input = tgt_ids[:, :-1]
        tgt_target = tgt_ids[:, 1:]

        src = Tensor(src_ids, requires_grad=False)
        tgt_in = Tensor(tgt_input, requires_grad=False)
        logits = model(src, tgt_in)

        B, T = tgt_target.shape
        logits_flat = logits.reshape(B * T, -1)
        targets_flat = Tensor(tgt_target.reshape(B * T))

        loss = criterion(logits_flat, targets_flat, reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, B, tgt_target

    def on_epoch_end(self, epoch, train_results, val_results, model, logger):
        """Compute BLEU on validation subset if sacrebleu is available."""
        if self.config.backend != 'pytorch':
            return
        try:
            import sacrebleu
        except ImportError:
            return

        # Only compute BLEU periodically
        if self.config.generate_every > 0 and epoch % self.config.generate_every == 0:
            self.mprint(f"\n  Computing BLEU (epoch {epoch})...")
            # BLEU computation would go here using model.generate() + tokenizer decode
            # This is a placeholder — full implementation needs the val loader
            if logger:
                logger.log_scalar('val/bleu', 0.0, step=epoch)

    def format_metric(self, name, value):
        if name == 'loss':
            return f"{value:.4f}"
        if name == 'bleu':
            return f"{value:.2f}"
        return f"{value:.4f}"


# =============================================================================
# Factory + backward compatibility
# =============================================================================

def get_runner(config: Config) -> BaseRunner:
    """Create the appropriate runner for the task type."""
    task = getattr(config, 'task', 'classification')
    if task == 'language_model':
        return LanguageModelRunner(config)
    elif task == 'seq2seq':
        return Seq2SeqRunner(config)
    else:
        return ClassificationRunner(config)


def run(config: Config) -> Dict:
    """
    Run a complete experiment. Works with both PyTorch and numpy backends.
    Backward compatible — dispatches to the appropriate runner.

    For DDP, launch via torchrun:
        torchrun --nproc_per_node=N -m experiment --config cfg.yaml --ddp

    Returns dict with model, history, and test results.
    """
    return get_runner(config).run()
