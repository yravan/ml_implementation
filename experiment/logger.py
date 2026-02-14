"""
Unified Experiment Logger.

Supports TensorBoard, Weights & Biases, and console output.
All three can run simultaneously with logger='all'.

Usage:
    logger = Logger(config)
    logger.log_scalar('train/loss', 0.5, step=100)
    logger.log_scalars('val', {'loss': 0.3, 'top1': 0.92}, step=200)
    logger.log_image('samples', img_array, step=300)
    logger.log_histogram('conv1/weights', weight_array, step=400)
    logger.finish()
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


class Logger:
    """Unified logging interface."""

    def __init__(self, config: "Config"):
        self.config = config
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._tb_writer = None
        self._wandb_run = None
        self._step = 0
        self._epoch = 0
        self._epoch_start = None
        self._train_start = None

        backends = config.logger.lower()
        self._use_tb = backends in ('tensorboard', 'tb', 'all')
        self._use_wandb = backends in ('wandb', 'wb', 'all')
        self._use_console = backends in ('console', 'all') or True  # always console

        if self._use_tb:
            self._init_tensorboard()
        if self._use_wandb:
            self._init_wandb()

        # Save config
        config.save(self.run_dir / 'config.json')

    # ── Init ─────────────────────────────────────────────────────────

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.run_dir / 'tb'
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"  TensorBoard: {tb_dir}")
            print(f"  Run: tensorboard --logdir {tb_dir}")
        except ImportError:
            print("  [Warning] tensorboard not installed. pip install tensorboard")
            self._use_tb = False

    def _init_wandb(self):
        try:
            import wandb
            wandb.login(key=self.config.wandb_api)
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.name,
                config=self.config.to_dict(),
                dir=str(self.run_dir),
                reinit=True,
            )
            print(f"  W&B: {wandb.run.url}")
        except ImportError:
            print("  [Warning] wandb not installed. pip install wandb")
            self._use_wandb = False

    # ── Scalars ──────────────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a single scalar value."""
        if step is None:
            step = self._step

        if self._use_tb and self._tb_writer:
            self._tb_writer.add_scalar(tag, value, step)

        if self._use_wandb and self._wandb_run:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, prefix: str, values: Dict[str, float],
                    step: Optional[int] = None):
        """Log multiple scalars with a common prefix."""
        if step is None:
            step = self._step

        for name, val in values.items():
            self.log_scalar(f"{prefix}/{name}", val, step)

    # ── Images ───────────────────────────────────────────────────────

    def log_image(self, tag: str, img: np.ndarray, step: Optional[int] = None):
        """Log an image. Expects (H, W, C) or (C, H, W) float/uint8."""
        if step is None:
            step = self._step

        if self._use_tb and self._tb_writer:
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                fmt = 'CHW'
            else:
                fmt = 'HWC'
            self._tb_writer.add_image(tag, img, step, dataformats=fmt)

        if self._use_wandb and self._wandb_run:
            import wandb
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            wandb.log({tag: wandb.Image(img)}, step=step)

    # ── Histograms ───────────────────────────────────────────────────

    def log_histogram(self, tag: str, values: np.ndarray,
                      step: Optional[int] = None):
        """Log a histogram of values."""
        if step is None:
            step = self._step

        if self._use_tb and self._tb_writer:
            self._tb_writer.add_histogram(tag, values, step)

        if self._use_wandb and self._wandb_run:
            import wandb
            wandb.log({tag: wandb.Histogram(values)}, step=step)

    # ── Model ────────────────────────────────────────────────────────

    def log_model_params(self, model, step: Optional[int] = None):
        """Log weight/gradient histograms for all parameters."""
        for name, p in model.named_parameters():
            data = p.detach().cpu().numpy() if hasattr(p, 'cpu') else p.data
            self.log_histogram(f"weights/{name}", data, step)
            if p.grad is not None:
                g = p.grad.detach().cpu().numpy() if hasattr(p.grad, 'cpu') else p.grad.data
                self.log_histogram(f"gradients/{name}", g, step)

    def watch_model(self, model):
        """W&B model watching (auto-logs gradients/params)."""
        if self._use_wandb and self._wandb_run:
            import wandb
            wandb.watch(model, log='all', log_freq=100)

    # ── Epoch Tracking ───────────────────────────────────────────────

    def begin_training(self):
        self._train_start = time.time()

    def begin_epoch(self, epoch: int):
        self._epoch = epoch
        self._epoch_start = time.time()

    def end_epoch(self, train_results: Dict[str, float],
                  val_results: Dict[str, float],
                  lr: Optional[float] = None,
                  throughput: Optional[float] = None):
        """Log end-of-epoch results to all backends."""
        epoch = self._epoch
        epoch_time = time.time() - self._epoch_start if self._epoch_start else 0

        # Log to backends
        self.log_scalars('train', train_results, step=epoch)
        self.log_scalars('val', val_results, step=epoch)
        if lr is not None:
            self.log_scalar('lr', lr, step=epoch)
        self.log_scalar('epoch_time', epoch_time, step=epoch)
        if throughput is not None:
            self.log_scalar('train/throughput', throughput, step=epoch)

        # Console output
        if self._use_console:
            parts_train = [f"Loss: {train_results.get('loss', 0):.4f}"]
            parts_val = [f"Loss: {val_results.get('loss', 0):.4f}"]
            for k, v in train_results.items():
                if k != 'loss':
                    parts_train.append(f"{k}: {v*100:.2f}%")
            for k, v in val_results.items():
                if k != 'loss':
                    parts_val.append(f"{k}: {v*100:.2f}%")

            print(f"  Train | {' | '.join(parts_train)}")
            print(f"  Val   | {' | '.join(parts_val)}")
            if lr is not None:
                print(f"  LR: {lr:.6f}")
            timing = f"  Time: {epoch_time:.1f}s"
            if throughput is not None:
                timing += f" | {throughput:.0f} img/s"
            print(timing)

    # ── Text / Artifacts ─────────────────────────────────────────────

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        if self._use_tb and self._tb_writer:
            self._tb_writer.add_text(tag, text, step or self._step)

    def log_artifact(self, path: str, name: Optional[str] = None):
        """Log a file as an artifact (W&B only)."""
        if self._use_wandb and self._wandb_run:
            import wandb
            artifact = wandb.Artifact(name or Path(path).stem, type='model')
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    # ── Housekeeping ─────────────────────────────────────────────────

    def set_step(self, step: int):
        self._step = step

    def flush(self):
        if self._tb_writer:
            self._tb_writer.flush()

    def finish(self):
        """Clean up all backends."""
        total = time.time() - self._train_start if self._train_start else 0
        if total:
            print(f"\n  Total training time: {total:.1f}s")

        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            import wandb
            wandb.finish()

        print(f"  Logs saved to: {self.run_dir}")