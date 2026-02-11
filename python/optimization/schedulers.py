"""
Learning Rate Schedulers
========================

Algorithms for adjusting learning rate during training.

Learning rate is arguably the most important hyperparameter in deep learning.
Schedulers systematically adjust it throughout training to improve convergence.

Why schedule learning rate?
1. **Fast early learning**: Higher LR explores quickly
2. **Fine-grained late learning**: Lower LR refines solution
3. **Escape local minima**: Cyclic schedules can help exploration
4. **Training stability**: Warmup prevents early divergence

Categories of schedulers:

1. **Step-based**: Discrete jumps at milestones
   - StepLR: Multiply by gamma every N steps
   - MultiStepLR: Multiply at specific milestones

2. **Continuous decay**:
   - ExponentialLR: Multiply by gamma every step
   - LinearLR: Linear interpolation
   - PolynomialLR: Polynomial decay

3. **Cyclic**:
   - CosineAnnealingLR: Cosine curve decay
   - CyclicLR: Triangular or other cyclic patterns
   - OneCycleLR: Single cosine cycle (super-convergence)

4. **Adaptive**:
   - ReduceLROnPlateau: Reduce when metric stops improving

5. **Warmup**:
   - LinearWarmup: Gradual increase from 0
   - Can be combined with any other scheduler

Theory
------
The learning rate affects convergence guarantees:
- Too large: Divergence, oscillation
- Too small: Slow convergence, stuck in local minima
- "Just right": Fast convergence to good solution

Key theoretical results:
- SGD with decreasing LR: Σ lr = ∞, Σ lr² < ∞ ensures convergence
- Adam is less sensitive to LR than SGD
- Warmup helps with adaptive optimizers early in training

Empirical findings:
- Cosine schedule often works best in practice
- Warmup crucial for Transformers
- OneCycleLR enables faster training (super-convergence)
- ReduceLROnPlateau good when unsure about schedule

References
----------
- "Cyclical Learning Rates for Training Neural Networks" Smith (2015)
  https://arxiv.org/abs/1506.01186
- "Super-Convergence" Smith & Topin (2018)
  https://arxiv.org/abs/1708.07120
- "SGDR: Stochastic Gradient Descent with Warm Restarts" Loshchilov & Hutter (2016)
  https://arxiv.org/abs/1608.03983
- "A disciplined approach to neural network hyper-parameters" Smith (2018)
  https://arxiv.org/abs/1803.09820

Implementation Notes
--------------------
- Schedulers modify optimizer.lr (or equivalent)
- step() is called once per epoch or once per batch (varies)
- Many schedulers can be chained (SequentialLR, ChainedScheduler)
- Warmup is typically separate and applied first
"""

# Implementation Status: NOT STARTED
# Complexity: Easy to Medium
# Prerequisites: Optimizer classes

import numpy as np
from typing import List, Optional, Callable, Dict, Any, Union
import math


# =============================================================================
# Base Scheduler Class
# =============================================================================

class LRScheduler:
    """
    Base class for all learning rate schedulers.

    Schedulers adjust the learning rate based on:
    - Number of epochs/steps completed
    - Validation metrics (adaptive schedulers)
    - Custom criteria

    Example:
        >>> optimizer = Adam(params, lr=0.001)
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     scheduler.step()
    """

    def __init__(self, optimizer: 'Optimizer', last_epoch: int = -1):
        """
        Initialize scheduler.

        Args:
            optimizer: Wrapped optimizer
            last_epoch: Index of last epoch (-1 = starting fresh)
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.defaults.get('lr', 0.001)
        self._step_count = 0

    def get_lr(self) -> float:
        """Compute learning rate for current step."""
        raise NotImplementedError("Subclasses must implement get_lr()")

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Update learning rate.

        Args:
            epoch: Current epoch (optional, increments automatically if None)
        """
        if epoch is None:
            self._step_count += 1
            self.last_epoch += 1
        else:
            self._step_count = epoch
            self.last_epoch = epoch

        lr = self.get_lr()
        self.optimizer.set_lr(lr)

    def get_last_lr(self) -> float:
        """Return last computed learning rate."""
        return self.get_lr()

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            'last_epoch': self.last_epoch,
            'base_lr': self.base_lr,
            '_step_count': self._step_count
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.last_epoch = state_dict['last_epoch']
        self.base_lr = state_dict['base_lr']
        self._step_count = state_dict['_step_count']


# =============================================================================
# Step-Based Schedulers
# =============================================================================

class StepLR(LRScheduler):
    """
    Decays learning rate by gamma every step_size epochs.

    lr = base_lr * gamma^(epoch // step_size)

    Simple and effective. Good default choice.

    Example:
        >>> # Decay by 0.1x every 30 epochs
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> # Epoch 0-29: lr=0.1, Epoch 30-59: lr=0.01, ...
    """

    def __init__(self, optimizer: 'Optimizer', step_size: int, gamma: float = 0.1,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            step_size: Period of learning rate decay (in epochs)
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement StepLR.get_lr()\n"
            "Hint:\n"
            "  num_decays = self.last_epoch // self.step_size\n"
            "  return self.base_lr * (self.gamma ** num_decays)"
        )


class MultiStepLR(LRScheduler):
    """
    Decays learning rate at specific epoch milestones.

    More flexible than StepLR - decay at arbitrary points.

    Example:
        >>> # Decay by 0.1x at epochs 30 and 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        >>> # Epoch 0-29: lr=0.1, Epoch 30-79: lr=0.01, Epoch 80+: lr=0.001
    """

    def __init__(self, optimizer: 'Optimizer', milestones: List[int],
                 gamma: float = 0.1, last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            milestones: List of epoch indices to decay at
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer, last_epoch)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement MultiStepLR.get_lr()\n"
            "Hint:\n"
            "  num_decays = sum(1 for m in self.milestones if self.last_epoch >= m)\n"
            "  return self.base_lr * (self.gamma ** num_decays)"
        )


# =============================================================================
# Continuous Decay Schedulers
# =============================================================================

class ExponentialLR(LRScheduler):
    """
    Decays learning rate by gamma every epoch.

    lr = base_lr * gamma^epoch

    Provides smooth, continuous decay.

    Example:
        >>> # Decay by 0.95x every epoch
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
    """

    def __init__(self, optimizer: 'Optimizer', gamma: float,
                 last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement ExponentialLR.get_lr()\n"
            "Hint:\n"
            "  return self.base_lr * (self.gamma ** self.last_epoch)"
        )


class LinearLR(LRScheduler):
    """
    Linear interpolation of learning rate.

    Linearly changes LR from start_factor*base_lr to end_factor*base_lr
    over total_iters steps.

    Useful for warmup (start_factor=0, end_factor=1) or
    linear decay (start_factor=1, end_factor=0).

    Example:
        >>> # Linear warmup over 5 epochs
        >>> scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    """

    def __init__(self, optimizer: 'Optimizer',
                 start_factor: float = 1.0/3,
                 end_factor: float = 1.0,
                 total_iters: int = 5,
                 last_epoch: int = -1):
        """
        Args:
            start_factor: Starting LR = base_lr * start_factor
            end_factor: Ending LR = base_lr * end_factor
            total_iters: Number of iterations to reach end_factor
        """
        super().__init__(optimizer, last_epoch)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement LinearLR.get_lr()\n"
            "Hint:\n"
            "  if self.last_epoch >= self.total_iters:\n"
            "      return self.base_lr * self.end_factor\n"
            "  \n"
            "  # Linear interpolation\n"
            "  progress = self.last_epoch / self.total_iters\n"
            "  factor = self.start_factor + progress * (self.end_factor - self.start_factor)\n"
            "  return self.base_lr * factor"
        )


class PolynomialLR(LRScheduler):
    """
    Polynomial learning rate decay.

    lr = (base_lr - end_lr) * (1 - progress)^power + end_lr

    Where progress = epoch / total_iters.

    Power controls decay shape:
    - power=1: Linear decay
    - power>1: Slower initial decay, faster later
    - power<1: Faster initial decay, slower later

    References:
        - Commonly used in semantic segmentation (DeepLab)
    """

    def __init__(self, optimizer: 'Optimizer',
                 total_iters: int,
                 power: float = 1.0,
                 last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.total_iters = total_iters
        self.power = power

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement PolynomialLR.get_lr()\n"
            "Hint:\n"
            "  if self.last_epoch >= self.total_iters:\n"
            "      return 0.0\n"
            "  \n"
            "  progress = self.last_epoch / self.total_iters\n"
            "  return self.base_lr * ((1 - progress) ** self.power)"
        )


# =============================================================================
# Cosine Schedulers
# =============================================================================

class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate schedule.

    lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2

    Provides smooth decay following a cosine curve. Very popular in practice.

    Math:
        The cosine goes from 1 to -1 over [0, π].
        (1 + cos(π * t/T)) / 2 goes from 1 to 0 over [0, T].
        This smoothly interpolates from base_lr to eta_min.

    References:
        - "SGDR: Stochastic Gradient Descent with Warm Restarts" (2016)
          https://arxiv.org/abs/1608.03983

    Example:
        >>> # Decay to 0 over 100 epochs using cosine curve
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
    """

    def __init__(self, optimizer: 'Optimizer', T_max: int,
                 eta_min: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of iterations (half a cosine cycle)
            eta_min: Minimum learning rate
        """
        super().__init__(optimizer, last_epoch)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement CosineAnnealingLR.get_lr()\n"
            "Hint:\n"
            "  # Cosine schedule formula\n"
            "  progress = self.last_epoch / self.T_max\n"
            "  cosine_factor = (1 + math.cos(math.pi * progress)) / 2\n"
            "  return self.eta_min + (self.base_lr - self.eta_min) * cosine_factor"
        )


class CosineAnnealingWarmRestarts(LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR).

    Periodically resets learning rate to initial value, following
    cosine decay within each restart period.

    The period can grow by factor T_mult after each restart.

    Math:
        Within each restart period:
        lr = eta_min + (base_lr - eta_min) * (1 + cos(π * T_cur / T_i)) / 2

        Where T_cur is time since last restart, T_i is current period length.

    Benefits:
        - Escapes local minima via periodic LR increases
        - Explores different parts of loss landscape
        - Often finds better solutions than monotonic decay

    References:
        - Loshchilov & Hutter "SGDR" (2016)
          https://arxiv.org/abs/1608.03983

    Example:
        >>> # Restart every 10 epochs, doubling period each time
        >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        >>> # Restarts at: 10, 30, 70, 150, ...
    """

    def __init__(self, optimizer: 'Optimizer',
                 T_0: int,
                 T_mult: int = 1,
                 eta_min: float = 0.0,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            T_0: Number of epochs for first restart
            T_mult: Factor to grow restart period by (1 = fixed period)
            eta_min: Minimum learning rate
        """
        super().__init__(optimizer, last_epoch)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement CosineAnnealingWarmRestarts.get_lr()\n"
            "Hint:\n"
            "  # Find which restart period we're in\n"
            "  if self.T_mult == 1:\n"
            "      T_cur = self.last_epoch % self.T_0\n"
            "      T_i = self.T_0\n"
            "  else:\n"
            "      # Geometric series: T_0 * (1 + T_mult + T_mult^2 + ...)\n"
            "      # Find n such that sum(T_0 * T_mult^k for k in 0..n-1) <= epoch\n"
            "      # ... compute T_cur and T_i ...\n"
            "  \n"
            "  # Cosine within current period\n"
            "  cosine_factor = (1 + math.cos(math.pi * T_cur / T_i)) / 2\n"
            "  return self.eta_min + (self.base_lr - self.eta_min) * cosine_factor"
        )


# =============================================================================
# Cyclic Schedulers
# =============================================================================

class CyclicLR(LRScheduler):
    """
    Cyclic learning rate policy.

    Cycles LR between base_lr and max_lr following a policy:
    - 'triangular': Simple triangle wave
    - 'triangular2': Triangle with halved amplitude each cycle
    - 'exp_range': Exponential decay of max_lr each cycle

    Benefits:
        - Can escape saddle points
        - May find wider minima (better generalization)
        - Useful for learning rate range finding

    References:
        - Smith "Cyclical Learning Rates for Training Neural Networks" (2015)
          https://arxiv.org/abs/1506.01186

    Example:
        >>> scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,
        ...                      step_size_up=2000, mode='triangular')
    """

    def __init__(self, optimizer: 'Optimizer',
                 base_lr: float,
                 max_lr: float,
                 step_size_up: int = 2000,
                 step_size_down: Optional[int] = None,
                 mode: str = 'triangular',
                 gamma: float = 1.0,
                 scale_fn: Optional[Callable[[int], float]] = None,
                 cycle_momentum: bool = True,
                 base_momentum: float = 0.8,
                 max_momentum: float = 0.9,
                 last_epoch: int = -1):
        """
        Args:
            base_lr: Initial learning rate (lower bound)
            max_lr: Maximum learning rate (upper bound)
            step_size_up: Steps in the increasing half of a cycle
            step_size_down: Steps in decreasing half (default = step_size_up)
            mode: 'triangular', 'triangular2', or 'exp_range'
            gamma: Decay factor for 'exp_range' mode
            scale_fn: Custom scaling function (overrides mode)
            cycle_momentum: Also cycle momentum (inversely to LR)
        """
        super().__init__(optimizer, last_epoch)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement CyclicLR.get_lr()\n"
            "Hint:\n"
            "  cycle_length = self.step_size_up + self.step_size_down\n"
            "  cycle = self.last_epoch // cycle_length\n"
            "  x = self.last_epoch % cycle_length\n"
            "  \n"
            "  if x < self.step_size_up:\n"
            "      # Increasing phase\n"
            "      progress = x / self.step_size_up\n"
            "  else:\n"
            "      # Decreasing phase\n"
            "      progress = 1 - (x - self.step_size_up) / self.step_size_down\n"
            "  \n"
            "  # Apply scaling based on mode\n"
            "  if self.mode == 'triangular':\n"
            "      scale = 1.0\n"
            "  elif self.mode == 'triangular2':\n"
            "      scale = 1.0 / (2 ** cycle)\n"
            "  elif self.mode == 'exp_range':\n"
            "      scale = self.gamma ** self.last_epoch\n"
            "  \n"
            "  return self.base_lr + (self.max_lr - self.base_lr) * progress * scale"
        )


class OneCycleLR(LRScheduler):
    """
    1cycle learning rate policy for super-convergence.

    A single cycle that:
    1. Warms up from initial_lr to max_lr
    2. Anneals from max_lr to min_lr

    Enables training with much larger learning rates, often achieving
    better results in fewer epochs.

    The "super-convergence" phenomenon: With 1cycle, models can converge
    in 10x fewer iterations while achieving equal or better accuracy.

    Math:
        Phase 1 (warmup): Linear or cosine increase to max_lr
        Phase 2 (anneal): Cosine decrease to final_div_factor * max_lr

    References:
        - Smith & Topin "Super-Convergence" (2018)
          https://arxiv.org/abs/1708.07120
        - Smith "A disciplined approach to neural network hyper-parameters" (2018)
          https://arxiv.org/abs/1803.09820

    Example:
        >>> # 30% warmup, 70% anneal, common setting
        >>> scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=10000,
        ...                        pct_start=0.3)
    """

    def __init__(self, optimizer: 'Optimizer',
                 max_lr: float,
                 total_steps: int,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 cycle_momentum: bool = True,
                 base_momentum: float = 0.85,
                 max_momentum: float = 0.95,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing LR
            anneal_strategy: 'cos' or 'linear' for annealing phase
            div_factor: initial_lr = max_lr / div_factor
            final_div_factor: final_lr = initial_lr / final_div_factor
            cycle_momentum: Also cycle momentum
        """
        super().__init__(optimizer, last_epoch)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum

        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement OneCycleLR.get_lr()\n"
            "Hint:\n"
            "  if self.last_epoch < self.step_size_up:\n"
            "      # Warmup phase: initial_lr -> max_lr\n"
            "      progress = self.last_epoch / self.step_size_up\n"
            "      lr = self.initial_lr + progress * (self.max_lr - self.initial_lr)\n"
            "  else:\n"
            "      # Annealing phase: max_lr -> final_lr\n"
            "      progress = (self.last_epoch - self.step_size_up) / self.step_size_down\n"
            "      if self.anneal_strategy == 'cos':\n"
            "          lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * progress)) / 2\n"
            "      else:  # linear\n"
            "          lr = self.max_lr - progress * (self.max_lr - self.final_lr)\n"
            "  return lr"
        )


# =============================================================================
# Adaptive Schedulers
# =============================================================================

class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.

    Monitors a quantity and reduces LR by factor when no improvement
    is seen for 'patience' epochs.

    Great when you don't know the optimal schedule in advance.

    Example:
        >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)
        >>> for epoch in range(100):
        ...     train()
        ...     val_loss = validate()
        ...     scheduler.step(val_loss)  # Pass metric to step()
    """

    def __init__(self, optimizer: 'Optimizer',
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: float = 0.0,
                 eps: float = 1e-8):
        """
        Args:
            optimizer: Wrapped optimizer
            mode: 'min' (lower is better) or 'max' (higher is better)
            factor: Factor to reduce LR by (new_lr = old_lr * factor)
            patience: Number of epochs with no improvement before reducing
            threshold: Threshold for measuring new optimum
            threshold_mode: 'rel' (relative) or 'abs' (absolute)
            cooldown: Number of epochs to wait after a LR reduction
            min_lr: Lower bound on the learning rate
            eps: Minimal decay applied to lr
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps

        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0

    def step(self, metric: float) -> None:
        """
        Update scheduler based on metric value.

        Args:
            metric: Value of monitored metric (e.g., validation loss)
        """
        raise NotImplementedError(
            "TODO: Implement ReduceLROnPlateau.step()\n"
            "Hint:\n"
            "  self.last_epoch += 1\n"
            "  \n"
            "  # Check if metric improved\n"
            "  if self.mode == 'min':\n"
            "      if self.threshold_mode == 'rel':\n"
            "          improved = metric < self.best * (1 - self.threshold)\n"
            "      else:\n"
            "          improved = metric < self.best - self.threshold\n"
            "  else:  # mode == 'max'\n"
            "      # ... similar for maximization\n"
            "  \n"
            "  if improved:\n"
            "      self.best = metric\n"
            "      self.num_bad_epochs = 0\n"
            "  else:\n"
            "      self.num_bad_epochs += 1\n"
            "  \n"
            "  # Reduce LR if patience exceeded (and not in cooldown)\n"
            "  if self.cooldown_counter > 0:\n"
            "      self.cooldown_counter -= 1\n"
            "  elif self.num_bad_epochs > self.patience:\n"
            "      current_lr = self.optimizer.defaults['lr']\n"
            "      new_lr = max(current_lr * self.factor, self.min_lr)\n"
            "      if current_lr - new_lr > self.eps:\n"
            "          self.optimizer.set_lr(new_lr)\n"
            "      self.cooldown_counter = self.cooldown\n"
            "      self.num_bad_epochs = 0"
        )


# =============================================================================
# Warmup Schedulers
# =============================================================================

class WarmupLR(LRScheduler):
    """
    Linear warmup scheduler.

    Linearly increases LR from 0 to base_lr over warmup_iters.
    After warmup, keeps LR constant.

    Warmup is crucial for:
    - Transformers (prevents early instability)
    - Large batch training
    - Adam and other adaptive optimizers

    Example:
        >>> scheduler = WarmupLR(optimizer, warmup_iters=1000)
    """

    def __init__(self, optimizer: 'Optimizer', warmup_iters: int,
                 warmup_factor: float = 0.0, last_epoch: int = -1):
        """
        Args:
            warmup_iters: Number of warmup iterations
            warmup_factor: Starting LR = base_lr * warmup_factor
        """
        super().__init__(optimizer, last_epoch)
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement WarmupLR.get_lr()\n"
            "Hint:\n"
            "  if self.last_epoch >= self.warmup_iters:\n"
            "      return self.base_lr\n"
            "  \n"
            "  # Linear warmup\n"
            "  progress = self.last_epoch / self.warmup_iters\n"
            "  factor = self.warmup_factor + progress * (1 - self.warmup_factor)\n"
            "  return self.base_lr * factor"
        )


class WarmupCosineSchedule(LRScheduler):
    """
    Linear warmup followed by cosine decay.

    Common schedule for Transformers and modern architectures:
    1. Linear warmup from 0 to base_lr
    2. Cosine decay from base_lr to min_lr

    References:
        - Used in BERT, GPT, ViT, and many other models
    """

    def __init__(self, optimizer: 'Optimizer',
                 warmup_iters: int,
                 total_iters: int,
                 min_lr: float = 0.0,
                 last_epoch: int = -1):
        """
        Args:
            warmup_iters: Number of warmup iterations
            total_iters: Total training iterations
            min_lr: Minimum LR at end of cosine decay
        """
        super().__init__(optimizer, last_epoch)
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement WarmupCosineSchedule.get_lr()\n"
            "Hint:\n"
            "  if self.last_epoch < self.warmup_iters:\n"
            "      # Warmup phase\n"
            "      return self.base_lr * self.last_epoch / self.warmup_iters\n"
            "  else:\n"
            "      # Cosine decay phase\n"
            "      progress = (self.last_epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters)\n"
            "      cosine_factor = (1 + math.cos(math.pi * progress)) / 2\n"
            "      return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor"
        )


# =============================================================================
# Composite Schedulers
# =============================================================================

class SequentialLR(LRScheduler):
    """
    Chains multiple schedulers sequentially.

    Runs scheduler_1 until milestone_1, then scheduler_2 until milestone_2, etc.

    Example:
        >>> warmup = LinearLR(optimizer, start_factor=0.1, total_iters=100)
        >>> cosine = CosineAnnealingLR(optimizer, T_max=900)
        >>> scheduler = SequentialLR(optimizer, [warmup, cosine], [100])
    """

    def __init__(self, optimizer: 'Optimizer',
                 schedulers: List[LRScheduler],
                 milestones: List[int],
                 last_epoch: int = -1):
        """
        Args:
            schedulers: List of schedulers to chain
            milestones: Epoch indices at which to switch schedulers
        """
        super().__init__(optimizer, last_epoch)
        self.schedulers = schedulers
        self.milestones = milestones

    def get_lr(self) -> float:
        raise NotImplementedError(
            "TODO: Implement SequentialLR.get_lr()\n"
            "Hint:\n"
            "  # Find which scheduler is active\n"
            "  scheduler_idx = 0\n"
            "  for i, m in enumerate(self.milestones):\n"
            "      if self.last_epoch >= m:\n"
            "          scheduler_idx = i + 1\n"
            "  \n"
            "  # Get LR from active scheduler\n"
            "  return self.schedulers[scheduler_idx].get_lr()"
        )


class ChainedScheduler:
    """
    Chains multiple schedulers that are applied multiplicatively.

    Each scheduler modifies the LR factor, and they're multiplied together.

    Example:
        >>> # Combine warmup with exponential decay
        >>> warmup = LinearLR(optimizer, start_factor=0.1, total_iters=100)
        >>> decay = ExponentialLR(optimizer, gamma=0.99)
        >>> scheduler = ChainedScheduler([warmup, decay])
    """

    def __init__(self, schedulers: List[LRScheduler]):
        self.schedulers = schedulers

    def step(self) -> None:
        """Step all schedulers."""
        for scheduler in self.schedulers:
            scheduler.step()


# =============================================================================
# Learning Rate Finder
# =============================================================================

class LRFinder:
    """
    Learning Rate Finder.

    Runs training with exponentially increasing LR to find optimal range.
    Used to determine good base_lr and max_lr values.

    Algorithm:
    1. Start with very small LR
    2. For each batch, increase LR exponentially
    3. Track loss vs LR
    4. Find LR range where loss decreases fastest

    The optimal LR is typically:
    - max_lr: Where loss is still decreasing
    - base_lr: ~10x smaller than max_lr

    References:
        - Smith "Cyclical Learning Rates" (2015)
          https://arxiv.org/abs/1506.01186

    Example:
        >>> finder = LRFinder(model, optimizer, criterion)
        >>> finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
        >>> finder.plot()  # Visualize loss vs LR
        >>> suggested_lr = finder.suggestion()
    """

    def __init__(self, model, optimizer: 'Optimizer', criterion,
                 device: str = 'cpu'):
        """
        Args:
            model: Neural network model
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to run on
        """
        raise NotImplementedError(
            "TODO: Initialize LRFinder\n"
            "Hint:\n"
            "  self.model = model\n"
            "  self.optimizer = optimizer\n"
            "  self.criterion = criterion\n"
            "  self.history = {'lr': [], 'loss': []}\n"
            "  self._original_state = None"
        )

    def range_test(self, train_loader, start_lr: float = 1e-7,
                   end_lr: float = 10.0, num_iter: int = 100,
                   smooth_f: float = 0.05, diverge_th: float = 5.0) -> None:
        """
        Run the LR range test.

        Args:
            train_loader: Training data loader
            start_lr: Initial learning rate
            end_lr: Final learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss
            diverge_th: Stop if loss exceeds diverge_th * best_loss
        """
        raise NotImplementedError(
            "TODO: Implement LRFinder.range_test()\n"
            "Hint:\n"
            "  # Save initial state\n"
            "  self._original_state = self.model.state_dict().copy()\n"
            "  \n"
            "  # Exponential LR schedule\n"
            "  lr_schedule = np.exp(np.linspace(np.log(start_lr), np.log(end_lr), num_iter))\n"
            "  \n"
            "  best_loss = float('inf')\n"
            "  for i, (inputs, targets) in enumerate(train_loader):\n"
            "      if i >= num_iter:\n"
            "          break\n"
            "      \n"
            "      # Set LR\n"
            "      self.optimizer.set_lr(lr_schedule[i])\n"
            "      \n"
            "      # Forward + backward + update\n"
            "      # ... training step ...\n"
            "      \n"
            "      # Record history\n"
            "      self.history['lr'].append(lr_schedule[i])\n"
            "      self.history['loss'].append(loss)\n"
            "      \n"
            "      # Stop if diverging\n"
            "      if loss > diverge_th * best_loss:\n"
            "          break\n"
            "      best_loss = min(best_loss, loss)"
        )

    def suggestion(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """
        Suggest optimal learning rate.

        Returns:
            Suggested learning rate (typically where gradient is steepest)
        """
        raise NotImplementedError(
            "TODO: Implement LRFinder.suggestion()\n"
            "Hint: Find LR where d(loss)/d(log_lr) is most negative"
        )

    def plot(self, skip_start: int = 10, skip_end: int = 5,
             log_lr: bool = True) -> None:
        """Plot loss vs learning rate."""
        raise NotImplementedError("TODO: Implement LRFinder.plot()")

    def reset(self) -> None:
        """Reset model to initial state."""
        raise NotImplementedError(
            "TODO: Implement LRFinder.reset()\n"
            "Hint: self.model.load_state_dict(self._original_state)"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer: 'Optimizer',
                                     num_warmup_steps: int,
                                     num_training_steps: int,
                                     num_cycles: float = 0.5) -> LRScheduler:
    """
    Create warmup + cosine schedule (HuggingFace style).

    Commonly used for fine-tuning Transformers.

    Args:
        optimizer: Wrapped optimizer
        num_warmup_steps: Steps for linear warmup
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (0.5 = half cycle = decay to 0)

    Returns:
        Configured scheduler
    """
    raise NotImplementedError(
        "TODO: Implement get_cosine_schedule_with_warmup\n"
        "Hint: Return WarmupCosineSchedule with appropriate params"
    )


def get_linear_schedule_with_warmup(optimizer: 'Optimizer',
                                     num_warmup_steps: int,
                                     num_training_steps: int) -> LRScheduler:
    """
    Create warmup + linear decay schedule.

    Args:
        optimizer: Wrapped optimizer
        num_warmup_steps: Steps for linear warmup
        num_training_steps: Total training steps

    Returns:
        Configured scheduler
    """
    raise NotImplementedError("TODO: Implement get_linear_schedule_with_warmup")
