"""
Experiment Configuration.

Simple dataclass config that can be created from:
  1. Python kwargs:  Config(model='resnet18', lr=0.001)
  2. YAML file:      Config.from_yaml('config.yaml')
  3. CLI args:        Config.from_cli()
  4. Mix:             Config.from_cli(defaults={'model': 'resnet18'})
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


@dataclass
class Config:
    """All experiment hyperparameters in one place."""

    # ── Experiment ───────────────────────────────────────────────────
    name: str = "experiment"
    seed: int = 42
    output_dir: str = "./outputs"
    backend: str = "pytorch"           # pytorch or numpy

    # ── Data ─────────────────────────────────────────────────────────
    dataset: str = "mnist"             # mnist, imagenette, imagenet, cifar10
    data_dir: str = "./data"
    val_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True

    # ── Model ────────────────────────────────────────────────────────
    model: str = "mlp"                 # mlp, cnn, resnet18, resnet34, resnet50
    model_args: Dict[str, Any] = field(default_factory=dict)
    pretrained: bool = False
    resume: Optional[str] = None       # path to checkpoint

    # ── Training ─────────────────────────────────────────────────────
    epochs: int = 10
    batch_size: int = 64
    lr: float = 0.001
    optimizer: str = "adamw"           # sgd, adam, adamw
    optimizer_args: Dict[str, Any] = field(default_factory=dict)
    scheduler: str = "cosine"          # step, cosine, none
    warmup_epochs: int = 0
    weight_decay: float = 1e-4
    momentum: float = 0.9             # for SGD
    grad_clip: Optional[float] = None

    # ── Logging ──────────────────────────────────────────────────────
    logger: str = "tensorboard"        # tensorboard, wandb, console, all
    wandb_project: str = "dl-experiments"
    wandb_entity: Optional[str] = None
    log_interval: int = 50             # batches between prints
    save_every: int = 0                # save checkpoint every N epochs (0 = best only)

    # ── Metrics ──────────────────────────────────────────────────────
    metrics: List[str] = field(default_factory=lambda: ["top1"])

    # ── Debug ────────────────────────────────────────────────────────
    profile: bool = False
    debug: bool = False
    subset: Optional[int] = None       # use N samples for quick testing

    # ── Performance ───────────────────────────────────────────────────
    amp: bool = False                  # automatic mixed precision (fp16/bf16)
    compile: bool = False              # torch.compile (PyTorch 2.x)
    cudnn_benchmark: bool = True       # auto-tune convolution algorithms

    # ── Derived ──────────────────────────────────────────────────────
    @property
    def run_dir(self) -> Path:
        """Output directory for this specific run."""
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Optional[str] = None):
        """Save config to JSON."""
        if path is None:
            path = self.run_dir / "config.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create config from dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_cli(cls, defaults: Optional[dict] = None) -> "Config":
        """
        Parse config from command-line arguments.

        All Config fields become CLI args automatically:
            python run.py --model resnet18 --lr 0.001 --epochs 90

        Supports --config path.yaml to load base config, then override with CLI args.
        """
        import argparse

        parser = argparse.ArgumentParser(description="Run experiment")
        parser.add_argument('--config', type=str, default=None,
                            help='Path to YAML config file')

        # Add all dataclass fields as CLI args
        for f in cls.__dataclass_fields__.values():
            name = f"--{f.name}"
            if f.type == bool:
                parser.add_argument(name, action='store_true', default=None)
                parser.add_argument(f"--no-{f.name}", dest=f.name,
                                    action='store_false')
            elif f.type in (Dict[str, Any], dict):
                parser.add_argument(name, type=json.loads, default=None,
                                    help=f'{f.name} as JSON string')
            elif f.type in (List[str], list):
                parser.add_argument(name, nargs='+', default=None)
            elif hasattr(f.type, '__args__'):  # Optional[X]
                inner = f.type.__args__[0]
                parser.add_argument(name, type=inner, default=None)
            else:
                parser.add_argument(name, type=f.type, default=None)

        args = parser.parse_args()

        # Start with defaults
        d = defaults.copy() if defaults else {}

        # Layer on YAML config
        if args.config:
            import yaml
            with open(args.config) as f:
                d.update(yaml.safe_load(f))

        # Layer on CLI args (only non-None values)
        for k, v in vars(args).items():
            if k != 'config' and v is not None:
                d[k] = v

        return cls.from_dict(d)

    def __str__(self):
        lines = ["Config:"]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)