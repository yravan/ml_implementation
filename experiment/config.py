"""
Experiment Configuration.

Simple dataclass config that can be created from:
  1. Python kwargs:  Config(model='resnet18', lr=0.001)
  2. YAML file:      Config.from_yaml('config.yaml')
  3. CLI args:        Config.from_cli()
  4. Mix:             Config.from_cli(defaults={'model': 'resnet18'})
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
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

    # ── Task Type ──────────────────────────────────────────────────────
    task: str = "classification"       # classification | language_model | seq2seq

    # ── Data ─────────────────────────────────────────────────────────
    dataset: str = "mnist"             # mnist, imagenette, imagenet, cifar10
    data_dir: str = "./data"
    val_split: float = 0.1
    num_workers: int = 8
    pin_memory: bool = True

    # ── Sequence / Tokenizer ─────────────────────────────────────────
    tokenizer: str = "gpt2"           # HuggingFace tokenizer name
    max_seq_len: int = 512
    src_max_seq_len: Optional[int] = None   # seq2seq source; defaults to max_seq_len
    tgt_max_seq_len: Optional[int] = None   # seq2seq target; defaults to max_seq_len

    # ── Model ────────────────────────────────────────────────────────
    model: str = "mlp"                 # mlp, cnn, resnet18, resnet34, resnet50
    model_args: Dict[str, Any] = field(default_factory=dict)
    pretrained: bool = False
    resume: Optional[str] = None       # path to checkpoint
    pretrained_weights: Optional[str] = None  # e.g. "gpt2", "gpt2-medium"

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
    gradient_accumulation_steps: int = 1  # accumulate gradients over N micro-batches
    label_smoothing: float = 0.0       # cross-entropy label smoothing (0.1 recommended)
    max_steps: Optional[int] = None    # if set, train for N steps (overrides epochs)
    warmup_steps: int = 0              # linear warmup steps (used when max_steps is set)
    save_every_steps: int = 0          # save/validate/generate every N steps (0=off)

    # ── Generation (language_model / seq2seq) ────────────────────────
    generate_every: int = 0           # generate samples every N epochs (0=off)
    generate_max_tokens: int = 100
    generate_temperature: float = 0.8
    generate_top_k: Optional[int] = 50
    generate_top_p: float = 0.95
    generate_prompts: List[str] = field(default_factory=lambda: ["Once upon a time"])
    num_generate_samples: int = 3

    # ── Logging ──────────────────────────────────────────────────────
    logger: str = "tensorboard"        # tensorboard, wandb, console, all
    wandb_project: str = "dl-experiments"
    wandb_entity: Optional[str] = None
    wandb_api: str = "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"
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
    compile_mode: str = 'default'
    compile_cache_dir: Optional[str] = None  # torch.compile cache dir (e.g. /tmp/torch_compile_cache)
    cudnn_benchmark: bool = True       # auto-tune convolution algorithms
    ffcv: bool = False                 # use FFCV data loading (requires pip install ffcv)
    beton_dir: Optional[str] = None    # directory for FFCV .beton files (default: <data_dir>/ffcv)
    beton_source_dir: Optional[str] = None  # pre-built betons to copy into beton_dir (e.g. pool storage)

    # ── Distributed (DDP) ────────────────────────────────────────────
    ddp: bool = False                  # enable DistributedDataParallel
    ddp_backend: str = "nccl"          # nccl (GPU) or gloo (CPU/fallback)
    ddp_find_unused: bool = False      # find_unused_parameters in DDP wrapper
    ddp_gradient_as_bucket: bool = True  # gradient_as_bucket_view (saves memory)

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
        """Create config from dict, ignoring unknown keys.

        Coerces values to match the declared field types so that YAML
        strings like '4e-3' are properly converted to float.
        """
        valid_fields = {f.name: f for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid_fields:
                continue
            # Coerce str → float / int when the field expects a numeric type
            ftype = valid_fields[k].type
            if v is not None and isinstance(v, str):
                if ftype is float or ftype == 'float':
                    v = float(v)
                elif ftype is int or ftype == 'int':
                    v = int(v)
            filtered[k] = v
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