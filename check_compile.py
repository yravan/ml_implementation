"""
Quick diagnostic: is torch.compile actually helping?
Checks for graph breaks and benchmarks compiled vs uncompiled forward+backward.

Usage: python check_compile.py
"""

import torch
import torch.nn.functional as F
import time
import logging

from experiment import Config
from experiment.registry import build_model

# ── Load config & build model ──────────────────────────────────────
config = Config.from_yaml('configs/imagenet_resnet18.yaml')
device = torch.device('cuda')
dtype = torch.float16  # match AMP

BS = config.batch_size
SZ = config.model_args.get('img_size', 224)
print(f"Model: {config.model}  |  Batch: {BS}  |  Img: {SZ}x{SZ}  |  Device: {device}")
print(f"{'='*70}")

# ── Fake batch (stays on GPU the whole time) ───────────────────────
x = torch.randn(BS, 3, SZ, SZ, device=device, dtype=dtype).to(memory_format=torch.channels_last)
y = torch.randint(0, 1000, (BS,), device=device)


def bench(model, label, warmup=5, iters=20):
    """Time forward + backward over `iters` iterations."""
    model.train()
    # warmup
    for _ in range(warmup):
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    ips = BS / avg
    print(f"  {label:30s}  {avg*1000:7.1f} ms/batch  |  {ips:,.0f} img/s")
    return avg


# ── 1) Uncompiled baseline ────────────────────────────────────────
print(f"\n{'─'*70}")
print("1. Uncompiled (eager mode)")
print(f"{'─'*70}")
model_eager = build_model(config).to(device, memory_format=torch.channels_last)
t_eager = bench(model_eager, "eager")
del model_eager
torch.cuda.empty_cache()

# ── 2) Compiled (default mode) ────────────────────────────────────
print(f"\n{'─'*70}")
print("2. torch.compile(model)  [default mode]")
print(f"{'─'*70}")
model_default = build_model(config).to(device, memory_format=torch.channels_last)
model_default = torch.compile(model_default)
print("  Compiling (first run triggers compilation)...")
t_default = bench(model_default, "compile (default)")
del model_default
torch.cuda.empty_cache()

# ── 3) Compiled (max-autotune) ────────────────────────────────────
print(f"\n{'─'*70}")
print("3. torch.compile(model, mode='max-autotune')")
print(f"{'─'*70}")
model_max = build_model(config).to(device, memory_format=torch.channels_last)
model_max = torch.compile(model_max, mode='max-autotune')
print("  Compiling (max-autotune takes longer, tries CUDA graphs + kernel tuning)...")
t_max = bench(model_max, "compile (max-autotune)")
del model_max
torch.cuda.empty_cache()

# ── 4) Check for graph breaks ─────────────────────────────────────
print(f"\n{'─'*70}")
print("4. Graph break check")
print(f"{'─'*70}")
model_check = build_model(config).to(device, memory_format=torch.channels_last)

# Enable graph break logging
torch._dynamo.config.log_level = logging.WARNING
torch._dynamo.reset()

explain_model = torch.compile(model_check, mode='max-autotune')
with torch.amp.autocast('cuda'):
    logits = explain_model(x)
    loss = F.cross_entropy(logits, y)
loss.backward()
torch.cuda.synchronize()

# Get graph break count
break_count = torch._dynamo.utils.counters.get("graph_break", {})
if break_count:
    print(f"  GRAPH BREAKS DETECTED: {break_count}")
    print("  ^ These prevent full optimization. Each break = a fallback to eager mode.")
else:
    print("  No graph breaks detected — torch.compile can optimize the full model.")

del model_check, explain_model
torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
speedup_default = t_eager / t_default
speedup_max = t_eager / t_max
print(f"  eager          → {BS/t_eager:>8,.0f} img/s  (baseline)")
print(f"  compile        → {BS/t_default:>8,.0f} img/s  ({speedup_default:.2f}x)")
print(f"  max-autotune   → {BS/t_max:>8,.0f} img/s  ({speedup_max:.2f}x)")
if speedup_max < 1.05:
    print(f"\n  ⚠  torch.compile isn't helping much (<5% speedup).")
    print(f"     Possible causes: graph breaks, small batch, or model already memory-bound.")
elif speedup_max > 1.20:
    print(f"\n  ✓  torch.compile is giving a solid {(speedup_max-1)*100:.0f}% speedup.")
else:
    print(f"\n  ~  Modest {(speedup_max-1)*100:.0f}% speedup from torch.compile.")
