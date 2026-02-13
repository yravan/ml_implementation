"""
Backward Pass Profiler
======================

Drop-in profiler that instruments the autograd backward pass
to show exactly which operations are slow.

Usage:
    from profile_backward import enable_backward_profiling, print_backward_profile, reset_backward_profile

    enable_backward_profiling()

    # ... your training loop ...
    loss.backward()

    print_backward_profile()  # shows per-operation timing
    reset_backward_profile()  # reset for next iteration
"""

import time
import numpy as np
from collections import defaultdict


# Global timing storage
_timings = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "calls": []})
_enabled = False


def reset_backward_profile():
    """Reset all collected timings."""
    _timings.clear()


def print_backward_profile(top_n=30):
    """Print backward pass timing breakdown."""
    if not _timings:
        print("No backward timings collected. Did you call enable_backward_profiling()?")
        return

    print("\n" + "=" * 80)
    print(" BACKWARD PASS PROFILING")
    print("=" * 80)

    total_ms = sum(v["total_ms"] for v in _timings.values())

    # Sort by total time
    sorted_ops = sorted(_timings.items(), key=lambda x: -x[1]["total_ms"])

    print(f"\n{'Operation':<45} {'Calls':>5} {'Total(ms)':>10} {'Avg(ms)':>10} {'%':>6}")
    print("-" * 80)

    for name, info in sorted_ops[:top_n]:
        pct = 100.0 * info["total_ms"] / total_ms if total_ms > 0 else 0
        avg = info["total_ms"] / info["count"] if info["count"] > 0 else 0
        print(f"  {name:<43} {info['count']:>5} {info['total_ms']:>10.2f} {avg:>10.3f} {pct:>5.1f}%")

    print("-" * 80)
    print(f"  {'TOTAL':<43} {'':>5} {total_ms:>10.2f}")

    # Group by category
    print(f"\n{'Category':<30} {'Total(ms)':>10} {'%':>6}")
    print("-" * 50)
    categories = defaultdict(float)
    for name, info in _timings.items():
        if "Conv2d" in name:
            categories["Conv2d backward"] += info["total_ms"]
        elif "MaxPool" in name:
            categories["MaxPool2d backward"] += info["total_ms"]
        elif "BatchNorm" in name:
            categories["BatchNorm2d backward"] += info["total_ms"]
        elif "MatMul" in name:
            categories["MatMul backward (FC layers)"] += info["total_ms"]
        elif "Add" in name:
            categories["Add backward (bias, residual)"] += info["total_ms"]
        elif "ReLU" in name or "Relu" in name:
            categories["ReLU backward"] += info["total_ms"]
        elif "Reshape" in name or "Transpose" in name:
            categories["Reshape/Transpose backward"] += info["total_ms"]
        elif "Softmax" in name or "LogSoftmax" in name:
            categories["Softmax/Loss backward"] += info["total_ms"]
        elif "AdaptiveAvg" in name:
            categories["AdaptiveAvgPool backward"] += info["total_ms"]
        elif "Slice" in name or "Concat" in name:
            categories["Slice/Concat backward"] += info["total_ms"]
        elif "Dropout" in name:
            categories["Dropout backward"] += info["total_ms"]
        else:
            categories[f"Other ({name.split('(')[0].strip()})"] += info["total_ms"]

    for cat, ms in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100.0 * ms / total_ms if total_ms > 0 else 0
        print(f"  {cat:<28} {ms:>10.2f} {pct:>5.1f}%")
    print("-" * 50)
    print(f"  {'TOTAL':<28} {total_ms:>10.2f}")
    print()


def enable_backward_profiling():
    """Monkey-patch the Tensor.backward to profile each operation."""
    global _enabled
    if _enabled:
        return

    from python.foundations.computational_graph import Tensor

    _original_backward = Tensor.backward

    def profiled_backward(self, grad=None):
        """Profiled version of backward that times each grad_fn call."""
        if self.grad is None:
            if self.ndim == 0:
                self.grad = np.array([1.0], dtype=self.dtype)
            else:
                self.grad = np.ones_like(self.data, dtype=self.data.dtype)
        else:
            if grad is not None:
                if grad.ndim != self.ndim:
                    raise RuntimeError("Expected grad with the same dim")
                if grad.shape != self.shape:
                    raise RuntimeError("Expected grad with the same shape")

        # Topological sort (same as original)
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._children:
                if child not in visited:
                    stack.append((child, False))

        # Profiled backward pass
        for node in reversed(topo):
            if node.requires_grad and node._grad_fn is not None:
                fn = node._grad_fn
                fn_name = type(fn).__name__

                # Add shape info for key operations
                grad_shape = node.grad.shape if hasattr(node.grad, 'shape') else '(scalar)'
                label = f"{fn_name} (grad={grad_shape})"

                t0 = time.perf_counter()
                child_grads = fn.backward(node.grad)
                dt_ms = (time.perf_counter() - t0) * 1000

                _timings[label]["count"] += 1
                _timings[label]["total_ms"] += dt_ms

                for child, child_grad in zip(node._children, child_grads):
                    if child.requires_grad:
                        if child.grad is None:
                            child.grad = child_grad
                        else:
                            child.grad += child_grad

    Tensor.backward = profiled_backward
    _enabled = True
    print("[profile_backward] Profiling enabled — backward() will now collect per-op timings")


if __name__ == "__main__":
    # Quick test with a simple model
    print("Run this in your training script:")
    print()
    print("  from profile_backward import enable_backward_profiling, print_backward_profile, reset_backward_profile")
    print("  enable_backward_profiling()")
    print()
    print("  # In your training loop, after loss.backward():")
    print("  print_backward_profile()")
    print("  reset_backward_profile()")