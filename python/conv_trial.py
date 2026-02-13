# Test conv in isolation
from python.foundations import Tensor
from python.nn_core import Conv2d
import numpy as np

conv = Conv2d(64, 192, 5, padding=2, bias=False)
# Force EXACT zero-mean weights
conv.weight.data -= conv.weight.data.mean()
print(f"Weight mean: {conv.weight.data.mean()}")  # should be 0.0

# Positive input (simulating post-ReLU)
x = Tensor(np.abs(np.random.randn(2, 64, 27, 27).astype(np.float32)))
out = conv(x)
print(f"Output mean: {out.data.mean():.4f}")  # should be ~0

# Even simpler: constant input
x2 = Tensor(np.ones((2, 64, 27, 27), dtype=np.float32))
out2 = conv(x2)
print(f"Constant input, output mean: {out2.data.mean():.4f}")  # MUST be ~0