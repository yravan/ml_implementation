"""
Numerical Gradient Checking
===========================

Utilities for verifying analytical gradients against numerical approximations.

Theory
------
Gradient checking is essential for debugging autodiff implementations. The idea is
simple: compare your analytical gradient (from backprop) with a numerical approximation
computed using finite differences.

The numerical gradient is computed by perturbing each input slightly and measuring
the change in output:

∂f/∂x_i ≈ [f(x + ε*e_i) - f(x - ε*e_i)] / (2ε)

where e_i is the unit vector in direction i and ε is a small perturbation (typically 1e-5).

If your analytical and numerical gradients match (relative error < 1e-5), your
implementation is likely correct. If they don't match, there's a bug.

Common causes of gradient check failures:
1. Bug in backward pass implementation
2. Numerical instability (try different epsilon)
3. Discontinuous functions (ReLU at 0)
4. Random operations (dropout - disable during check)

Math
----
# Forward difference (less accurate):
# ∂f/∂x_i ≈ [f(x + ε*e_i) - f(x)] / ε
# Error: O(ε)

# Central difference (more accurate, what we use):
# ∂f/∂x_i ≈ [f(x + ε*e_i) - f(x - ε*e_i)] / (2ε)
# Error: O(ε²)

# Relative error for comparing gradients:
# rel_error = |a - n| / max(|a|, |n|, 1e-8)
# where a = analytical gradient, n = numerical gradient

# Good: rel_error < 1e-5
# OK: rel_error < 1e-3 (might be numerical precision issues)
# Bad: rel_error > 1e-2 (likely a bug)

Algorithm
---------
For each parameter element x_i:
    1. Save original value
    2. x_i += epsilon
    3. Forward pass to get f_plus
    4. x_i -= 2*epsilon  # Now at original - epsilon
    5. Forward pass to get f_minus
    6. x_i += epsilon  # Restore original
    7. numerical_grad[i] = (f_plus - f_minus) / (2*epsilon)

Compare numerical_grad with analytical_grad element-wise.

References
----------
- CS231n: Gradient checking
  https://cs231n.github.io/neural-networks-3/#gradcheck
- "Numerical Recipes" Ch. 5.7: Numerical Derivatives
- PyTorch gradcheck documentation
  https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html

Implementation Notes
--------------------
- Use float64 for better numerical precision
- Central difference is more accurate than forward difference
- Test on small inputs first (faster, easier to debug)
- Disable dropout, batch norm in eval mode during checking
- For ReLU, avoid checking exactly at 0
- Check both dense gradients and sparse gradients
"""

# Implementation Status: NOT STARTED
# Complexity: Easy
# Prerequisites: computational_graph, autograd

import numpy as np
from typing import Callable, List, Tuple, Optional, Union, TYPE_CHECKING
from .computational_graph import Tensor

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from ..nn_core.module import Module


def numerical_gradient(func: Callable[[Tensor], Union[Tensor, float, np.ndarray]],
                       x: Tensor,
                       epsilon: float = 1e-3) -> np.ndarray:
    """
    Compute numerical gradient using central differences.

    This computes the gradient of a function with respect to
    each element of the input array.

    Args:
        func: Function that takes array
        x: Point at which to compute gradient
        epsilon: Perturbation size

    Returns:
        Numerical gradient, same shape as x

    Example:
        >>> def f(x):
        ...     return np.sum(x ** 2)
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> numerical_gradient(f, x)
        array([2., 4., 6.])  # Analytical: 2*x
    """
    iter = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
    output = func(x)
    grad = np.zeros_like(x.data)
    if isinstance(output, np.ndarray):
        output_shape = output.shape
        # Add newaxis for each dimension in output_shape
        for _ in range(len(output_shape)):
            grad = grad[..., np.newaxis]
        grad = np.broadcast_to(grad, x.shape + output_shape).copy()
    while not iter.finished:
        idx = iter.multi_index
        original = x.data[idx]
        x.data[idx] = original + epsilon
        f_plus = float(func(x).data)
        x.data[idx] = original - epsilon
        f_minus = float(func(x).data)
        x.data[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        iter.iternext()
    return grad


def gradient_check(func: Callable,
                   inputs: List[np.ndarray],
                   analytical_grads: List[np.ndarray],
                   epsilon: float = 1e-5,
                   rtol: float = 1e-4,
                   atol: float = 1e-6,
                   verbose: bool = True) -> Tuple[bool, List[float]]:
    """
    Check analytical gradients against numerical gradients.

    Args:
        func: Function that takes *inputs
        inputs: List of input arrays
        analytical_grads: Analytical gradients to check (from backprop)
        epsilon: Perturbation for numerical gradient
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: Whether to print results

    Returns:
        Tuple of (passed: bool, relative_errors: List[float])

    Example:
        >>> def f(W, b, x):
        ...     return np.sum((W @ x + b) ** 2)
        >>>
        >>> W = np.random.randn(3, 2)
        >>> b = np.random.randn(3)
        >>> x = np.random.randn(2)
        >>>
        >>> # Compute analytical gradients (you implement this)
        >>> y = W @ x + b
        >>> grad_y = 2 * y
        >>> grad_W = np.outer(grad_y, x)
        >>> grad_b = grad_y
        >>>
        >>> passed, errors = gradient_check(
        ...     lambda W, b: f(W, b, x),
        ...     [W, b],
        ...     [grad_W, grad_b]
        ... )
        >>> passed
        True
    """
    relative_errors = []
    passed = True
    for i, (input, analytical) in enumerate(zip(inputs, analytical_grads)):
        def f(x):
            inputs[i] = x
            return func(*inputs)
        numerical = numerical_gradient(f, input, epsilon)
        inputs[i] = input
        if not np.allclose(numerical, analytical, rtol=rtol, atol=atol): passed = False
        rel_err = compute_relative_error(analytical, numerical, epsilon)
        relative_errors.append(rel_err)
    return passed, relative_errors

def compute_relative_error(a: np.ndarray, b: np.ndarray,
                           eps: float = 1e-8) -> float:
    """
    Compute relative error between two arrays.

    Uses the formula: |a - b| / max(|a|, |b|, eps)
    This handles the case where gradients are near zero.

    Args:
        a: First array (typically analytical gradient)
        b: Second array (typically numerical gradient)
        eps: Small constant to avoid division by zero

    Returns:
        Maximum relative error across all elements

    Example:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.001, 2.002, 3.003])
        >>> compute_relative_error(a, b)
        0.001  # Approximately
    """
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    scale = np.maximum(scale, eps)
    return (diff / scale).max()


def check_layer_gradients(layer: 'Module',
                          input_shape: Tuple[int, ...],
                          epsilon: float = 1e-5,
                          seed: int = 42) -> bool:
    """
    Check gradients for a neural network layer.

    Convenience function for testing layer implementations.

    Args:
        layer: Layer to test (must have forward, backward, and parameters)
        input_shape: Shape of input tensor
        epsilon: Perturbation size
        seed: Random seed for reproducibility

    Returns:
        True if all gradient checks pass

    Example:
        >>> linear = Linear(10, 5)
        >>> check_layer_gradients(linear, input_shape=(2, 10))
        True
    """
    rng = np.random.default_rng(seed)
    input = rng.uniform(size=input_shape); input = Tensor(input)
    output = layer(input); output.backward()
    analytical_grads = [input.grad]
    if not gradient_check(layer, [input.data], analytical_grads, epsilon):
        return False
    for name, param in layer.named_parameters():
        analytical_grads = [param.grad]
        def func(x):
            layer.register_parameter(name, x)
            return layer(input)
        if not gradient_check(func, [param.data], analytical_grads, epsilon):
            return False
        layer.register_parameter(name, param)
    return True


def check_loss_gradient(loss_fn: Callable,
                        pred_shape: Tuple[int, ...],
                        target_shape: Tuple[int, ...],
                        epsilon: float = 1e-5,
                        seed: int = 42) -> bool:
    """
    Check gradients for a loss function.

    Args:
        loss_fn: Loss function (predictions, targets) -> scalar
        pred_shape: Shape of predictions
        target_shape: Shape of targets
        epsilon: Perturbation size
        seed: Random seed

    Returns:
        True if gradient check passes

    Example:
        >>> def mse_loss(pred, target):
        ...     return np.mean((pred - target) ** 2)
        >>> check_loss_gradient(mse_loss, (10, 5), (10, 5))
        True
    """
    rng = np.random.default_rng(seed)
    pred = rng.random(size = pred_shape)
    pred = Tensor(pred, requires_grad = True)
    target = rng.random(size = target_shape)
    loss = loss_fn(pred, target)
    loss.backward()
    analytical_grad = pred.grad
    passed, _ =  gradient_check(lambda p: loss_fn(p, target), [pred], analytical_grad, epsilon)
    return passed


class GradientChecker:
    """
    Comprehensive gradient checker for neural network modules.

    Provides detailed reports on gradient accuracy including:
    - Per-parameter error breakdown
    - Identification of problematic elements
    - Suggestions for debugging

    Example:
        >>> checker = GradientChecker(epsilon=1e-5, tolerance=1e-4)
        >>> model = Sequential([Linear(10, 5), ReLU(), Linear(5, 2)])
        >>> report = checker.check(model, input_shape=(4, 10))
        >>> print(report)
        Gradient Check Report
        =====================
        Linear_0.weight: PASS (max_error=1.2e-7)
        Linear_0.bias: PASS (max_error=8.3e-8)
        ...
    """

    def __init__(self, epsilon: float = 1e-5, tolerance: float = 1e-4):
        """
        Initialize GradientChecker.

        Args:
            epsilon: Perturbation size for numerical gradients
            tolerance: Maximum allowed relative error
        """
        self.epsilon = epsilon
        self.tolerance = tolerance

    def check(self, module: 'Module', input_shape: Tuple[int, ...],
              seed: int = 42) -> 'GradCheckReport':
        """
        Run comprehensive gradient check on a module.

        Args:
            module: Neural network module to check
            input_shape: Shape of input data
            seed: Random seed for reproducibility

        Returns:
            GradCheckReport with detailed results
        """
        rng = np.random.default_rng(seed)
        input = rng.random(size = input_shape); input = Tensor(input)
        output = module(input); output.backward()
        report = GradCheckReport()
        for name, param in module.named_parameters(recurse=True):
            analytical_grad = param.grad
            param_module = module
            complete_name = name
            while name.find(".") != -1:
                param_module = param_module._modules[name[:name.find(".")]]
                name = name[name.find(".") + 1:]
            def func(x):
                param_module.register_parameter(name, x)
                return module(input)
            passed, rel_error = gradient_check(func, [param.data], [analytical_grad], self.epsilon)
            report.add_result(complete_name, passed, rel_error[0])
            param_module.register_parameter(name, param)
        return report


class GradCheckReport:
    """Report from gradient checking."""

    def __init__(self):
        self.results = {}  # param_name -> (passed, error, details)
        self.overall_passed = True

    def add_result(self, name: str, passed: bool, error: float,
                   details: Optional[str] = None) -> None:
        """Add a result for a parameter."""
        self.results[name] = (passed, error, details)
        self.overall_passed = self.overall_passed and passed

    def __str__(self) -> str:
        """Format report as string."""
        output = """
        Gradient Check Report
        =====================
        """
        for name, passed, error, details in self.results:
            output += f"""{name}: """
            if passed:
                output += f"""PASS (max_error={error})"""
            else:
                output += f"""FAIL (max_error={error})"""
            output += "\n"
        output += f"""
            Overall Passed: {self.overall_passed}
        """
        return output

def gradcheck(func: Callable, inputs: Tuple[Tensor, ...],
              eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3,
              raise_exception: bool = True) -> bool:
    """
    PyTorch-style gradcheck function.

    Checks gradients of func with respect to inputs.

    Args:
        func: Function to check (inputs -> output)
        inputs: Tuple of Tensors (with requires_grad=True for those to check)
        eps: Perturbation size
        atol: Absolute tolerance
        rtol: Relative tolerance
        raise_exception: If True, raise on failure

    Returns:
        True if gradients are correct

    Example:
        >>> x = Tensor(np.random.randn(3, 4), requires_grad=True)
        >>> y = Tensor(np.random.randn(4, 2), requires_grad=True)
        >>> gradcheck(lambda x, y: (x @ y).sum(), (x, y))
        True
    """
    # Zero gradients before backward to prevent accumulation from prior calls
    inputs = list(inputs)
    for inp in inputs:
        if hasattr(inp, 'grad') and inp.grad is not None:
            inp.grad = None
    output = func(*inputs); output.backward()
    for i, input in enumerate(inputs):
        if not input.requires_grad: continue
        def f(x: Tensor, _i=i):  # capture i by value
            inputs[_i] = x
            x.requires_grad = True
            return func(*inputs)
        analytical = input.grad.copy()  # copy to avoid mutation from later calls
        numerical = numerical_gradient(f, input, epsilon=eps)
        abs_err = np.abs(analytical - numerical).max()
        rel_err = compute_relative_error(analytical, numerical, eps=eps)
        if abs_err >= atol or rel_err >= rtol:
            if raise_exception:
                raise RuntimeError("Gradient check failed")
            return False
    return True


def gradgradcheck(func: Callable, inputs: Tuple[Tensor, ...],
                  grad_outputs: Optional[Tuple[Tensor, ...]] = None,
                  eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3
                  ) -> bool:
    """
    Check second derivatives (gradient of gradient).

    Verifies that the backward pass itself is differentiable
    and correct. Important for:
    - Double backprop (e.g., MAML)
    - Hessian computation
    - Some regularization techniques

    Args:
        func: Function to check
        inputs: Input tensors
        grad_outputs: Upstream gradients (random if None)
        eps, atol, rtol: Tolerances

    Returns:
        True if second derivatives are correct
    """
    raise NotImplementedError(
        "TODO: Implement gradgradcheck\n"
        "Hint: Check gradient of the backward pass"
    )


def check_gradient(func: Callable,
                   grad_func: Callable,
                   x: np.ndarray,
                   epsilon: float = 1e-5,
                   tolerance: float = 1e-4) -> bool:
    """
    Simple gradient check function (convenience wrapper).

    Compares analytical gradient (from grad_func) against numerical approximation.

    Args:
        func: Scalar function f(x) -> float
        grad_func: Gradient function that computes analytical gradient
        x: Point at which to check gradient
        epsilon: Perturbation for numerical gradient
        tolerance: Maximum allowed relative error

    Returns:
        True if gradient check passes, False otherwise

    Example:
        >>> def f(x):
        ...     return np.sum(x ** 2)
        >>> def grad_f(x):
        ...     return 2 * x  # Gradient of sum(x^2) is 2x
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> passed = check_gradient(f, grad_f, x)
        >>> print(f"Passed: {passed}")
    """
    analytical_grad = grad_func(x)
    numerical_grad = numerical_gradient(func, x, epsilon)
    relative_error = compute_relative_error(analytical_grad, numerical_grad)
    return bool(relative_error < tolerance)
