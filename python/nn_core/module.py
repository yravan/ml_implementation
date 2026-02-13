"""
Module and Parameter Base Classes
=================================

Base classes for building neural network modules, similar to PyTorch's nn.Module.

Theory
------
A Module is a container for:
1. Parameters (learnable weights)
2. Submodules (other Modules)
3. Buffers (non-learnable state like running mean in BatchNorm)

The Module class provides:
- Automatic parameter discovery via `parameters()`
- Recursive gradient zeroing via `zero_grad()`
- Training/evaluation mode switching
- State dict save/load
- Pretty printing of architecture

Parameters are Tensors that are automatically registered when assigned as attributes.

References
----------
- PyTorch nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- JAX Flax Module: https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html
"""

# Implementation Status: NOT STARTED
# Complexity: Medium
# Prerequisites: foundations/autograd

import numpy as np
from typing import Iterator, Dict, List, Optional, Tuple, Any, Set, Callable, Union
from collections import OrderedDict

from python.foundations import Tensor


class Parameter(Tensor):
    """
    A learnable parameter tensor.

    Parameters are automatically registered when assigned to Module attributes.
    They wrap numpy arrays and track gradients.

    Attributes:
        data: The parameter values (numpy array)
        grad: Accumulated gradients (None until backward is called)
        requires_grad: Whether to compute gradients (always True for Parameters)

    Example:
        >>> param = Parameter(np.random.randn(256, 128))
        >>> param.shape
        (256, 128)
        >>> param.grad  # None until backward
        None
    """

    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        """
        Initialize a Parameter.

        Args:
            data: Initial parameter values as numpy array
            requires_grad: Whether to track gradients (default True)
        """
        super().__init__(data=data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return f"Parameter({self.shape}, requires_grad={self.requires_grad})"


class Module:
    """
    Base class for all neural network modules.

    Modules can contain Parameters (learnable weights), other Modules (submodules),
    and buffers (non-learnable state). This class provides:

    - Automatic parameter registration when Parameters are assigned as attributes
    - Recursive parameter discovery via `parameters()`
    - Gradient zeroing via `zero_grad()`
    - Training/eval mode switching
    - State dict save/load for checkpointing

    Subclasses should implement:
    - `__init__`: Define parameters and submodules
    - `forward`: Define the forward computation

    Example:
        >>> class MLP(Module):
        ...     def __init__(self, in_dim, hidden_dim, out_dim):
        ...         super().__init__()
        ...         self.fc1 = Linear(in_dim, hidden_dim)
        ...         self.fc2 = Linear(hidden_dim, out_dim)
        ...
        ...     def forward(self, x):
        ...         x = self.fc1(x)
        ...         x = np.maximum(x, 0)  # ReLU
        ...         x = self.fc2(x)
        ...         return x
        >>>
        >>> model = MLP(784, 256, 10)
        >>> list(model.parameters())  # Returns all parameters
        [Parameter((256, 784)), Parameter((256,)), ...]
    """

    def __init__(self):
        """Initialize the Module."""
        self._parameters: Dict[str, Parameter] = OrderedDict()
        self._modules: Dict[str, 'Module'] = OrderedDict()
        self._buffers: Dict[str, np.ndarray] = OrderedDict()
        self.training: bool = True


    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override attribute setting to automatically register Parameters and Modules.
        """
        # Handle special attributes
        if name.startswith('_') or name == 'training':
            object.__setattr__(self, name, value)
            return

        # Initialize internal dicts if needed (during __init__)
        if not hasattr(self, '_parameters'):
            object.__setattr__(self, '_parameters', OrderedDict())
        if not hasattr(self, '_modules'):
            object.__setattr__(self, '_modules', OrderedDict())
        if not hasattr(self, '_buffers'):
            object.__setattr__(self, '_buffers', OrderedDict())

        # Register Parameters
        if isinstance(value, Parameter):
            self._parameters[name] = value
        # Register Modules
        elif isinstance(value, Module):
            self._modules[name] = value
        # Regular attributes
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """Get attribute, checking parameters and modules."""
        if '_parameters' in self.__dict__:
            parameters = self.__dict__['_parameters']
            if name in parameters:
                return parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_buffers' in self.__dict__:
            buffers = self.__dict__['_buffers']
            if name in buffers:
                return buffers[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, *args, **kwargs) -> Any:
        """
        Define the forward computation.

        Subclasses must override this method.

        Args:
            *args: Input tensors
            **kwargs: Additional arguments

        Returns:
            Output tensor(s)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs) -> Any:
        """Make the module callable."""
        return self.forward(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Return an iterator over module parameters.

        Args:
            recurse: If True, yield parameters of submodules recursively

        Yields:
            Parameter: Module parameters

        Example:
            >>> for param in model.parameters():
            ...     print(param.shape)
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True
                         ) -> Iterator[Tuple[str, Parameter]]:
        """
        Return an iterator over module parameters, yielding (name, parameter) pairs.

        Args:
            prefix: Prefix to add to parameter names
            recurse: If True, yield parameters of submodules recursively

        Yields:
            Tuple of (name, Parameter)
        """
        # Own parameters
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param

        # Recurse into submodules
        if recurse:
            for mod_name, module in self._modules.items():
                subprefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from module.named_parameters(prefix=subprefix, recurse=True)

    def modules(self) -> Iterator['Module']:
        """
        Return an iterator over all modules (including self).

        Yields:
            Module: This module and all submodules recursively
        """
        yield self
        for name, module in self._modules.items():
            yield from module.modules()

    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """
        Return an iterator over all modules, yielding (name, module) pairs.

        Args:
            prefix: Prefix for module names

        Yields:
            Tuple of (name, Module)
        """
        yield prefix, self
        for name, module in self._modules.items():
            subprefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(prefix=subprefix)

    def children(self) -> Iterator['Module']:
        """
        Return an iterator over immediate child modules.

        Yields:
            Module: Immediate child modules (not recursive)
        """
        for name, module in self._modules.items():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """
        Return an iterator over immediate child modules, yielding (name, module) pairs.

        Yields:
            Tuple of (name, Module)
        """
        for name, module in self._modules.items():
            yield name, module

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """
        Register a parameter with the module.

        Args:
            name: Name of the parameter
            param: Parameter to register, or None
        """
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"Expected Parameter, got {type(param)}")
        else:
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: np.ndarray) -> None:
        """
        Register a buffer (non-learnable state) with the module.

        Buffers are included in state_dict but not in parameters().
        Used for things like running mean in BatchNorm.

        Args:
            name: Name of the buffer
            tensor: Buffer data
        """
        self._buffers[name] = tensor

    def zero_grad(self) -> None:
        """
        Zero out gradients for all parameters.

        Should be called before each backward pass during training.
        """
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True) -> 'Module':
        """
        Set the module to training mode.

        This affects layers like Dropout and BatchNorm.

        Args:
            mode: True for training, False for evaluation

        Returns:
            self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        """
        Set the module to evaluation mode.

        Equivalent to self.train(False).

        Returns:
            self
        """
        return self.train(False)

    def num_parameters(self, only_trainable: bool = True) -> int:
        """
        Count the number of parameters.

        Args:
            only_trainable: If True, only count parameters with requires_grad=True

        Returns:
            Total number of parameter elements
        """
        total = 0
        for param in self.parameters():
            if only_trainable and not param.requires_grad:
                continue
            total += param.size
        return total

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary containing the module state.

        Includes all parameters and buffers.

        Returns:
            Dictionary mapping names to numpy arrays
        """
        state = OrderedDict()

        # Parameters
        for name, param in self.named_parameters():
            state[name] = param.data.copy()

        # Buffers
        for name, buf in self._buffers.items():
            state[name] = buf.copy()

        # Recurse into submodules for their buffers
        for mod_name, module in self._modules.items():
            for buf_name, buf in module._buffers.items():
                state[f"{mod_name}.{buf_name}"] = buf.copy()

        return state

    def load_state_dict(self, state_dict: Dict[str, np.ndarray],
                        strict: bool = True) -> None:
        """
        Load parameters and buffers from state_dict.

        Args:
            state_dict: Dictionary mapping names to numpy arrays
            strict: If True, raise error for missing/unexpected keys
        """
        own_state = self.state_dict()

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - set(own_state.keys())
            if missing:
                raise KeyError(f"Missing keys: {missing}")
            if unexpected:
                raise KeyError(f"Unexpected keys: {unexpected}")

        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].copy()

        for name, buf in self._buffers.items():
            if name in state_dict:
                self._buffers[name] = state_dict[name].copy()

    def apply(self, fn: Callable[['Module'], None]) -> 'Module':
        """
        Apply a function to all submodules (including self).

        Args:
            fn: Function to apply to each module

        Returns:
            self
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _get_name(self) -> str:
        """Get the class name."""
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """
        Extra representation string for subclasses to override.

        Returns:
            String with extra info to display
        """
        return ''

    def __repr__(self) -> str:
        """Pretty print the module architecture."""
        extra = self.extra_repr()
        if not self._modules:
            return f"{self._get_name()}({extra})"

        lines = [f"{self._get_name()}("]
        if extra:
            lines[0] = f"{self._get_name()}({extra},"

        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '\n'.join('  ' + line for line in mod_str.split('\n'))
            lines.append(f"  ({name}): {mod_str.strip()}")

        lines.append(")")
        return '\n'.join(lines)

    def _init_parameters(self, fn: Callable[['Tensor'], None]):
        def initialize(module: 'Module') -> None:
            for name, param in module.named_parameters():
                fn(param)
        self.apply(initialize)







class Sequential(Module):
    """
    A sequential container of modules.

    Modules are added in order and called sequentially during forward.

    Example:
        >>> model = Sequential(
        ...     Linear(784, 256),
        ...     ReLU(),
        ...     Linear(256, 10)
        ... )
        >>> output = model(input)
    """

    def __init__(self, *modules: Module):
        """
        Initialize Sequential with modules.

        Args:
            *modules: Modules to add in order
        """
        super().__init__()
        for idx, module in enumerate(modules):
            self._modules[str(idx)] = module

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all modules sequentially.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        """Get module by index."""
        return self._modules[str(idx)]

    def __len__(self) -> int:
        """Number of modules."""
        return len(self._modules)

    def append(self, module: Module) -> 'Sequential':
        """
        Append a module to the end.

        Args:
            module: Module to append

        Returns:
            self
        """
        idx = len(self._modules)
        self._modules[str(idx)] = module
        return self


class ModuleList(Module):
    """
    A list of modules.

    Unlike Sequential, ModuleList does not define a forward method.
    Useful for storing a list of modules that will be used in a custom way.

    Example:
        >>> layers = ModuleList([Linear(10, 10) for _ in range(5)])
        >>> for layer in layers:
        ...     x = layer(x) + x  # Custom skip connection
    """

    def __init__(self, modules: Optional[List[Module]] = None):
        """
        Initialize ModuleList.

        Args:
            modules: Optional list of modules to add
        """
        super().__init__()
        if modules is not None:
            for idx, module in enumerate(modules):
                self._modules[str(idx)] = module

    def __getitem__(self, idx: int) -> Module:
        """Get module by index."""
        if idx < 0:
            idx = len(self) + idx
        return self._modules[str(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        """Set module by index."""
        self._modules[str(idx)] = module

    def __len__(self) -> int:
        """Number of modules."""
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        """Iterate over modules."""
        return iter(self._modules.values())

    def append(self, module: Module) -> 'ModuleList':
        """Append a module."""
        idx = len(self._modules)
        self._modules[str(idx)] = module
        return self

    def extend(self, modules: List[Module]) -> 'ModuleList':
        """Extend with a list of modules."""
        for module in modules:
            self.append(module)
        return self


class ModuleDict(Module):
    """
    A dictionary of modules.

    Useful for storing named modules that can be accessed by key.

    Example:
        >>> branches = ModuleDict({
        ...     'left': Linear(10, 10),
        ...     'right': Linear(10, 10)
        ... })
        >>> left_out = branches['left'](x)
    """

    def __init__(self, modules: Optional[Dict[str, Module]] = None):
        """
        Initialize ModuleDict.

        Args:
            modules: Optional dictionary of modules
        """
        super().__init__()
        if modules is not None:
            for name, module in modules.items():
                self._modules[name] = module

    def __getitem__(self, key: str) -> Module:
        """Get module by key."""
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        """Set module by key."""
        self._modules[key] = module

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._modules

    def __len__(self) -> int:
        """Number of modules."""
        return len(self._modules)

    def keys(self):
        """Return keys."""
        return self._modules.keys()

    def values(self):
        """Return values."""
        return self._modules.values()

    def items(self):
        """Return items."""
        return self._modules.items()


class ParameterList(Module):
    """
    A list of parameters.

    Example:
        >>> params = ParameterList([Parameter(np.zeros(10)) for _ in range(5)])
        >>> for p in params:
        ...     print(p.shape)
    """

    def __init__(self, parameters: Optional[List[Parameter]] = None):
        """
        Initialize ParameterList.

        Args:
            parameters: Optional list of parameters
        """
        super().__init__()
        if parameters is not None:
            for idx, param in enumerate(parameters):
                self._parameters[str(idx)] = param

    def __getitem__(self, idx: int) -> Parameter:
        """Get parameter by index."""
        if idx < 0:
            idx = len(self) + idx
        return self._parameters[str(idx)]

    def __setitem__(self, idx: int, param: Parameter) -> None:
        """Set parameter by index."""
        self._parameters[str(idx)] = param

    def __len__(self) -> int:
        """Number of parameters."""
        return len(self._parameters)

    def __iter__(self) -> Iterator[Parameter]:
        """Iterate over parameters."""
        return iter(self._parameters.values())

    def append(self, param: Parameter) -> 'ParameterList':
        """Append a parameter."""
        idx = len(self._parameters)
        self._parameters[str(idx)] = param
        return self


class ParameterDict(Module):
    """
    A dictionary of parameters.

    Example:
        >>> params = ParameterDict({
        ...     'weight': Parameter(np.randn(10, 10)),
        ...     'bias': Parameter(np.zeros(10))
        ... })
        >>> params['weight'].shape
        (10, 10)
    """

    def __init__(self, parameters: Optional[Dict[str, Parameter]] = None):
        """
        Initialize ParameterDict.

        Args:
            parameters: Optional dictionary of parameters
        """
        super().__init__()
        if parameters is not None:
            for name, param in parameters.items():
                self._parameters[name] = param

    def __getitem__(self, key: str) -> Parameter:
        """Get parameter by key."""
        return self._parameters[key]

    def __setitem__(self, key: str, param: Parameter) -> None:
        """Set parameter by key."""
        self._parameters[key] = param

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._parameters

    def __len__(self) -> int:
        """Number of parameters."""
        return len(self._parameters)

    def keys(self):
        """Return keys."""
        return self._parameters.keys()

    def values(self):
        """Return values."""
        return self._parameters.values()

    def items(self):
        """Return items."""
        return self._parameters.items()

