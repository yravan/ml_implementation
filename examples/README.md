# Examples

This directory contains runnable example scripts demonstrating how to use the ML implementations.

## Quick Start

```bash
cd examples/
python <example_script>.py
```

## Available Examples

### Reinforcement Learning

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| `rl_gridworld_qlearning.py` | Train Q-Learning on GridWorld | `envs/gridworld`, `rl/tabular/q_learning` |
| `rl_dqn_cartpole.py` | Train DQN on CartPole-v1 | `rl/value_based/dqn`, gymnasium |
| `rl_ppo_continuous.py` | Train PPO on continuous control | `rl/policy_gradient/ppo`, gymnasium |

### Supervised Learning

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| `train_mlp_mnist.py` | Train MLP on MNIST | `architectures/mlp`, `foundations/autograd` |

### Generative Models

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| `train_vae_mnist.py` | Train VAE on MNIST | `generative/autoencoders/vae` |

## Implementation Order

To run these examples, implement the modules in this order:

### Phase 1: Foundations (Week 1-2)
```
foundations/computational_graph.py  # Tensors and operations
foundations/autograd.py             # Automatic differentiation
foundations/gradient_check.py       # Numerical gradient verification
```

### Phase 2: Core NN (Week 3-4)
```
nn_core/layers/linear.py           # Linear layer
nn_core/activations/relu.py        # ReLU activation
nn_core/activations/softmax.py     # Softmax activation
optimization/losses/cross_entropy.py  # Cross-entropy loss
optimization/optimizers/sgd.py     # SGD optimizer
optimization/optimizers/adam.py    # Adam optimizer
```

### Phase 3: RL Basics (Week 5-6)
```
envs/gridworld.py                  # GridWorld environment
rl/core/policies.py                # Policy representations
rl/core/value_functions.py         # Value function classes
rl/core/replay_buffer.py           # Experience replay
rl/tabular/q_learning.py           # Q-Learning
```

### Phase 4: Deep RL (Week 7-8)
```
rl/value_based/dqn.py              # Deep Q-Network
rl/policy_gradient/reinforce.py    # REINFORCE algorithm
rl/policy_gradient/ppo.py          # PPO algorithm
rl/core/advantage.py               # GAE computation
```

### Phase 5: Generative (Week 9-10)
```
generative/autoencoders/vae.py     # Variational Autoencoder
generative/autoencoders/vanilla_ae.py  # Basic autoencoder
```

## Example Workflow

1. **Start with Q-Learning on GridWorld** (simplest)
   ```bash
   python rl_gridworld_qlearning.py
   ```
   This requires only tabular methods, no neural networks.

2. **Move to DQN on CartPole**
   ```bash
   pip install gymnasium
   python rl_dqn_cartpole.py
   ```
   This introduces deep learning + RL.

3. **Try PPO for continuous control**
   ```bash
   python rl_ppo_continuous.py
   ```
   More advanced policy gradient method.

4. **Train MLP on MNIST**
   ```bash
   python train_mlp_mnist.py
   ```
   Standard supervised learning.

5. **Train VAE on MNIST**
   ```bash
   python train_vae_mnist.py
   ```
   Introduction to generative models.

## Tips

- Each example prints its prerequisites at the top
- Examples generate plots if matplotlib is installed
- Use `pytest ../tests/` to verify implementations
- Implement gradient checking for all new layers

## Debugging

If an example fails:

1. Check if all prerequisites are implemented
2. Run the corresponding tests: `pytest ../tests/test_<module>.py`
3. Use gradient checking to verify backward passes
4. Print intermediate shapes and values

## External Dependencies

```bash
pip install numpy matplotlib
pip install gymnasium           # For RL examples
pip install scikit-learn       # For MNIST loading
pip install pytest             # For running tests
```
