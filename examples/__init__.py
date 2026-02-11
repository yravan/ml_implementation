"""
Example Scripts for ML Implementation.

This directory contains runnable examples that demonstrate
how to use the implemented modules.

Examples:
---------

1. rl_gridworld_qlearning.py
   Train Q-Learning on GridWorld
   Prerequisites: envs/gridworld, rl/tabular/q_learning

2. rl_dqn_cartpole.py
   Train DQN on CartPole-v1
   Prerequisites: rl/value_based/dqn, rl/core/replay_buffer
   External: gymnasium

3. rl_ppo_continuous.py
   Train PPO on continuous control
   Prerequisites: rl/policy_gradient/ppo
   External: gymnasium

4. train_mlp_mnist.py
   Train MLP on MNIST classification
   Prerequisites: architectures/mlp, foundations/autograd
   External: sklearn (optional)

Running Examples:
-----------------

cd examples/
python rl_gridworld_qlearning.py

Or from project root:
python -m examples.rl_gridworld_qlearning
"""
