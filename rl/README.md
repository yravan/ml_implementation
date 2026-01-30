# Reinforcement Learning Implementation Problems

## Based on MIT 6.484: Machine Learning for Sequential Decision Making

A comprehensive set of **55+ implementation problems** covering all major RL algorithms from the lecture notes, along with a practical framework for training, testing, and visualizing RL agents.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a problem (after implementation)
cd problems/ch2_bandits
python p03_ucb.py
```

## Repository Structure

```
rl/
├── rl_framework/              # Core framework code
│   ├── __init__.py
│   ├── networks.py            # Neural network architectures
│   │   ├── MLP, DiscretePolicy, GaussianPolicy
│   │   ├── SquashedGaussianPolicy (for SAC)
│   │   ├── ValueNetwork, QNetwork, DiscreteQNetwork
│   │   ├── DuelingQNetwork, TwinQNetwork
│   │   ├── ActorCritic
│   │   └── LinearUCBModel
│   ├── buffers.py             # Experience replay buffers
│   │   ├── ReplayBuffer
│   │   ├── PrioritizedReplayBuffer
│   │   ├── RolloutBuffer (for PPO/A2C)
│   │   ├── HERBuffer
│   │   └── TrajectoryBuffer
│   ├── environments.py        # Environment wrappers
│   │   ├── NormalizedEnv
│   │   ├── FrameStack, EpisodeMonitor
│   │   ├── MultiArmedBandit
│   │   ├── ContextualBandit
│   │   └── GoalConditionedWrapper
│   └── utils.py               # Utilities
│       ├── Training: set_seed, soft_update, compute_gae
│       ├── Logging: Logger, RunningMeanStd
│       ├── Visualization: plot_learning_curves, plot_regret
│       └── Evaluation: evaluate_policy, record_video
├── problems/                  # Implementation problems
│   ├── ch2_bandits/           # Problems 1-8
│   ├── ch3_mdp/               # Problems 9-11
│   ├── ch4_policy_gradient/   # Problems 12-24
│   ├── ch6_imitation/         # Problems 25-33
│   ├── ch8_model_based/       # Problems 34-38
│   ├── ch9_dp_qlearning/      # Problems 39-53
│   └── advanced/              # Problems 54-57
├── PROBLEMS.md                # Detailed problem descriptions
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Algorithms Covered

### Chapter 2: Multi-Armed Bandits
- Epsilon-Greedy, Explore-then-Commit (ETC)
- Upper Confidence Bound (UCB, UCB-Tuned, KL-UCB)
- Thompson Sampling
- LinUCB for Contextual Bandits
- Neural Contextual Bandits

### Chapter 3: MDPs
- MDP Fundamentals
- Reward Shaping
- Finite vs Infinite Horizon

### Chapter 4: Policy Gradient Methods
- REINFORCE (with baseline, reward-to-go)
- Advantage Actor-Critic (A2C, A3C)
- Generalized Advantage Estimation (GAE)
- Natural Policy Gradient (NPG)
- Trust Region Policy Optimization (TRPO)
- Proximal Policy Optimization (PPO)

### Chapter 6: Learning from Demonstrations
- Behavior Cloning (BC)
- DAgger
- Multimodal BC (Mixture of Gaussians, CVAE)
- Maximum Margin IRL
- Maximum Entropy IRL
- GAIL

### Chapter 8: Model-Based Control
- Model Predictive Control (MPC)
- Cross Entropy Method (CEM)
- Conservative CEM
- Model Learning

### Chapter 9: Value-Based Methods
- Policy Iteration, Value Iteration
- Tabular Q-Learning
- Fitted Q-Iteration (FQI)
- DQN (with Replay, Target Networks)
- Double DQN, Dueling DQN
- Prioritized Experience Replay
- DDPG, TD3, SAC

### Advanced Topics
- Hindsight Experience Replay (HER)
- Hierarchical RL
- Curriculum Learning
- Offline RL

## How to Use

### 1. Start with the Framework
First, familiarize yourself with the framework code in `rl_framework/`:

```python
# Example: Using the framework
from rl_framework.networks import DiscretePolicy, ValueNetwork
from rl_framework.buffers import ReplayBuffer, RolloutBuffer
from rl_framework.environments import make_env, MultiArmedBandit
from rl_framework.utils import set_seed, plot_learning_curves

# Create environment
env = make_env("CartPole-v1", seed=42)

# Create networks
policy = DiscretePolicy(state_dim=4, action_dim=2)
value = ValueNetwork(state_dim=4)
```

### 2. Work Through Problems
See `PROBLEMS.md` for detailed problem descriptions. Each problem has:
- Difficulty rating (⭐ to ⭐⭐⭐⭐)
- Algorithm pseudocode
- Implementation requirements
- Test environments

### 3. Recommended Order

**Beginners:**
1. Problems 1-3: Epsilon-greedy, ETC, UCB (bandits)
2. Problem 9: MDP basics
3. Problems 12-14: REINFORCE variants
4. Problems 39-43: DP and tabular Q-learning
5. Problem 25: Behavior cloning

**Intermediate:**
1. Problems 4-6: Advanced bandits, LinUCB
2. Problems 15-16, 20: A2C, GAE, PPO
3. Problems 45-48: DQN variants
4. Problems 27, 34-35: DAgger, MPC, CEM

**Advanced:**
1. Problems 18-19: NPG, TRPO
2. Problems 51-53: DDPG, TD3, SAC
3. Problems 31-33: IRL, GAIL
4. Problems 54-57: HER, HRL, Offline RL

## Testing Environments

The framework supports all Gymnasium environments:

```python
# Classic Control
"CartPole-v1", "MountainCar-v0", "Pendulum-v1", "LunarLander-v2"

# MuJoCo (requires mujoco package)
"HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"

# Custom Bandits
MultiArmedBandit(n_arms=10)
ContextualBandit(n_arms=5, context_dim=10)
```

## Example Implementations

### UCB for Bandits
```python
class UCB:
    def __init__(self, n_arms, c=2.0):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.c = c
        self.t = 0

    def select_arm(self):
        self.t += 1
        # Pull unpulled arms first
        for a in range(len(self.counts)):
            if self.counts[a] == 0:
                return a
        # UCB formula
        ucb = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb)
```

### PPO Update
```python
def ppo_update(states, actions, old_log_probs, advantages, returns):
    # Compute new log probs
    new_log_probs, entropy, values = policy(states, actions)

    # Ratio
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = F.mse_loss(values, returns)

    # Combined loss
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
    return loss
```

## Tips for Success

1. **Start simple**: Get basic algorithms working before adding improvements
2. **Visualize everything**: Plot learning curves, Q-values, policy actions
3. **Compare with baselines**: Random policy, simple heuristics
4. **Hyperparameter tuning**: Learning rate and network size matter most
5. **Use seeds**: For reproducibility during debugging
6. **Read the papers**: Original papers often have implementation details

## References

- Lecture Notes: MIT 6.484 Machine Learning for Sequential Decision Making
- Sutton & Barto: Reinforcement Learning: An Introduction (2018)
- Spinning Up in Deep RL: https://spinningup.openai.com/

## License

Educational use. Based on MIT 6.484 course materials.
