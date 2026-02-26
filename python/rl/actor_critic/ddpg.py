"""
Deep Deterministic Policy Gradient (DDPG) Algorithm

Implementation Status: Stub with comprehensive documentation
Complexity: Advanced (continuous control with deterministic policies)
Prerequisites: PyTorch/TensorFlow, NumPy, experience with RL fundamentals

DDPG (Lillicrap et al., 2015) extends policy gradient methods to continuous action
spaces by learning a deterministic policy μ(s|θ) directly, rather than a stochastic
policy. The algorithm maintains both an actor network (policy) and a critic network
(Q-function) that are learned off-policy using experience replay.

Key innovations:
1. Deterministic Policy Gradient Theorem: Enables efficient policy gradient computation
2. Experience Replay: Decorrelates experience samples for stable learning
3. Target Networks: Stabilize value function estimates
4. Ornstein-Uhlenbeck Noise: Provides smooth exploration in continuous space

The deterministic policy gradient theorem states that the gradient of expected return
with respect to policy parameters is proportional to the gradient of the Q-function
with respect to actions, evaluated at the deterministic policy.

References:
    - DDPG Paper: https://arxiv.org/abs/1509.02971
    - Policy Gradient Methods: https://arxiv.org/abs/1604.06778
    - Experience Replay: https://www.nature.com/articles/nature14236
"""

from typing import Optional, Tuple, Dict, List, Any, Callable
from dataclasses import dataclass, field
from collections import deque, namedtuple
import numpy as np
from abc import ABC, abstractmethod
import warnings


# Type definitions for clarity
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
BatchData = namedtuple('BatchData', ['states', 'actions', 'rewards', 'next_states', 'dones'])


@dataclass
class DDPGConfig:
    """
    Configuration dataclass for DDPG algorithm.

    This class encapsulates all hyperparameters needed for DDPG training. It provides
    sensible defaults based on the original paper and empirical best practices.

    Attributes:
        state_dim (int): Dimension of the state space (continuous or flattened)
        action_dim (int): Dimension of the action space (must be continuous)
        action_min (float): Minimum action value for clipping
        action_max (float): Maximum action value for clipping
        actor_learning_rate (float): Learning rate for actor network optimization
        critic_learning_rate (float): Learning rate for critic network optimization
        gamma (float): Discount factor for future rewards (0.99 is standard)
        tau (float): Soft update coefficient for target networks (typically 0.001-0.01)
        replay_buffer_size (int): Maximum size of experience replay buffer
        batch_size (int): Minibatch size for gradient updates
        num_hidden_layers (int): Number of hidden layers in actor/critic networks
        hidden_layer_size (int): Neurons per hidden layer
        l2_reg (float): L2 regularization coefficient for network weights
        gradient_clip (float): Max gradient norm for clipping (0 = no clipping)
        ou_theta (float): Mean reversion coefficient for OU noise
        ou_sigma (float): Scale of OU noise process
        exploration_noise_scale (float): Initial exploration noise magnitude
        noise_decay (float): Rate of exploration noise decay per episode
        min_noise_scale (float): Minimum exploration noise magnitude
        polyak_averaging (bool): Use polyak averaging instead of hard target updates
    """
    state_dim: int
    action_dim: int
    action_min: float = -1.0
    action_max: float = 1.0

    # Learning rates
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3

    # Discount and soft update parameters
    gamma: float = 0.99
    tau: float = 0.001

    # Experience replay
    replay_buffer_size: int = 1_000_000
    batch_size: int = 64

    # Network architecture
    num_hidden_layers: int = 2
    hidden_layer_size: int = 256
    l2_reg: float = 1e-2
    gradient_clip: float = 1.0

    # Exploration noise (Ornstein-Uhlenbeck)
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    exploration_noise_scale: float = 0.1
    noise_decay: float = 0.9999
    min_noise_scale: float = 0.01

    # Learning parameters
    polyak_averaging: bool = True

    def validate(self) -> None:
        """
        Validate configuration parameters for consistency.

        Raises:
            ValueError: If configuration parameters violate constraints
            warnings.UserWarning: If parameters are outside recommended ranges
        """
        raise NotImplementedError(
            "Implement configuration validation. Check:\n"
            "  - action_min < action_max\n"
            "  - 0 < gamma <= 1\n"
            "  - 0 < tau <= 1\n"
            "  - batch_size <= replay_buffer_size\n"
            "  - All learning rates > 0"
        )


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck Process for exploration in continuous action spaces.

    The OU process generates temporally correlated noise, which is more suitable for
    physical control tasks than white noise. It follows the SDE:

        dX_t = θ(μ - X_t)dt + σ dW_t

    where θ controls mean reversion rate, μ is the mean level, and σ is volatility.

    In discrete time:
        X_{t+1} = X_t + θ(μ - X_t)Δt + σ√Δt · ε_t

    Attributes:
        action_dim (int): Dimensionality of the action space
        theta (float): Mean reversion rate coefficient
        sigma (float): Volatility/scale of the noise
        mu (float): Mean/drift level of the process
        dt (float): Timestep for discrete approximation
        current_state (np.ndarray): Current value of the OU process
    """

    def __init__(
        self,
        action_dim: int,
        theta: float = 0.15,
        sigma: float = 0.2,
        mu: float = 0.0,
        dt: float = 1.0,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the Ornstein-Uhlenbeck noise process.

        Args:
            action_dim: Dimensionality of the action space
            theta: Mean reversion coefficient (higher = faster reversion to mean)
            sigma: Scale of the noise (higher = noisier exploration)
            mu: Mean level (typically 0 for zero-mean exploration)
            dt: Discrete timestep (typically 1.0 for single-step integration)
            seed: Random seed for reproducibility

        Theory:
            The OU process provides smooth, correlated noise suitable for continuous
            control. Unlike white noise, consecutive samples are correlated, creating
            smooth exploration trajectories that respect the temporal structure of
            control problems.
        """
        raise NotImplementedError(
            "Implement OU noise initialization:\n"
            "  - Store action_dim, theta, sigma, mu, dt\n"
            "  - Initialize current_state as zeros array of shape (action_dim,)\n"
            "  - Set random seed if provided using np.random.seed()\n"
            "  - Pre-allocate arrays for efficient noise generation"
        )

    def reset(self) -> None:
        """
        Reset the OU process state to zero.

        This should be called at the beginning of each episode to ensure
        noise doesn't carry over between independent trajectories.
        """
        raise NotImplementedError(
            "Reset the OU process:\n"
            "  - Set current_state back to zeros\n"
            "  - Can optionally use a small random initialization"
        )

    def sample(self) -> np.ndarray:
        """
        Generate next sample from the OU process.

        Returns:
            np.ndarray: Noise vector of shape (action_dim,)

        Implementation Note:
            The update equation is:
                X_{t+1} = X_t + theta*(mu - X_t)*dt + sigma*sqrt(dt)*eps_t
            where eps_t ~ N(0, I)
        """
        raise NotImplementedError(
            "Generate OU noise sample:\n"
            "  - Sample epsilon from standard normal: eps ~ N(0, I)\n"
            "  - Update state: X = X + theta*(mu - X)*dt + sigma*sqrt(dt)*eps\n"
            "  - Return the updated state\n"
            "  - Keep state for next call (temporal correlation)"
        )

    def get_noise_scale(self, episode: int, noise_scale: float, decay: float) -> float:
        """
        Compute decayed noise scale based on episode number.

        Args:
            episode: Current episode number
            noise_scale: Initial noise scale
            decay: Decay rate per episode

        Returns:
            float: Decayed noise scale
        """
        raise NotImplementedError(
            "Implement noise scale decay:\n"
            "  - Return: noise_scale * (decay ** episode)\n"
            "  - Or return: noise_scale * exp(-episode * decay_rate)"
        )


class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy learning.

    The replay buffer stores (state, action, reward, next_state, done) tuples
    from the agent's interactions with the environment. During training, minibatches
    are sampled uniformly at random from this buffer, which decorrelates the temporal
    structure of the data and stabilizes learning.

    Key properties:
    - Stores up to max_size experiences (oldest dropped when full)
    - Efficient sampling via uniform random selection
    - Converts Python lists to NumPy arrays for efficient computation

    The experience replay mechanism is crucial for off-policy learning as it:
    1. Breaks temporal correlations in the data
    2. Allows reuse of old experiences multiple times
    3. Enables more efficient learning from limited data
    4. Reduces variance in gradient estimates

    Attributes:
        max_size (int): Maximum capacity of the buffer
        buffer (deque): Efficient container for bounded circular buffer
        seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        max_size: int = 1_000_000,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the replay buffer.

        Args:
            max_size: Maximum number of experiences to store
            seed: Random seed for reproducible sampling

        Theory:
            A deque with maxlen automatically maintains FIFO (first-in-first-out)
            behavior, dropping the oldest experience when capacity is exceeded.
        """
        raise NotImplementedError(
            "Initialize replay buffer:\n"
            "  - Create deque with maxlen=max_size for automatic FIFO eviction\n"
            "  - Set random seed if provided\n"
            "  - Store max_size and seed as instance variables"
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add an experience tuple to the buffer.

        Args:
            state: Current state observation (np.ndarray of shape (state_dim,))
            action: Action taken (np.ndarray of shape (action_dim,))
            reward: Scalar reward received
            next_state: Resulting state (np.ndarray of shape (state_dim,))
            done: Whether episode terminated or truncated

        Implementation:
            Create an Experience namedtuple and append to buffer. The deque
            automatically handles eviction when capacity is exceeded.
        """
        raise NotImplementedError(
            "Add experience to buffer:\n"
            "  - Create Experience(state, action, reward, next_state, done)\n"
            "  - Append to self.buffer\n"
            "  - Handle automatic capacity management via deque"
        )

    def sample(self, batch_size: int) -> BatchData:
        """
        Sample a minibatch of experiences uniformly at random.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            BatchData: Named tuple containing:
                - states: (batch_size, state_dim) array
                - actions: (batch_size, action_dim) array
                - rewards: (batch_size,) array
                - next_states: (batch_size, state_dim) array
                - dones: (batch_size,) boolean array

        Implementation:
            Use np.random.choice or random.sample to select batch_size indices
            uniformly at random (with replacement). Stack the selected experiences
            into contiguous arrays for efficient batch processing.
        """
        raise NotImplementedError(
            "Sample minibatch from buffer:\n"
            "  - Check buffer has at least batch_size experiences\n"
            "  - Sample batch_size random indices (with replacement)\n"
            "  - Extract and stack experiences into numpy arrays\n"
            "  - Return BatchData with all components as arrays\n"
            "  - Handle edge case: buffer might be smaller than batch_size early training"
        )

    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        raise NotImplementedError("Return len(self.buffer)")

    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for meaningful learning."""
        raise NotImplementedError(
            "Return True if buffer size >= batch_size (typically waiting until\n"
            "buffer has at least batch_size samples before training begins)"
        )


class ActorNetwork(ABC):
    """
    Abstract base class for deterministic policy (actor) networks.

    The actor network represents the deterministic policy μ(s|θ) that maps states
    to continuous actions. Unlike stochastic policies, this outputs a single action
    value per state, which is then used to compute the policy gradient.

    Architecture (typical):
        Input: state (continuous, normalized)
        Hidden: 2-3 fully connected layers with ReLU activation
        Output: action vector with tanh activation (squashed to [-1, 1])

    The tanh output layer is crucial because:
    1. It produces bounded outputs suitable for continuous control
    2. It's differentiable (required for policy gradient computation)
    3. The actions can be rescaled to the environment's action bounds

    Properties:
        trainable: Whether network parameters are trainable
        device: Computation device (CPU/GPU)
    """

    @abstractmethod
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Compute deterministic policy action for given state(s).

        Args:
            state: State observation(s)
                - Shape: (state_dim,) for single state
                - Shape: (batch_size, state_dim) for batch

        Returns:
            np.ndarray: Deterministic action(s)
                - Shape: (action_dim,) for single state
                - Shape: (batch_size, action_dim) for batch
                - Values in approximately [-1, 1] range (due to tanh output)

        Theory:
            This implements the deterministic policy μ(s|θ). The output is a
            deterministic function of the state, unlike stochastic policies that
            output probability distributions.
        """
        pass

    @abstractmethod
    def backward(self, loss: float) -> None:
        """
        Perform backpropagation to update actor parameters.

        Args:
            loss: Scalar loss value to minimize

        Theory:
            This implements the deterministic policy gradient update:
                θ ← θ + α∇_θ J(θ)
            where ∇_θ J(θ) is computed via chain rule through the critic.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get network parameters as dictionary."""
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set network parameters from dictionary."""
        pass


class CriticNetwork(ABC):
    """
    Abstract base class for value function (critic) networks.

    The critic network represents the Q-function Q(s,a|φ) that estimates the
    expected return from taking action a in state s under the current policy.
    It takes both state and action as input (state-action value function).

    Architecture (typical):
        Input: concatenation of state and action
        Hidden: 2-3 fully connected layers with ReLU activation
        Output: single scalar Q-value estimate

    Why concatenate state and action:
    1. Q(s,a) depends on both state and action together
    2. Unlike actor (deterministic), action is provided as input
    3. Enables efficient batch computation with arbitrary actions

    Key roles in DDPG:
    1. Estimates return Q(s,a) for learning actor parameters
    2. Provides TD target: r + γ Q(s',μ(s')) for Bellman update
    3. Uses target network for stability

    Properties:
        trainable: Whether parameters are trainable
        device: Computation device (CPU/GPU)
    """

    @abstractmethod
    def forward(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Estimate Q-value for state-action pairs.

        Args:
            state: State observation(s)
                - Shape: (state_dim,) or (batch_size, state_dim)
            action: Action(s)
                - Shape: (action_dim,) or (batch_size, action_dim)

        Returns:
            np.ndarray: Q-value estimate(s)
                - Shape: () scalar for single state-action
                - Shape: (batch_size,) for batch

        Theory:
            Computes Q_φ(s,a) as an approximation of Q^π(s,a), the expected
            cumulative discounted reward from state s, taking action a, then
            following the current policy.
        """
        pass

    @abstractmethod
    def backward(self, loss: float) -> None:
        """
        Perform backpropagation to update critic parameters.

        Args:
            loss: Scalar loss value to minimize

        Theory:
            This updates parameters φ via gradient descent on the TD error:
                φ ← φ - β∇_φ (r + γQ(s',μ(s')) - Q(s,a))²
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get network parameters as dictionary."""
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set network parameters from dictionary."""
        pass


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent.

    DDPG is an off-policy actor-critic algorithm for continuous control that learns
    a deterministic policy μ(s|θ) and a critic Q(s,a|φ). It combines three key ideas:

    1. **Deterministic Policy Gradient Theorem (DPG)**:
       The gradient of expected return under a deterministic policy is:
           ∇_θ J(θ) = E[∇_a Q(s,a|φ)|_{a=μ(s|θ)} ∇_θ μ(s|θ)]

       This is more sample efficient than stochastic policy gradients because:
       - The integral over actions is avoided
       - Only the policy gradient, not the expectation over actions, needs integration
       - Results in lower-variance gradient estimates

    2. **Experience Replay**:
       Stores transitions in a replay buffer and samples minibatches uniformly:
       - Breaks temporal correlations in the data
       - Enables reuse of experiences
       - Reduces variance in gradient estimates
       - Allows off-policy learning (data can come from any policy)

    3. **Target Networks**:
       Maintains separate target networks for both actor and critic:
       - Q'(s,a) = fixed copy of critic for computing TD targets
       - μ'(s) = fixed copy of actor for computing targets
       - Updated every C steps via Polyak averaging: θ' ← τθ + (1-τ)θ'
       - Reduces moving target problem and stabilizes learning

    Update Rules:

    **Critic Update** (TD learning):
        L(φ) = E[(r + γQ'(s',μ'(s')|φ') - Q(s,a|φ))²]
        φ ← φ - β∇_φ L(φ)

    **Actor Update** (DPG):
        J(θ) = E[Q(s,μ(s|θ)|φ)]
        θ ← θ + α∇_θ J(θ) = θ + α∇_θ Q(s,μ(s|θ)|φ)
              = θ + α E[∇_a Q(s,a|φ)|_{a=μ(s)} ∇_θ μ(s|θ)]

    **Target Network Updates** (Polyak averaging):
        φ' ← τφ + (1-τ)φ'  (soft update, typically τ = 0.001)
        θ' ← τθ + (1-τ)θ'

    Exploration:
    - Uses Ornstein-Uhlenbeck process for smooth exploration noise
    - Adds to deterministic policy: a_t = μ(s_t|θ) + N_t
    - Noise decays during training as policy improves

    Key Hyperparameters:
    - Learning rates: α (actor), β (critic)
    - Discount: γ (typically 0.99)
    - Soft update: τ (typically 0.001)
    - Replay buffer: size (1M typical), batch_size (64 typical)
    - Exploration: OU process parameters, noise decay

    Advantages:
    - Sample efficient (off-policy learning with experience replay)
    - Works well in high-dimensional continuous control
    - Deterministic policy reduces variance vs stochastic policies
    - Stable training with target networks

    Limitations:
    - Function approximation error can accumulate (addressed by TD3)
    - Requires careful hyperparameter tuning
    - Can overestimate Q-values in some cases
    - Exploration can be insufficient with decaying noise

    References:
        - DDPG Paper: https://arxiv.org/abs/1509.02971
        - DPG Theorem: https://arxiv.org/abs/1604.06778
        - Continuous Control Benchmark: https://openai.com/research/benchmarking-deep-reinforcement-learning/
    """

    def __init__(
        self,
        config: DDPGConfig,
        actor_network: ActorNetwork,
        critic_network: CriticNetwork,
        device: str = "cpu"
    ) -> None:
        """
        Initialize DDPG agent with networks and configuration.

        Args:
            config: DDPGConfig instance with all hyperparameters
            actor_network: ActorNetwork instance (deterministic policy μ)
            critic_network: CriticNetwork instance (Q-function)
            device: "cpu" or "cuda" for computation

        Implementation:
            - Validate configuration
            - Initialize actor and critic networks
            - Create target networks (copies for stability)
            - Initialize experience replay buffer
            - Initialize OU exploration process
            - Set up optimizers
        """
        raise NotImplementedError(
            "Initialize DDPG agent:\n"
            "  1. Validate config via config.validate()\n"
            "  2. Store device, config, networks\n"
            "  3. Create target networks as copies of actor/critic\n"
            "  4. Initialize replay buffer with config.replay_buffer_size\n"
            "  5. Initialize OU noise process\n"
            "  6. Create optimizers for actor and critic\n"
            "  7. Initialize episode counter and training step counter"
        )

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Select action from policy, with optional exploration noise.

        Args:
            state: Current state observation (shape: (state_dim,))
            training: Whether in training mode (adds exploration noise) or eval mode

        Returns:
            np.ndarray: Action to take (shape: (action_dim,))
                - Range approximately [-action_max, action_max]

        Algorithm:
            1. Compute deterministic policy: a = μ(s|θ)
            2. If training: add exploration noise a ← a + N_t
            3. Clip to action bounds: a ← clip(a, action_min, action_max)
            4. Return action

        Theory:
            The deterministic policy output (tanh) is scaled to match environment
            action bounds. Exploration noise enables discovery of better policies.
            The noise is decayed as training progresses (reduced exploration).
        """
        raise NotImplementedError(
            "Select action from policy:\n"
            "  1. Convert state to tensor, add batch dimension if needed\n"
            "  2. Call actor network: action = actor.forward(state)\n"
            "  3. If training:\n"
            "    a. Get current noise scale (with decay)\n"
            "    b. Sample OU noise: noise = OU.sample()\n"
            "    c. Add scaled noise: action += noise_scale * noise\n"
            "  4. Clip to [action_min, action_max]\n"
            "  5. Return action as numpy array"
        )

    def update(self, batch_size: int = None) -> Dict[str, float]:
        """
        Perform one step of DDPG training.

        Args:
            batch_size: Minibatch size (uses config.batch_size if None)

        Returns:
            Dict[str, float]: Training metrics
                - "critic_loss": MSE loss on TD target
                - "actor_loss": Negative mean Q-value (we maximize Q)
                - "q_mean": Mean Q-value estimate for logging

        Algorithm:
            1. Sample minibatch from replay buffer
            2. Compute TD targets using target networks
            3. Update critic via gradient descent on TD loss
            4. Update actor via deterministic policy gradient
            5. Update target networks via Polyak averaging

        Mathematical Details:

        **Critic Update**:
            y = r + γQ'(s',μ'(s')|φ')  [TD target]
            L(φ) = (1/N)Σ(y - Q(s,a|φ))²  [MSE loss]
            φ ← φ - β∇_φ L(φ)

        **Actor Update**:
            J(θ) = (1/N)Σ Q(s,μ(s|θ)|φ)  [maximize Q-values]
            θ ← θ + α∇_θ J(θ)  [gradient ascent on J]

        **Target Network Update**:
            φ' ← τφ + (1-τ)φ'
            θ' ← τθ + (1-τ)θ'

        Why this order:
            - Update critic first: it provides gradient signal for actor
            - Update actor: using fresh critic estimates
            - Update targets: stabilizes both networks
        """
        raise NotImplementedError(
            "Perform DDPG update step:\n"
            "  1. Check if replay buffer has enough samples\n"
            "  2. Sample minibatch: states, actions, rewards, next_states, dones\n"
            "  3. Compute TD target:\n"
            "    - next_actions = target_actor(next_states)\n"
            "    - y = rewards + gamma*(1-dones)*target_critic(next_states, next_actions)\n"
            "  4. Update critic:\n"
            "    - q_pred = critic(states, actions)\n"
            "    - loss = MSE(q_pred, y)\n"
            "    - critic.backward(loss)\n"
            "  5. Update actor:\n"
            "    - actions_from_policy = actor(states)\n"
            "    - q_values = critic(states, actions_from_policy)\n"
            "    - loss = -mean(q_values)  [maximize Q]\n"
            "    - actor.backward(loss)\n"
            "  6. Update target networks with Polyak averaging:\n"
            "    - for param, target_param in zip(critic.params, target_critic.params):\n"
            "        target_param = tau*param + (1-tau)*target_param\n"
            "  7. Return metrics dict with losses and Q-value mean"
        )

    def train_episode(
        self,
        env: Any,
        max_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Execute one training episode with environment interaction.

        Args:
            env: Environment instance with standard Gym API
                (reset() -> state, step(action) -> (state, reward, done, info))
            max_steps: Maximum steps per episode

        Returns:
            Dict[str, float]: Episode metrics
                - "episode_reward": Total undiscounted reward
                - "episode_length": Number of steps taken
                - "critic_loss": Average critic loss
                - "actor_loss": Average actor loss

        Algorithm:
            1. Reset environment and OU noise
            2. Loop until episode ends:
                a. Select action with exploration
                b. Take step in environment
                c. Store experience in replay buffer
                d. Perform learning update (if buffer ready)
            3. Log episode statistics
            4. Decay exploration noise
            5. Return episode metrics

        Training Loop Pseudocode:
            state = env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)

                if self.replay_buffer.is_ready:
                    metrics = self.update(batch_size)

                episode_reward += reward
                state = next_state
                if done:
                    break

            self.ou_noise.reset()
            return {episode metrics}
        """
        raise NotImplementedError(
            "Execute one training episode:\n"
            "  1. Reset environment: state, info = env.reset()\n"
            "  2. Reset exploration: self.ou_noise.reset()\n"
            "  3. Loop for max_steps or until done:\n"
            "    a. Select action: action = self.select_action(state, training=True)\n"
            "    b. Step environment: next_state, reward, terminated, truncated, info = env.step(action)\n"
            "    c. Compute done = terminated or truncated\n"
            "    d. Add to replay buffer\n"
            "    e. If buffer ready, call self.update() and track metrics\n"
            "    f. Update state and accumulate reward\n"
            "    g. Break if done\n"
            "  4. Increment episode counter\n"
            "  5. Return dict with episode_reward, episode_length, avg losses"
        )

    def eval_episode(
        self,
        env: Any,
        max_steps: int = 1000,
        render: bool = False
    ) -> float:
        """
        Execute one evaluation episode without learning.

        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            render: Whether to render environment

        Returns:
            float: Total episode reward (undiscounted)

        Algorithm:
            1. Reset environment (no noise)
            2. Loop until episode ends:
                a. Select action from policy (no noise)
                b. Take step in environment
                c. Optionally render
            3. Return total reward
        """
        raise NotImplementedError(
            "Execute evaluation episode (deterministic policy, no learning):\n"
            "  1. Set actor to eval mode (no gradients)\n"
            "  2. Reset environment\n"
            "  3. Loop:\n"
            "    - Select action with training=False (no noise)\n"
            "    - Step environment\n"
            "    - Accumulate reward\n"
            "    - Break if done\n"
            "  4. Return accumulated reward"
        )

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent state to checkpoint file.

        Args:
            filepath: Path to save checkpoint (typically .pt or .pkl)

        Implementation:
            Save actor, critic, target networks, replay buffer state,
            and all hyperparameters needed for full restoration.
        """
        raise NotImplementedError(
            "Save checkpoint:\n"
            "  - Save actor network parameters\n"
            "  - Save critic network parameters\n"
            "  - Save target actor/critic parameters\n"
            "  - Save config\n"
            "  - Save optimizer states\n"
            "  - Use torch.save or pickle depending on framework"
        )

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load agent state from checkpoint file.

        Args:
            filepath: Path to checkpoint file
        """
        raise NotImplementedError(
            "Load checkpoint:\n"
            "  - Load all network parameters\n"
            "  - Load optimizer states\n"
            "  - Verify loaded config matches current config"
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information and statistics.

        Returns:
            Dict with agent state, network architecture, training statistics
        """
        raise NotImplementedError(
            "Return agent info dict with:\n"
            "  - episode_count\n"
            "  - total_steps\n"
            "  - network architectures\n"
            "  - buffer size\n"
            "  - recent metrics"
        )
