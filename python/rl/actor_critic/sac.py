"""
Soft Actor-Critic (SAC) Algorithm

Implementation Status: Stub with comprehensive documentation
Complexity: Advanced (continuous control with entropy regularization)
Prerequisites: DDPG/TD3 understanding, PyTorch/TensorFlow, probabilistic RL

SAC (Haarnoja et al., 2018) is a state-of-the-art off-policy algorithm combining
three core ideas:

1. **Maximum Entropy Reinforcement Learning**:
   Augment reward with entropy term to encourage exploration:
       J(π) = E[Σ γ^t (r + α H(π(·|s)))]
   where H(π(·|s)) = -E[log π(a|s)] is the policy entropy

2. **Stochastic Policy Gradient**:
   Unlike DDPG/TD3 (deterministic policies), SAC learns a stochastic policy π(a|s).
   This provides natural exploration without separate noise processes.

3. **Automatic Entropy Coefficient Tuning**:
   Instead of fixed entropy weight α, learns it automatically:
       J(α) = E[-α log π(a|s) - α H_target]
   Controls exploration-exploitation trade-off automatically

Key Components:

- **Actor (Policy Network)**: Maps states to action distributions π(a|s)
- **Two Q-Networks**: Dual critics like TD3 to address overestimation
- **Value Network**: Optional, but improves stability
- **Entropy Coefficient**: Learned online or set to constant

Mathematical Framework:

**SAC Objective** (combining value and policy):
    J(θ,φ) = E[Q(s,a|φ) - α log π(a|s|θ)]

This objective balances:
- Maximizing expected Q-values (performance)
- Maximizing policy entropy (exploration)

The entropy term -α log π(a|s) encourages stochastic actions. High entropy means
uniform action distribution (uniform exploration). Low entropy means concentrated
distribution (exploitation).

**Q-Function Update** (similar to TD3 but using stochastic target policy):
    y = r + γ(1-d) * (min(Q'_1(s',a'|φ'_1), Q'_2(s',a'|φ'_2)) - α log π(a'|s'|θ'))
    Loss_i = E[(y - Q_i(s,a|φ_i))²]

The key difference from TD3: target action a' is sampled from π(·|s') not computed
deterministically. This is more robust and requires less exploration noise.

**Actor Update** (policy gradient with entropy regularization):
    J(θ) = E[α log π(a|s|θ) - Q(s,a|φ)]
    θ ← θ - α∇_θ J(θ)

The negative log-probability term pushes the policy towards high-entropy distributions.
The Q-value term pulls the policy towards high-value actions.

**Entropy Coefficient Update** (automatic tuning):
    α ← α + β∇_α [log α + α(H_target - E[log π(a|s)])]

This automatically adjusts α to maintain target entropy level.

Why Entropy Helps:

1. **Exploration**: Higher entropy = more diverse actions = better exploration
2. **Robustness**: Stochastic policies are more robust than deterministic ones
3. **Reward Shaping**: Entropy regularization prevents premature convergence
4. **Multi-Modal Tasks**: Entropy allows learning multiple good policies

Advantages over DDPG/TD3:

1. **Better Exploration**: Natural exploration through stochasticity, not added noise
2. **Automatic Entropy**: Self-adjusting exploration via α learning
3. **Stability**: Multiple sources of randomness (stochastic policy + Q-functions)
4. **Robustness**: Works across diverse continuous control tasks
5. **Elegant Framework**: Principled maximum entropy RL objective

Comparison with TD3:
- TD3: Deterministic policy + exploration noise + twin critics
- SAC: Stochastic policy + entropy regularization + twin critics
- SAC usually more sample efficient in early training
- TD3 sometimes better final performance with proper tuning

Implementation Considerations:

**Reparameterization Trick**:
    Instead of sampling a ~ π(·|s), use reparameterization:
        a = μ(s) + σ(s) ⊙ ε,  ε ~ N(0, I)
    This allows gradients to flow through the sampling operation:
        ∇_θ E[f(a)] = E[∇_θ f(a)]

**Entropy Calculation**:
    H(π) = -E[log π(a|s)]
    Can be computed analytically for Gaussian policies:
        H(N(μ,σ²)) = 0.5*log(2πeσ²)
    Or estimated via samples during training

**Target Entropy**:
    Default target entropy: H_target = -action_dim
    Encourages actions to explore over all dimensions
    Can be tuned: higher → more exploration, lower → faster convergence

**Value Network** (optional):
    Can include separate V(s) network for stability:
        V(s) = min(Q_1(s,a~), Q_2(s,a~)) - α log π(a~|s)
    Avoids computing Q-values for many action samples

Network Architecture:

1. **Actor (Policy Network)**:
   Input: state
   Output: mean μ(s), std σ(s) of Gaussian distribution
   Policy: a = tanh(μ + σ*ε) where ε ~ N(0,I)

2. **Q-Networks** (two identical networks):
   Input: [state, action]
   Output: scalar Q-value
   Same as TD3

3. **Optional Value Network**:
   Input: state only
   Output: scalar V-value
   Used in some implementations for stability

Key Hyperparameters:

- Learning rates: α_actor, α_critic (separate from entropy coefficient)
- Gamma: discount factor
- Tau: soft update rate for targets
- Target entropy: H_target = -action_dim (auto-tune α around this)
- Alpha learning rate: β (for automatic entropy tuning)
- Entropy coefficient bounds: min_alpha, max_alpha

Empirical Performance:
- Outperforms DDPG/TD3 on many continuous control benchmarks
- More stable training with fewer divergent runs
- Better sample efficiency in early training
- Works well with both fixed and learned entropy coefficients
- Good performance across diverse task properties

Known Issues & Solutions:
1. Too much entropy: Reduce target entropy or manually set α
2. Too little entropy: Increase target entropy or warm-start α
3. Unstable critic: Standard TD3 fixes work (gradient clipping, etc.)
4. Slow convergence: Decrease tau (faster target updates) or increase entropy

References:
    - SAC Paper: https://arxiv.org/abs/1801.01290
    - SAC with Automatic Entropy: https://arxiv.org/abs/1812.05905
    - Maximum Entropy RL: https://arxiv.org/abs/1705.10528
    - Reparameterization Trick: https://arxiv.org/abs/1312.6114
    - Benchmarks: https://openai.com/research/benchmarking-deep-rl/
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import warnings


@dataclass
class SACConfig:
    """
    Configuration dataclass for SAC algorithm.

    Combines continuous control hyperparameters with maximum entropy regularization
    parameters. SAC learns a stochastic policy which provides natural exploration.

    Attributes:
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
        action_min (float): Minimum action value
        action_max (float): Maximum action value

        # Learning rates
        actor_learning_rate (float): Learning rate for policy network
        critic_learning_rate (float): Learning rate for Q-networks
        alpha_learning_rate (float): Learning rate for entropy coefficient α
        value_learning_rate (float): Learning rate for value network (optional)

        # Learning parameters
        gamma (float): Discount factor (typically 0.99)
        tau (float): Soft target update coefficient (typically 0.005)
        replay_buffer_size (int): Experience replay buffer capacity
        batch_size (int): Minibatch size for training

        # Network architecture
        num_hidden_layers (int): Number of hidden layers in networks
        hidden_layer_size (int): Neurons per hidden layer
        l2_reg (float): L2 regularization on network weights
        gradient_clip (float): Max gradient norm for clipping

        # Maximum Entropy RL
        target_entropy (float): Target entropy level for auto-tuning
                               Default: -action_dim (encourage exploration over all dims)
        initial_alpha (float): Initial entropy coefficient value
        learnable_alpha (bool): Whether to learn α or use fixed value
        alpha_min (float): Minimum allowed α value
        alpha_max (float): Maximum allowed α value

        # Policy distribution
        use_gaussian_policy (bool): Use Gaussian policy (vs other distributions)
        squashed_output (bool): Use tanh to squash actions to [-1, 1]

        # Optional components
        use_value_network (bool): Include separate value network V(s)
    """
    state_dim: int
    action_dim: int
    action_min: float = -1.0
    action_max: float = 1.0

    # Learning rates
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4

    # Learning parameters
    gamma: float = 0.99
    tau: float = 0.005

    # Experience replay
    replay_buffer_size: int = 1_000_000
    batch_size: int = 64

    # Network architecture
    num_hidden_layers: int = 2
    hidden_layer_size: int = 256
    l2_reg: float = 1e-2
    gradient_clip: float = 1.0

    # Maximum Entropy RL parameters
    target_entropy: Optional[float] = None  # Defaults to -action_dim if None
    initial_alpha: float = 0.2
    learnable_alpha: bool = True
    alpha_min: float = 1e-5
    alpha_max: float = 10.0

    # Policy parameters
    use_gaussian_policy: bool = True
    squashed_output: bool = True

    # Optional components
    use_value_network: bool = False

    def __post_init__(self) -> None:
        """Set default target entropy based on action dimension."""
        if self.target_entropy is None:
            self.target_entropy = -self.action_dim

    def validate(self) -> None:
        """Validate configuration parameters."""
        raise NotImplementedError(
            "Validate SAC configuration:\n"
            "  - Check all learning rates > 0\n"
            "  - Check 0 < gamma <= 1\n"
            "  - Check 0 < tau <= 1\n"
            "  - Check initial_alpha > 0\n"
            "  - Check alpha_min < alpha_max\n"
            "  - Warn if alpha_min is very small (numerical issues)"
        )


class SACActorNetwork(ABC):
    """
    Abstract base class for SAC stochastic policy networks.

    Unlike DDPG/TD3 which output deterministic actions, SAC's actor outputs
    the parameters of a probability distribution over actions (typically Gaussian).

    The network learns:
    - mean μ(s): Preferred action direction
    - log standard deviation log σ(s): Exploration magnitude

    Architecture:
        Input: state (continuous, normalized)
        Hidden: ReLU layers
        Output: [mean (action_dim,), log_std (action_dim,)]

    The policy is defined as:
        π(a|s) = N(μ(s), σ(s)²) with tanh squashing
        a = tanh(μ(s) + σ(s) ⊙ ε)  where ε ~ N(0, I)

    The tanh squashing:
    - Maps unbounded Gaussian to bounded [-1, 1] range
    - Requires log-probability correction via change of variables
    - Encourages bounded actions suitable for physical systems

    Key Properties:
    - Stochastic (explores naturally via sampling)
    - Differentiable (for reparameterization trick)
    - Output bounded (via tanh squashing)
    - Entropy tractable (analytical for Gaussian)
    """

    @abstractmethod
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute policy parameters: mean and log standard deviation.

        Args:
            state: State observation(s)
                Shape: (state_dim,) or (batch_size, state_dim)

        Returns:
            Tuple of (mean, log_std):
            - mean: Policy mean μ(s), shape (action_dim,) or (batch_size, action_dim)
            - log_std: Log standard deviation log σ(s), same shape as mean

        Theory:
            The policy is a diagonal Gaussian: π(a|s) = N(μ(s), diag(σ(s)²))
            We output log σ instead of σ for numerical stability.
            σ(s) = exp(log_std(s)), ensuring σ > 0
        """
        pass

    @abstractmethod
    def sample_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample action from policy and compute log probability.

        Args:
            state: State observation(s)
            deterministic: If True, use mean without sampling (for evaluation)

        Returns:
            Tuple of (action, log_prob):
            - action: Sampled action(s), squashed to [-1, 1]
            - log_prob: Log probability of action under policy

        Algorithm:
            1. Compute mean and log_std from network: μ, log σ = forward(s)
            2. If deterministic: return (tanh(μ), log_prob(tanh(μ)))
            3. If stochastic:
               a. Sample ε ~ N(0, I)
               b. Compute unsquashed action: a_raw = μ + σ⊙ε
               c. Squash with tanh: a = tanh(a_raw)
               d. Compute log probability (with correction for tanh squashing)
               e. Return (a, log_prob)

        Log Probability Correction:
            The change of variables from a_raw to a = tanh(a_raw) introduces
            a Jacobian term:
                log π(a|s) = log π(a_raw|s) - Σ log(1 - a²)
            This correction ensures probabilities integrate to 1 over action space.
        """
        pass

    @abstractmethod
    def backward(self, loss: float) -> None:
        """Perform backpropagation to update policy parameters."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get network parameters."""
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set network parameters."""
        pass


class SACCriticNetwork(ABC):
    """
    Abstract base class for SAC Q-function networks.

    Identical to TD3's critic networks. SAC uses twin critics to address
    overestimation bias in Q-learning, same as TD3.

    Architecture:
        Input: concatenated [state, action]
        Hidden: ReLU layers
        Output: scalar Q-value estimate

    Both networks learn independently, with targets computed as:
        y = r + γ(1-d) * (min(Q'_1, Q'_2) - α log π(a'|s'))

    The stochastic target policy (sampling from π) is more robust than
    DDPG's deterministic policy, and the entropy term encourages exploration.
    """

    @abstractmethod
    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Estimate Q-value for state-action pairs."""
        pass

    @abstractmethod
    def backward(self, loss: float) -> None:
        """Perform backpropagation."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get network parameters."""
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set network parameters."""
        pass


class SACValueNetwork(ABC):
    """
    Optional value network for SAC stability.

    Some SAC implementations include a separate value network V(s) to
    avoid computing Q-values for many action samples. This network learns:

        V(s) = E_a~π[Q(s,a) - α log π(a|s)]

    This is the expected return adjusted for entropy, which helps stabilize
    critic learning and reduces variance.

    Architecture:
        Input: state
        Hidden: ReLU layers
        Output: scalar value estimate

    When used, the Q-targets become:
        y = r + γ(1-d) * V(s')

    Instead of sampling actions from π(·|s') to compute Q(s',a'), we use
    the pre-computed value function.
    """

    @abstractmethod
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Estimate state value V(s)."""
        pass

    @abstractmethod
    def backward(self, loss: float) -> None:
        """Perform backpropagation."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get network parameters."""
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set network parameters."""
        pass


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent for continuous control.

    SAC combines maximum entropy reinforcement learning with actor-critic methods
    to achieve excellent performance on continuous control tasks. The key insight
    is to learn a stochastic policy that maximizes expected return PLUS policy
    entropy, encouraging both high performance and exploration.

    Core Algorithm:

    **Objective Function** (Maximum Entropy RL):
        J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
        where H(π) = -E[log π(a|s)] is the differential entropy

    This is rewritten as:
        J(π, Q) = E[Q(s,a) - α log π(a|s)]

    The objective combines:
    - Q(s,a): Temporal difference target (performance)
    - -α log π(a|s): Entropy penalty (exploration)

    When α is high → policy explores uniformly
    When α is low → policy exploits high-return actions

    **Training Loop**:

    Every step:
    1. **Critic Update** (both Q-networks):
       Sample batch from replay buffer
       Sample target action from stochastic policy: a' ~ π(·|s')
       Compute target:
           y = r + γ(1-d) * (min(Q'_1(s',a'), Q'_2(s',a')) - α log π(a'|s'))
       Update critics:
           Loss_i = ||y - Q_i(s,a)||²
           φ_i ← φ_i - β∇_φ_i Loss_i

    2. **Actor Update** (policy network):
       Sample actions from current policy: a ~ π(·|s)
       Compute policy loss:
           J(θ) = E[α log π(a|s) - Q(s,a)]
       Update actor:
           θ ← θ - β∇_θ J(θ)

    3. **Entropy Coefficient Update** (if learnable):
       Update α to maintain target entropy:
           J(α) = E[-α log π(a|s) - α H_target]
           α ← α - β_α∇_α J(α)

    4. **Target Network Updates**:
       φ'_i ← τφ_i + (1-τ)φ'_i
       θ' ← τθ + (1-τ)θ'

    **Entropy Regularization**:

    The entropy term -α log π(a|s) encourages the policy to be stochastic:
    - log π(a|s) is negative (probabilities < 1), so -log π is positive
    - Higher entropy → more uniform action distribution
    - Automatic entropy tuning adjusts α to maintain desired entropy level

    **Target Entropy Concept**:
    Default: H_target = -action_dim
    This means the target entropy is set so the policy explores equally over
    all action dimensions. For a D-dimensional Gaussian:
        H(N(μ,σ²)) = 0.5*log(2πeσ²) ≈ 0.5*log(e) = 0.5 for σ²≈1
    But we typically want H_target = -D to encourage more exploration.

    **Advantages of Stochastic Policy**:
    1. Natural exploration: No need for separate noise process
    2. Robust: Multiple action samples reduce value estimation errors
    3. Elegant: Principled maximum entropy objective
    4. Automatic tuning: α learning removes one hyperparameter
    5. Good generalization: Entropy regularization prevents overfitting to specific trajectories

    **Comparison with DDPG/TD3**:
    DDPG/TD3:
    - Deterministic policy μ(s) + exploration noise
    - Simple but requires careful noise tuning
    - Lower exploration early in training
    TD3:
    - Adds twin critics and delayed updates
    - More stable but more complex
    SAC:
    - Stochastic policy with entropy regularization
    - Natural exploration and auto-tuned entropy
    - Excellent performance across diverse tasks
    - Slightly higher computational cost (samples from policy)

    **Hyperparameter Tuning**:
    - Learning rates: Usually same for actor and critic (e.g., 3e-4)
    - Alpha learning rate: Often smaller, e.g., 3e-4
    - Target entropy: Default -action_dim works well, can adjust for more/less exploration
    - Tau: 0.005 is typical (slower target updates than TD3's 0.001)
    - Batch size: 64-256 typical

    **Implementation Tips**:
    1. Use reparameterization trick for differentiable sampling
    2. Compute log probabilities with tanh Jacobian correction
    3. Clamp log_std to prevent extreme values
    4. Use gradient clipping for stability
    5. Monitor entropy to verify auto-tuning is working

    References:
        - SAC Paper: https://arxiv.org/abs/1801.01290
        - Automatic Entropy: https://arxiv.org/abs/1812.05905
        - Maximum Entropy RL: https://arxiv.org/abs/1705.10528
        - Continuous Control Benchmark: https://openai.com/research/benchmarking-deep-rl/
        - SAC + Other Improvements: https://arxiv.org/abs/1910.07207
    """

    def __init__(
        self,
        config: SACConfig,
        actor_network: SACActorNetwork,
        critic_network_1: SACCriticNetwork,
        critic_network_2: SACCriticNetwork,
        value_network: Optional[SACValueNetwork] = None,
        device: str = "cpu"
    ) -> None:
        """
        Initialize SAC agent with stochastic policy and dual critics.

        Args:
            config: SACConfig instance with all hyperparameters
            actor_network: SACActorNetwork instance (stochastic policy)
            critic_network_1: First Q-function network
            critic_network_2: Second Q-function network
            value_network: Optional separate value network for stability
            device: "cpu" or "cuda" for computation

        Implementation:
            - Validate configuration
            - Store networks and config
            - Create target networks for critics
            - Initialize experience replay buffer
            - Create optimizers for all networks
            - Initialize entropy coefficient α
            - Initialize value network optimizer if used
        """
        raise NotImplementedError(
            "Initialize SAC agent:\n"
            "  1. Validate config\n"
            "  2. Store device, config, networks\n"
            "  3. Create target networks:\n"
            "     - critic_1_target, critic_2_target\n"
            "     - Optionally: value_network_target\n"
            "  4. Initialize replay buffer\n"
            "  5. Create optimizers:\n"
            "     - actor_optimizer\n"
            "     - critic_1_optimizer\n"
            "     - critic_2_optimizer\n"
            "     - value_optimizer (if used)\n"
            "  6. Initialize entropy coefficient:\n"
            "     - alpha = initial_alpha\n"
            "     - If learnable_alpha: create log_alpha parameter\n"
            "     - Create alpha_optimizer\n"
            "  7. Set target entropy\n"
            "  8. Initialize counters (episode, total_steps)"
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action from stochastic policy.

        Args:
            state: Current state observation
            deterministic: If True, use mean action (for evaluation)

        Returns:
            np.ndarray: Action sampled from π(·|s)

        Algorithm:
            1. Compute policy parameters: μ, log_std = actor(state)
            2. If deterministic: return tanh(μ)
            3. If stochastic:
               a. Sample ε ~ N(0, I)
               b. Compute raw action: a_raw = μ + exp(log_std) * ε
               c. Apply tanh: a = tanh(a_raw)
               d. Return squashed action in [-1, 1]

        Theory:
            Reparameterization trick enables gradient flow through sampling:
                a = tanh(μ(s) + σ(s) ⊙ ε)
            This is differentiable with respect to θ (policy parameters).
        """
        raise NotImplementedError(
            "Select action from stochastic policy:\n"
            "  1. Convert state to tensor, add batch dimension if needed\n"
            "  2. Call actor: mean, log_std = actor.forward(state)\n"
            "  3. If deterministic:\n"
            "    a. Return tanh(mean)\n"
            "  4. If stochastic:\n"
            "    a. Sample epsilon ~ N(0, I)\n"
            "    b. Compute raw action: a_raw = mean + exp(log_std)*epsilon\n"
            "    c. Apply tanh: action = tanh(a_raw)\n"
            "    d. Return as numpy array in [-1, 1]"
        )

    def update(self, batch_size: int = None) -> Dict[str, float]:
        """
        Perform one step of SAC training.

        Args:
            batch_size: Minibatch size (uses config.batch_size if None)

        Returns:
            Dict[str, float]: Training metrics
                - "critic_1_loss": MSE loss for Q1
                - "critic_2_loss": MSE loss for Q2
                - "actor_loss": Policy gradient loss
                - "alpha_loss": Entropy coefficient loss (if learnable)
                - "alpha": Current entropy coefficient value
                - "policy_entropy": Mean policy entropy
                - "q1_mean": Mean Q1-value estimate
                - "q2_mean": Mean Q2-value estimate
                - "value_loss": Value network loss (if used)

        Algorithm:

        **1. Sample minibatch and compute targets**:
            - states, actions, rewards, next_states, dones from buffer
            - Sample next actions from policy: a' ~ π(·|s')
            - Compute log probabilities: log π(a'|s')
            - Compute Q-targets:
                y = r + γ(1-d) * (min(Q'_1(s',a'), Q'_2(s',a')) - α log π(a'|s'))

        **2. Update both Q-networks**:
            - Compute Q predictions: Q_i(s,a)
            - Loss_i = ||y - Q_i(s,a)||²
            - Update: φ_i ← φ_i - β∇_φ_i Loss_i

        **3. Update actor (policy network)**:
            - Sample new actions: a ~ π(·|s) using reparameterization
            - Compute log probabilities: log π(a|s)
            - Compute policy loss: J = E[α log π(a|s) - Q(s,a)]
            - Update: θ ← θ - β∇_θ J

        **4. Update entropy coefficient (if learnable)**:
            - Compute alpha loss: J_α = -log α * (log π(a|s) + H_target)
            - Update: α ← α - β_α∇_α J_α
            - Ensure α stays in [alpha_min, alpha_max]

        **5. Update target networks** (Polyak averaging):
            - φ'_i ← τφ_i + (1-τ)φ'_i
            - θ' ← τθ + (1-τ)θ' (not typically done for actor in SAC)

        **6. Optional: Update value network** (if used):
            - Compute value targets: V_target = E[Q - α log π]
            - Update: ψ ← ψ - β∇_ψ ||V_target - V||²

        Mathematical Details:

        **Policy Loss Derivation**:
            J(θ) = E[α log π(a|s) - Q(s,a)]
            The actor learns to:
            - Maximize Q(s,a) (perform well)
            - Maximize log π(a|s) (entropy, scaled by α)

            The critic gradient is:
            ∇_θ J = -β E[∇_a Q(s,a)|_{a~π} ∇_θ log π(a|s)]
            This is computed via reparameterization trick.

        **Entropy Coefficient Loss**:
            J(α) = E[-α log π(a|s) - α H_target]
            Minimizing this adjusts α to maintain target entropy:
            - If current entropy < target: α decreases → less entropy penalty → more exploration
            - If current entropy > target: α increases → more entropy penalty → less exploration

        **Critical Implementation Notes**:
            1. Sample a ~ π(·|s) from CURRENT policy, not target policy
            2. Use first Q-network for actor gradient (reduce variance)
            3. Clamp log_std to prevent numerical issues
            4. Use tanh Jacobian correction for log probabilities
            5. Monitor α to verify it's tuning properly
        """
        raise NotImplementedError(
            "Perform SAC update step:\n"
            "  ========== EVERY STEP ==========\n"
            "  1. Check if replay buffer ready\n"
            "  2. Sample minibatch\n"
            "  3. Sample next actions from policy:\n"
            "    a. a', log_pi_a' = actor.sample_action(next_states, deterministic=False)\n"
            "  4. Compute Q-targets using minimum and entropy term:\n"
            "    a. q1_target = target_critic_1(next_states, a')\n"
            "    b. q2_target = target_critic_2(next_states, a')\n"
            "    c. q_target = min(q1_target, q2_target)\n"
            "    d. y = rewards + gamma*(1-dones)*(q_target - alpha*log_pi_a')\n"
            "  5. Update critic 1:\n"
            "    a. q1_pred = critic_1(states, actions)\n"
            "    b. loss1 = MSE(q1_pred, y)\n"
            "    c. critic_1.backward(loss1)\n"
            "  6. Update critic 2 (same as critic 1)\n"
            "  7. Update actor:\n"
            "    a. a, log_pi = actor.sample_action(states, deterministic=False)\n"
            "    b. q = critic_1(states, a)  [use first critic]\n"
            "    c. loss = mean(alpha*log_pi - q)\n"
            "    d. actor.backward(loss)\n"
            "  8. Update alpha (entropy coefficient):\n"
            "    a. loss_alpha = -log_alpha * (log_pi + target_entropy)\n"
            "    b. alpha_optimizer.backward(loss_alpha)\n"
            "    c. alpha = exp(log_alpha)\n"
            "    d. Clamp alpha to [alpha_min, alpha_max]\n"
            "  9. Update target networks (Polyak):\n"
            "    a. For critic_1 and critic_2:\n"
            "       target_param = tau*param + (1-tau)*target_param\n"
            "  10. Return metrics dict"
        )

    def train_episode(
        self,
        env: Any,
        max_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Execute one training episode with environment interaction.

        Args:
            env: Environment instance with Gym API
            max_steps: Maximum steps per episode

        Returns:
            Dict[str, float]: Episode metrics
                - "episode_reward": Total undiscounted reward
                - "episode_length": Number of steps
                - "avg_critic_loss": Average Q-network loss
                - "avg_actor_loss": Average policy loss
                - "avg_alpha": Average entropy coefficient
                - "avg_entropy": Average policy entropy

        Algorithm:
            1. Reset environment
            2. Loop until episode ends:
                a. Select action from stochastic policy
                b. Step environment
                c. Store in replay buffer
                d. Perform SAC update if buffer ready
                e. Accumulate reward
            3. Log statistics
            4. Return episode metrics
        """
        raise NotImplementedError(
            "Execute one training episode:\n"
            "  1. Reset environment: state, info = env.reset()\n"
            "  2. Loop for max_steps or until done:\n"
            "    a. Select action: action = self.select_action(state, deterministic=False)\n"
            "    b. Step environment: next_state, reward, terminated, truncated, info = env.step(action)\n"
            "    c. Store experience: self.replay_buffer.add(...)\n"
            "    d. If buffer ready:\n"
            "       - Call self.update() and track metrics\n"
            "    e. Accumulate reward\n"
            "    f. Update state\n"
            "    g. Break if done\n"
            "  3. Increment episode counter\n"
            "  4. Return metrics dict with avg losses and entropy"
        )

    def eval_episode(
        self,
        env: Any,
        max_steps: int = 1000,
        render: bool = False,
        deterministic: bool = True
    ) -> float:
        """
        Execute one evaluation episode without learning.

        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            render: Whether to render environment
            deterministic: If True, use mean action; if False, sample from policy

        Returns:
            float: Total episode reward
        """
        raise NotImplementedError(
            "Execute evaluation episode:\n"
            "  1. Set actor to eval mode (no gradients)\n"
            "  2. Reset environment\n"
            "  3. Loop:\n"
            "    a. Select action with deterministic=True (use mean)\n"
            "    b. Step environment\n"
            "    c. Accumulate reward\n"
            "    d. Break if done\n"
            "  4. Return total reward"
        )

    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint."""
        raise NotImplementedError(
            "Save checkpoint:\n"
            "  - Save actor and both critic parameters\n"
            "  - Save target networks\n"
            "  - Save alpha (entropy coefficient)\n"
            "  - Save optimizer states\n"
            "  - Save value network parameters (if used)\n"
            "  - Save config"
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint."""
        raise NotImplementedError(
            "Load checkpoint:\n"
            "  - Load all network parameters\n"
            "  - Load alpha value\n"
            "  - Load optimizer states\n"
            "  - Verify config consistency"
        )

    def get_info(self) -> Dict[str, Any]:
        """Get agent information and statistics."""
        raise NotImplementedError(
            "Return agent info dict with:\n"
            "  - episode_count\n"
            "  - total_steps\n"
            "  - current alpha value\n"
            "  - target entropy\n"
            "  - network architectures\n"
            "  - buffer size"
        )
