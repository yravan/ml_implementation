"""
Intrinsic Motivation for Exploration.

Implementation Status: STUB
Complexity: ★★★☆☆ (Intermediate)
Prerequisites: rl/core/networks, foundations/autograd

Intrinsic motivation methods provide exploration bonuses based on novelty,
curiosity, or prediction error, enabling exploration in sparse reward settings.

References:
    - Pathak et al. (2017): Curiosity-driven Exploration (ICM)
      https://arxiv.org/abs/1705.05363
    - Burda et al. (2019): Exploration by Random Network Distillation (RND)
      https://arxiv.org/abs/1810.12894
    - Bellemare et al. (2016): Count-Based Exploration
      https://arxiv.org/abs/1606.01868
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any


# =============================================================================
# THEORY: INTRINSIC MOTIVATION AND CURIOSITY
# =============================================================================
"""
THE EXPLORATION PROBLEM:
======================

In sparse reward environments, random exploration rarely discovers rewards.
Intrinsic motivation provides internal reward signals that encourage
exploring novel or surprising states.

Types of intrinsic motivation:
1. Prediction error: Reward for states where predictions fail
2. Novelty: Reward for states visited infrequently
3. Information gain: Reward for learning about the environment

PREDICTION ERROR METHODS:
========================

ICM (Intrinsic Curiosity Module):
    r_i = ||φ(s') - f(φ(s), a)||²
    where f predicts next state features

RND (Random Network Distillation):
    r_i = ||f_θ(s) - f_target(s)||²
    where f_target is a random fixed network

COUNT-BASED METHODS:
===================

Bonus inversely proportional to visit count:
    r_i = β / √N(s)

In large/continuous spaces, use pseudo-counts:
    N(s) ≈ density estimation (e.g., PixelCNN)

INFORMATION-THEORETIC METHODS:
=============================

Empowerment:
    r_i = I(a; s' | s)  (mutual information)

VIME (Variational Information Maximizing Exploration):
    r_i = KL(p(θ|D∪(s,a,s')) || p(θ|D))  (info gain about model)
"""


class IntrinsicReward:
    """
    Base class for intrinsic reward modules.

    Provides interface for computing exploration bonuses
    to augment environment rewards.

    Args:
        bonus_scale: Scaling factor for intrinsic rewards
        normalize: Whether to normalize intrinsic rewards
    """

    def __init__(
        self,
        bonus_scale: float = 0.01,
        normalize: bool = True
    ):
        """Initialize intrinsic reward module."""
        self.bonus_scale = bonus_scale
        self.normalize = normalize
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def compute_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ) -> np.ndarray:
        """
        Compute intrinsic reward bonus.

        Args:
            states: Current states [batch, state_dim]
            actions: Actions taken [batch, action_dim]
            next_states: Resulting states [batch, state_dim]

        Returns:
            Intrinsic rewards [batch]
        """
        raise NotImplementedError("Subclasses must implement compute_reward")

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ) -> Dict[str, float]:
        """
        Update intrinsic motivation module.

        Returns:
            Training metrics
        """
        raise NotImplementedError("Subclasses must implement update")

    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics."""
        raise NotImplementedError(
            "Running normalization:\n"
            "- Update mean/std with new rewards\n"
            "- Return (rewards - mean) / (std + eps)"
        )


class ICM(IntrinsicReward):
    """
    Intrinsic Curiosity Module.

    ICM uses prediction error in a learned feature space as curiosity.
    A forward model predicts next state features, and error provides reward.

    Theory:
        ICM consists of three components:
        1. Feature encoder φ(s) that maps states to features
        2. Inverse model p(a|φ(s), φ(s')) that predicts actions
        3. Forward model f(φ(s), a) that predicts φ(s')

        The inverse model ensures features capture action-relevant info.
        The forward model prediction error provides curiosity reward.

    Mathematical Formulation:
        Feature encoder: φ(s) ∈ R^d

        Inverse loss (trains encoder to capture action-relevant features):
            L_I = CrossEntropy(p_inverse(φ(s), φ(s')), a)

        Forward loss:
            L_F = ||f(φ(s), a) - φ(s')||²

        Intrinsic reward:
            r_i = η * ||f(φ(s), a) - φ(s')||²

    References:
        - Pathak et al. (2017): Curiosity-driven Exploration
          https://arxiv.org/abs/1705.05363

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        feature_dim: Feature embedding dimension
        inverse_weight: Weight for inverse model loss
        forward_weight: Weight for forward model loss
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        inverse_weight: float = 0.2,
        forward_weight: float = 0.8,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        """Initialize ICM."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.inverse_weight = inverse_weight
        self.forward_weight = forward_weight

        # Networks
        self.encoder = None  # φ(s)
        self.inverse_model = None  # p(a|φ(s), φ(s'))
        self.forward_model = None  # f(φ(s), a)
        self.optimizer = None

        self._build_networks(learning_rate)

    def _build_networks(self, learning_rate: float) -> None:
        """
        Build ICM networks.

        Implementation Hints:
            Encoder: MLP [state_dim] -> [feature_dim]
            Inverse: MLP [2*feature_dim] -> [action_dim] (classification/regression)
            Forward: MLP [feature_dim + action_dim] -> [feature_dim]
        """
        raise NotImplementedError(
            "Build ICM networks:\n"
            "- Encoder: state_dim -> feature_dim\n"
            "- Inverse: 2*feature_dim -> action_dim\n"
            "- Forward: feature_dim + action_dim -> feature_dim"
        )

    def encode(self, states: np.ndarray) -> np.ndarray:
        """Encode states to features."""
        raise NotImplementedError("Forward pass through encoder")

    def compute_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ) -> np.ndarray:
        """
        Compute curiosity reward from forward prediction error.

        Returns:
            Curiosity rewards [batch]
        """
        raise NotImplementedError(
            "Compute ICM reward:\n"
            "- phi = encode(states)\n"
            "- phi_next = encode(next_states)\n"
            "- phi_next_pred = forward_model(phi, actions)\n"
            "- reward = ||phi_next_pred - phi_next||²\n"
            "- Return bonus_scale * reward"
        )

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ) -> Dict[str, float]:
        """
        Update ICM networks.

        Returns:
            inverse_loss, forward_loss, total_loss
        """
        raise NotImplementedError(
            "Update ICM:\n"
            "- phi = encode(states)\n"
            "- phi_next = encode(next_states).detach()  # stop gradient\n"
            "- Inverse loss: predict action from (phi, phi_next)\n"
            "- Forward loss: predict phi_next from (phi, action)\n"
            "- Total = inverse_weight * L_I + forward_weight * L_F\n"
            "- Backprop and update"
        )


class RND(IntrinsicReward):
    """
    Random Network Distillation.

    RND uses a randomly initialized target network as a measure of novelty.
    A predictor network is trained to match the target's output, and the
    prediction error provides exploration bonus.

    Theory:
        RND consists of two networks:
        1. Target network f_target: randomly initialized, fixed
        2. Predictor network f_θ: trained to match target

        For frequently visited states, the predictor learns to match
        the target output. For novel states, prediction error is high.

        Key insight: prediction error decreases with experience, so
        novel states have high error and get higher exploration bonus.

    Mathematical Formulation:
        Intrinsic reward:
            r_i = ||f_θ(s) - f_target(s)||²

        Training objective:
            L = E[||f_θ(s) - f_target(s)||²]

    References:
        - Burda et al. (2019): Exploration by Random Network Distillation
          https://arxiv.org/abs/1810.12894

    Args:
        state_dim: State dimension
        feature_dim: Output dimension of networks
        use_observation_norm: Whether to normalize observations
    """

    def __init__(
        self,
        state_dim: int,
        feature_dim: int = 512,
        hidden_dims: List[int] = [512, 512],
        use_observation_norm: bool = True,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        """Initialize RND."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.use_observation_norm = use_observation_norm

        # Target network (random, fixed)
        self.target = None
        # Predictor network (trained)
        self.predictor = None
        self.optimizer = None

        # Observation normalization
        self.obs_mean = np.zeros(state_dim)
        self.obs_std = np.ones(state_dim)
        self.obs_count = 0

        self._build_networks(hidden_dims, learning_rate)

    def _build_networks(
        self,
        hidden_dims: List[int],
        learning_rate: float
    ) -> None:
        """
        Build RND networks.

        Implementation Hints:
            Both networks: MLP [state_dim] -> [feature_dim]
            Target: randomly initialized, NO gradient
            Predictor: trained with gradient
        """
        raise NotImplementedError(
            "Build RND networks:\n"
            "- Target: MLP state_dim -> feature_dim (fixed, no grad)\n"
            "- Predictor: MLP state_dim -> feature_dim (trainable)\n"
            "- Initialize target with random weights"
        )

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics."""
        raise NotImplementedError(
            "Normalize observations:\n"
            "- Update running mean/std\n"
            "- Return (obs - mean) / (std + eps)\n"
            "- Clip to reasonable range"
        )

    def compute_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray = None,
        next_states: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute RND exploration bonus.

        Note: RND only uses states, not actions or next_states.

        Returns:
            Exploration bonuses [batch]
        """
        raise NotImplementedError(
            "Compute RND reward:\n"
            "- Normalize states\n"
            "- target_features = target(states)  # no grad\n"
            "- pred_features = predictor(states)\n"
            "- reward = ||pred_features - target_features||²\n"
            "- Return bonus_scale * normalized reward"
        )

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray = None,
        next_states: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Update predictor network.

        Returns:
            rnd_loss
        """
        raise NotImplementedError(
            "Update RND predictor:\n"
            "- Normalize states\n"
            "- target_features = target(states).detach()\n"
            "- pred_features = predictor(states)\n"
            "- loss = mean(||pred - target||²)\n"
            "- Backprop predictor only\n"
            "- Return loss"
        )


class CountBasedBonus(IntrinsicReward):
    """
    Count-based exploration bonus.

    Provides bonus inversely proportional to state visitation count.
    Uses hash-based discretization for continuous state spaces.

    Theory:
        Classic exploration bonus for tabular settings:
            r_i = β / √N(s)

        For continuous states, we discretize using:
        1. Hash functions (SimHash)
        2. Density models (pseudo-counts)
        3. K-nearest neighbors

    Mathematical Formulation:
        Basic count bonus:
            r_i = β / √(N(s) + 1)

        With pseudo-count from density model:
            N̂(s) = ρ(s) * (1 - ρ'(s)) / (ρ'(s) - ρ(s))
            where ρ'(s) is density after observing s

    References:
        - Bellemare et al. (2016): Unifying Count-Based Exploration
          https://arxiv.org/abs/1606.01868

    Args:
        state_dim: State dimension
        n_bins: Number of bins per dimension for discretization
        method: 'hash', 'density', or 'knn'
    """

    def __init__(
        self,
        state_dim: int,
        n_bins: int = 32,
        method: str = 'hash',
        **kwargs
    ):
        """Initialize count-based exploration."""
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.n_bins = n_bins
        self.method = method

        # State visit counts
        self.counts = {}

        # State bounds for discretization
        self.state_min = None
        self.state_max = None

    def _discretize(self, states: np.ndarray) -> np.ndarray:
        """
        Discretize continuous states.

        Implementation Hints:
            1. Clip to [state_min, state_max]
            2. Normalize to [0, 1]
            3. Multiply by n_bins and floor
            4. Return hash of bin indices
        """
        raise NotImplementedError(
            "Discretize states:\n"
            "- Normalize: (states - min) / (max - min)\n"
            "- bins = floor(normalized * n_bins)\n"
            "- Return tuple(bins) as hash key"
        )

    def compute_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray = None,
        next_states: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute count-based exploration bonus.

        Returns:
            Count bonuses [batch]
        """
        raise NotImplementedError(
            "Count bonus:\n"
            "- Discretize states to get keys\n"
            "- For each key: reward = bonus_scale / sqrt(counts[key] + 1)\n"
            "- Return rewards array"
        )

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray = None,
        next_states: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Update state visit counts.

        Returns:
            Average count, unique states
        """
        raise NotImplementedError(
            "Update counts:\n"
            "- Discretize states\n"
            "- For each key: counts[key] += 1\n"
            "- Return average count, unique states"
        )


class NoisyNetworks(IntrinsicReward):
    """
    Noisy Networks for Exploration.

    Instead of epsilon-greedy, add learnable noise to network parameters.
    The network learns when and how to explore.

    Theory:
        Replace linear layers with noisy linear layers:
            y = (W + σ_w ⊙ ε_w)x + (b + σ_b ⊙ ε_b)

        where ε are noise samples and σ are learned noise scales.
        The network can learn to reduce noise when exploration is not needed.

    References:
        - Fortunato et al. (2018): Noisy Networks for Exploration
          https://arxiv.org/abs/1706.10295
    """

    def __init__(
        self,
        sigma_init: float = 0.5,
        factorized: bool = True
    ):
        """Initialize noisy networks."""
        self.sigma_init = sigma_init
        self.factorized = factorized

    def noisy_linear(
        self,
        in_features: int,
        out_features: int
    ) -> Dict[str, np.ndarray]:
        """
        Create a noisy linear layer.

        Returns:
            Dictionary with weights, biases, and noise parameters
        """
        raise NotImplementedError(
            "Noisy linear layer:\n"
            "- mu_w: learnable weight mean\n"
            "- sigma_w: learnable weight noise scale\n"
            "- mu_b, sigma_b: for bias\n"
            "- Initialize mu ~ Uniform, sigma = sigma_init"
        )

    def sample_noise(
        self,
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Sample noise for noisy layer.

        For factorized noise: ε = sign(x) * sqrt(|x|)
        where x ~ N(0,1)
        """
        raise NotImplementedError(
            "Sample noise:\n"
            "- If factorized: use factorized gaussian\n"
            "- Else: sample N(0,1) of full shape"
        )

    def reset_noise(self) -> None:
        """Resample noise for all noisy layers."""
        raise NotImplementedError("Resample all noise tensors")
