"""
Module Functions
==============

Objective functions for training neural networks.

Module functions measure the discrepancy between model predictions and targets.
The choice of loss function depends on the task:

1. **Regression Modulees** (continuous targets):
   - MSE: Mean Squared Error (L2 loss)
   - MAE: Mean Absolute Error (L1 loss)
   - Huber: Smooth combination of L1 and L2
   - SmoothL1: Similar to Huber, used in object detection

2. **Classification Modulees** (discrete targets):
   - CrossEntropy: Standard multi-class classification
   - BCE: Binary Cross-Entropy for binary/multi-label
   - NLL: Negative Log-Likelihood (with log_softmax)
   - Focal: Handles class imbalance by down-weighting easy examples

3. **Sequence Modulees**:
   - CTC: Connectionist Temporal Classification (ASR, OCR)
   - LabelSmoothing: Regularized cross-entropy for sequences

4. **Metric Learning Modulees**:
   - Triplet: Learn embeddings where similar items are close
   - Contrastive: Push apart dissimilar pairs

5. **Distribution Modulees**:
   - KLDiv: Kullback-Leibler divergence between distributions

Theory
------
A loss function L(ŷ, y) measures prediction error. Training minimizes:

    θ* = argmin_θ E_{(x,y)~D}[L(f_θ(x), y)]

Key properties to consider:
- **Convexity**: Convex losses (MSE, CE) have unique minima
- **Robustness**: Some losses are robust to outliers (Huber, MAE)
- **Gradient behavior**: Affects optimization dynamics (CE has nice gradients)
- **Calibration**: Some losses produce calibrated probabilities (CE)

Relationship to Maximum Likelihood:
- MSE ↔ Gaussian likelihood: p(y|x) = N(f(x), σ²)
- BCE ↔ Bernoulli likelihood: p(y|x) = Ber(σ(f(x)))
- CE ↔ Categorical likelihood: p(y|x) = Cat(softmax(f(x)))

References
----------
- "Understanding Deep Learning" Ch. 5: Module Functions
  https://udlbook.github.io/udlbook/
- "Pattern Recognition and Machine Learning" Bishop, Ch. 4
- "Focal Module for Dense Object Detection" Lin et al. (2017)
  https://arxiv.org/abs/1708.02002

Implementation Notes
--------------------
- All losses support 'mean', 'sum', 'none' reduction modes
- forward() computes loss, backward() computes gradient w.r.t. predictions
- For numerical stability, combine softmax+CE, sigmoid+BCE
- Store intermediate values in forward() for backward()
"""
from math import prod

# Implementation Status: NOT STARTED
# Complexity: Easy to Medium
# Prerequisites: None (some need utils/math_utils for logsumexp)

import numpy as np
from typing import Optional, Literal, Union, Tuple

from python.foundations import Function, Tensor, minimum
from python.nn_core import Module


# =============================================================================
# Base Module Class
# =============================================================================
def _reduce(loss: Tensor, reduction: str) -> Tensor:
    """Apply reduction to loss."""
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:  # mean
        return loss.mean()


# =============================================================================
# Regression Modulees
# =============================================================================

class MSEModule(Module):
    """
    Mean Squared Error Module (L2 Module).

    L = (1/n) * Σ(y - ŷ)²

    Properties:
    - Differentiable everywhere
    - Penalizes large errors heavily (squared)
    - Corresponds to Gaussian MLE

    Math:
        L = mean((ŷ - y)²)
        ∂L/∂ŷ = (2/n) * (ŷ - y)

    Example:
        >>> loss_fn = MSEModule()
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> loss = loss_fn(pred, target)
        0.25  # mean([0.25, 0.25, 0.25])
    """

    def forward(self, predictions: Tensor, targets: Tensor, reduction: str="mean") -> Tensor:
        if predictions.shape != targets.shape:
            raise RuntimeError("Predictions and targets must have the same shape", predictions.shape, targets.shape)
        loss = (predictions - targets)**2
        return _reduce(loss, reduction)


class MAEModule(Module):
    """
    Mean Absolute Error Module (L1 Module).

    L = (1/n) * Σ|y - ŷ|

    Properties:
    - More robust to outliers than MSE
    - Non-differentiable at 0 (use subgradient)
    - Corresponds to Laplace distribution MLE

    Math:
        L = mean(|ŷ - y|)
        ∂L/∂ŷ = (1/n) * sign(ŷ - y)

    Example:
        >>> loss_fn = MAEModule()
        >>> loss = loss_fn(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        0.5  # mean([0.5, 0.5])
    """

    def forward(self, predictions: Tensor, targets: Tensor, reduction: str="mean") -> Tensor:
        if predictions.shape != targets.shape:
            raise RuntimeError("Predictions and targets must have the same shape", predictions.shape, targets.shape)
        self.reduction = reduction
        loss = (predictions - targets).abs()
        return _reduce(loss, reduction)



class HuberModule(Module):
    """
    Huber Module (Smooth L1).

    Combines MSE (for small errors) and MAE (for large errors).
    Less sensitive to outliers than MSE, but smoother than MAE.

    Math:
        L = | 0.5 * (y - ŷ)²     if |y - ŷ| < δ
            | δ * (|y - ŷ| - 0.5δ)  otherwise

        ∂L/∂ŷ = | (ŷ - y)           if |y - ŷ| < δ
                | δ * sign(ŷ - y)   otherwise

    The transition at δ ensures the function and its derivative are continuous.

    References:
        - Huber "Robust Estimation of a Location Parameter" (1964)

    Example:
        >>> loss = HuberModule(delta=1.0)
        >>> loss(np.array([0.0, 10.0]), np.array([0.5, 0.5]))
        # Small error uses MSE, large error uses MAE
    """

    def forward(self, predictions: Tensor, targets: Tensor, delta: float = 1.0, reduction: str="mean") -> Tensor:
        mse = (predictions - targets)**2 * 0.5
        mae = delta * ((predictions - targets).abs() - 0.5 * delta)
        loss = minimum(mse, mae)
        return _reduce(loss, reduction)



class SmoothL1Module(Module):
    """
    Smooth L1 Module.

    Similar to Huber but with β parameter. Used extensively in object detection
    (Faster R-CNN, SSD) for bounding box regression.

    Math:
        L = | 0.5 * (y - ŷ)² / β   if |y - ŷ| < β
            | |y - ŷ| - 0.5 * β    otherwise

    When β = 1.0, equivalent to Huber with δ = 1.0.

    References:
        - Girshick "Fast R-CNN" (2015)
          https://arxiv.org/abs/1504.08083
    """

    def forward(self, predictions: Tensor, targets: Tensor, beta: float = 1.0, reduction: str='mean') -> Tensor:
        mse = (predictions - targets)**2 * 0.5 / beta
        mae = (predictions - targets).abs() - 0.5 * beta
        loss = minimum(mse, mae)
        return _reduce(loss, reduction)


class RMSEModule(Module):
    """
    Root Mean Squared Error Module.

    L = √(mean((y - ŷ)²))

    Same scale as target variable, more interpretable than MSE.
    Gradient requires MSE, then chain rule through sqrt.
    """

    def forward(self, predictions: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
        ms = ((predictions - targets)**2)
        loss = _reduce(ms, reduction) ** 0.5
        return loss


# =============================================================================
# Classification Modulees
# =============================================================================

class CrossEntropyModule(Module):
    """
    Cross-Entropy Module for multi-class classification.

    Combines softmax and negative log-likelihood for numerical stability.

    L = -log(softmax(x)[target]) = -x[target] + log(Σexp(x))

    The gradient has a beautiful form:
        ∂L/∂x_c = softmax(x)_c - 1{c = target}
               = p_c - y_c  (predicted prob minus one-hot)

    Math:
        # Stable computation:
        log_softmax = x - logsumexp(x)
        L = -log_softmax[target]

        # Gradient:
        ∂L/∂x = softmax(x) - one_hot(target)

    References:
        - "Understanding Deep Learning" Ch. 5.2
        - CS231n Softmax classifier

    Example:
        >>> loss_fn = CrossEntropyModule()
        >>> logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        >>> targets = np.array([2, 0])  # Class indices
        >>> loss = loss_fn(logits, targets)
    """
    def forward(self, logits: Tensor, targets: Tensor, axis: int = -1, ignore_index: int = -100, label_smoothing: float = 0.0, weight: Optional[np.ndarray] = None, reduction: str = 'mean') -> Tensor:
        B = logits.shape[0]
        K = logits.shape[axis]

        # log softmax
        log_probs = logits.log_softmax(axis=axis)

        # one-hot labels
        labels_data = np.zeros(logits.shape)
        labels_data[np.arange(B), targets.data] = 1.0
        labels = Tensor(labels_data)

        # label smoothing
        if label_smoothing > 0:
            labels = labels * (1 - label_smoothing) + label_smoothing / K

        # per-sample loss (sum over classes)
        loss = -(labels * log_probs).sum(axis=axis)  # (B,)

        # per-class weights
        if weight is not None:
            loss = loss * weight[targets.data]

        # mask ignored samples
        valid = targets.data != ignore_index
        loss = loss * Tensor(valid.astype(float))
        loss = _reduce(loss, reduction)
        return loss

class BinaryCrossEntropyModule(Module):
    """
    Binary Cross-Entropy Module.

    For binary classification or multi-label classification.

    L = -[y * log(p) + (1-y) * log(1-p)]

    Math:
        L = -(y log(p) + (1-y) log(1-p))
        ∂L/∂p = (p - y) / (p(1-p))

    For numerical stability, use BCEWithLogitsModule instead.

    Example:
        >>> loss_fn = BinaryCrossEntropyModule()
        >>> pred = np.array([0.9, 0.1, 0.8])  # Probabilities
        >>> target = np.array([1, 0, 1])
        >>> loss = loss_fn(pred, target)
    """

    def forward(self, probs: Tensor, targets: Tensor, pos_weight: float = 1.0, reduction: str = 'mean') -> Tensor:
        loss = -(targets * probs.log() * pos_weight + (1 - targets) * (1 - probs).log())
        loss = _reduce(loss, reduction)
        return loss

class BCEWithLogitsModule(Module):
    """
    BCE with built-in sigmoid, numerically stable.

    Combines sigmoid + BCE in a numerically stable way:
    L = max(x, 0) - x*y + log(1 + exp(-|x|))

    This avoids computing sigmoid directly, preventing overflow/underflow.

    Math:
        # Stable formulation:
        L = max(x, 0) - xy + log(1 + exp(-|x|))

        # Gradient (w.r.t. logits):
        ∂L/∂x = sigmoid(x) - y

    References:
        - PyTorch BCEWithLogitsModule documentation
    """

    def forward(self, logits: Tensor, targets: Tensor, pos_weight: float = 1.0, reduction: str = 'mean') -> Tensor:
        loss = -(targets * logits.log_sigmoid() * pos_weight + (1 - targets) * (-logits).log_sigmoid())
        loss = _reduce(loss, reduction)
        return loss


class NLLModule(Module):
    """
    Negative Log-Likelihood Module.

    Used with log_softmax output. Equivalent to CrossEntropyModule when
    combined with log_softmax.

    L = -log_probs[target]

    Example:
        >>> # Typically used with log_softmax:
        >>> log_probs = log_softmax(logits)
        >>> loss = NLLModule()(log_probs, targets)
    """

    def forward(self, log_probs: Tensor, targets: Tensor, axis: int = -1, ignore_index: int = -100, label_smoothing: float = 0.0, weight: Optional[np.ndarray] = None, reduction: str = 'mean') -> Tensor:
        B = log_probs.shape[0]
        # one-hot labels
        labels_data = np.zeros(log_probs.shape)
        labels_data[np.arange(B), targets.data] = 1.0
        labels = Tensor(labels_data)

        # label smoothing
        if label_smoothing > 0:
            labels = labels * (1 - label_smoothing) + label_smoothing / K

        # per-sample loss (sum over classes)
        loss = -(labels * log_probs).sum(axis=axis)  # (B,)

        # per-class weights
        if weight is not None:
            loss = loss * weight[targets.data]

        # mask ignored samples
        valid = targets.data != ignore_index
        loss = loss * Tensor(valid.astype(float))
        loss = _reduce(loss, reduction)
        return loss

class FocalModule(Module):
    """
    Focal Module for handling class imbalance.

    Down-weights easy examples (high confidence correct predictions),
    focusing training on hard examples.

    Math:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where p_t = p if y=1 else 1-p (probability of correct class).

    γ (gamma) controls how much to down-weight easy examples:
    - γ = 0: equivalent to cross-entropy
    - γ > 0: reduces loss for well-classified examples

    Originally developed for object detection where background
    overwhelmingly outnumbers objects.

    References:
        - Lin et al. "Focal Module for Dense Object Detection" (2017)
          https://arxiv.org/abs/1708.02002

    Example:
        >>> loss = FocalModule(gamma=2.0, alpha=0.25)
        >>> loss(logits, targets)  # Easy examples contribute less
    """

    def forward(self, logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = 'mean') -> Tensor:
        B = logits.shape[0]
        K = logits.shape[-1]

        # Compute softmax probabilities
        probs = logits.softmax(axis=-1)

        # Get probability of correct class: p_t
        # Create one-hot and extract p_t via element-wise multiply + sum
        labels_data = np.zeros(logits.shape)
        labels_data[np.arange(B), targets.data] = 1.0
        labels = Tensor(labels_data)

        p_t = (probs * labels).sum(axis=-1)  # (B,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** gamma

        # Cross-entropy for correct class: -log(p_t)
        ce = -p_t.log()

        # Focal loss
        loss = focal_weight * ce

        # Optional alpha weighting (for binary, alpha weights positive class)
        if alpha is not None:
            alpha_t = Tensor(np.where(targets.data == 1, alpha, 1 - alpha))
            loss = alpha_t * loss

        return _reduce(loss, reduction)


# =============================================================================
# Sequence Modulees
# =============================================================================

class CTCModule(Module):
    """
    Connectionist Temporal Classification Module.

    For sequence-to-sequence problems where alignment is unknown,
    e.g., speech recognition, OCR.

    CTC sums over all possible alignments between input and output sequences,
    allowing the model to predict without knowing the exact timing.

    Key concepts:
    - Blank token: Allows model to output "nothing" at a time step
    - Alignment: Many-to-one mapping from input frames to output tokens
    - Forward-backward algorithm: Efficiently sums over all alignments

    Math:
        L = -log P(y|x) = -log Σ_{alignments A} P(A|x)

    The sum over alignments is computed efficiently using dynamic programming.

    References:
        - Graves et al. "Connectionist Temporal Classification" (2006)
          https://www.cs.toronto.edu/~graves/icml_2006.pdf

    Note:
        CTC is complex to implement with autograd. This implementation provides
        the forward pass structure; for full gradient support, consider using
        a specialized library or implementing the forward-backward algorithm.
    """

    def forward(self, log_probs: Tensor, targets: Tensor,
                input_lengths: Tensor, target_lengths: Tensor,
                blank: int = 0, reduction: str = 'mean') -> Tensor:
        """
        Args:
            log_probs: (T, N, C) log probabilities from log_softmax
            targets: (S,) or (N, S) flattened target sequences
            input_lengths: (N,) length of each input sequence
            target_lengths: (N,) length of each target sequence
            blank: Index of blank label (default: 0)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            CTC loss

        Implementation uses forward-backward algorithm:
        1. Expand targets with blanks: [b, t1, b, t2, b, ...]
        2. Forward pass: α[t,s] = P(output up to s at time t)
        3. Loss = -log(Σ_s α[T,s])
        """
        T, N, C = log_probs.shape

        # For simplicity, compute per-sample losses
        losses = []
        target_offset = 0

        for n in range(N):
            T_n = int(input_lengths.data[n])
            S_n = int(target_lengths.data[n])

            # Extract this sample's targets
            if targets.data.ndim == 1:
                sample_targets = targets.data[target_offset:target_offset + S_n]
                target_offset += S_n
            else:
                sample_targets = targets.data[n, :S_n]

            # Expand targets with blanks: [b, t1, b, t2, b, ...]
            L = 2 * S_n + 1
            expanded = np.zeros(L, dtype=int)
            expanded[0::2] = blank
            expanded[1::2] = sample_targets

            # Forward algorithm (in log space for stability)
            log_alpha = np.full((T_n, L), -np.inf)

            # Initialize
            log_alpha[0, 0] = log_probs.data[0, n, blank]
            if L > 1:
                log_alpha[0, 1] = log_probs.data[0, n, expanded[1]]

            # Forward pass
            for t in range(1, T_n):
                for s in range(L):
                    label = expanded[s]
                    log_prob = log_probs.data[t, n, label]

                    # Can come from same state
                    score = log_alpha[t-1, s]

                    # Can come from previous state
                    if s > 0:
                        score = np.logaddexp(score, log_alpha[t-1, s-1])

                    # Can skip blank (if not blank and not same as s-2)
                    if s > 1 and label != blank and label != expanded[s-2]:
                        score = np.logaddexp(score, log_alpha[t-1, s-2])

                    log_alpha[t, s] = score + log_prob

            # Total probability: sum of last two states
            log_prob_total = log_alpha[T_n-1, L-1]
            if L > 1:
                log_prob_total = np.logaddexp(log_prob_total, log_alpha[T_n-1, L-2])

            losses.append(-log_prob_total)

        loss = Tensor(np.array(losses))
        return _reduce(loss, reduction)


# =============================================================================
# Metric Learning Modulees
# =============================================================================

class TripletModule(Module):
    """
    Triplet Module for metric learning.

    Learns embeddings where similar items (anchor, positive) are closer
    than dissimilar items (anchor, negative).

    L = max(d(a,p) - d(a,n) + margin, 0)

    Where:
    - a: anchor embedding
    - p: positive (similar to anchor)
    - n: negative (dissimilar)
    - d: distance function (typically Euclidean)

    References:
        - Schroff et al. "FaceNet: A Unified Embedding for Face Recognition" (2015)
          https://arxiv.org/abs/1503.03832

    Example:
        >>> loss = TripletModule()
        >>> loss(anchor, positive, negative, margin=1.0)
    """

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor,
                margin: float = 1.0, p: int = 2, reduction: str = 'mean') -> Tensor:
        """
        Args:
            anchor: (N, D) anchor embeddings
            positive: (N, D) positive embeddings (similar to anchor)
            negative: (N, D) negative embeddings (dissimilar)
            margin: Margin between positive and negative distances
            p: p-norm for distance (2 = Euclidean)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Triplet loss
        """
        # Compute squared distances (more numerically stable)
        d_pos_sq = ((anchor - positive) ** 2).sum(axis=-1)
        d_neg_sq = ((anchor - negative) ** 2).sum(axis=-1)

        if p == 2:
            # Euclidean distance
            d_pos = d_pos_sq ** 0.5
            d_neg = d_neg_sq ** 0.5
        else:
            # For other p-norms, use abs and power
            d_pos = ((anchor - positive).abs() ** p).sum(axis=-1) ** (1.0 / p)
            d_neg = ((anchor - negative).abs() ** p).sum(axis=-1) ** (1.0 / p)

        # Triplet loss: max(d_pos - d_neg + margin, 0)
        loss = (d_pos - d_neg + margin).relu()

        return _reduce(loss, reduction)


class ContrastiveModule(Module):
    """
    Contrastive Module for learning similarity.

    For pairs of items, learns embeddings where similar pairs are close
    and dissimilar pairs are far apart.

    L = (1-y) * d² + y * max(margin - d, 0)²

    Where y=0 for similar pairs (pull together), y=1 for dissimilar (push apart).

    References:
        - Hadsell et al. "Dimensionality Reduction by Learning an Invariant Mapping" (2006)
          http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Example:
        >>> loss = ContrastiveModule()
        >>> loss(x1, x2, labels, margin=1.0)
    """

    def forward(self, x1: Tensor, x2: Tensor, y: Tensor,
                margin: float = 1.0, reduction: str = 'mean') -> Tensor:
        """
        Args:
            x1: (N, D) first embeddings
            x2: (N, D) second embeddings
            y: (N,) labels (0 = similar, 1 = dissimilar)
            margin: Margin for dissimilar pairs
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Contrastive loss
        """
        # Euclidean distance
        d_sq = ((x1 - x2) ** 2).sum(axis=-1)
        d = d_sq ** 0.5

        # Similar pairs: minimize distance
        loss_similar = (1 - y) * d_sq

        # Dissimilar pairs: push apart up to margin
        margin_diff = (Tensor(np.full(d.shape, margin)) - d).relu()
        loss_dissimilar = y * (margin_diff ** 2)

        loss = loss_similar + loss_dissimilar
        return _reduce(loss, reduction)


class InfoNCEModule(Module):
    """
    InfoNCE Module (Noise Contrastive Estimation).

    The loss function behind SimCLR, CLIP, and many contrastive learning methods.
    Maximizes mutual information between positive pairs.

    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

    Where sim is cosine similarity and τ is temperature.

    References:
        - Oord et al. "Representation Learning with Contrastive Predictive Coding" (2018)
          https://arxiv.org/abs/1807.03748
        - Chen et al. "A Simple Framework for Contrastive Learning" (SimCLR, 2020)
          https://arxiv.org/abs/2002.05709

    Example:
        >>> loss = InfoNCEModule()
        >>> loss(query, positive_key, temperature=0.07)
    """

    def forward(self, query: Tensor, key: Tensor,
                negatives: Optional[Tensor] = None,
                temperature: float = 0.07, reduction: str = 'mean') -> Tensor:
        """
        Args:
            query: (N, D) query embeddings
            key: (N, D) positive key embeddings
            negatives: (M, D) negative key embeddings (optional, uses other batch items if None)
            temperature: Temperature scaling factor
            reduction: 'mean', 'sum', or 'none'

        Returns:
            InfoNCE loss
        """
        # L2 normalize embeddings
        query_norm = (query ** 2).sum(axis=-1, keepdims=True) ** 0.5
        key_norm = (key ** 2).sum(axis=-1, keepdims=True) ** 0.5
        query = query / query_norm
        key = key / key_norm

        # Positive similarity: dot product of corresponding pairs
        pos_sim = (query * key).sum(axis=-1) / temperature  # (N,)

        if negatives is None:
            # Use other batch items as negatives (SimCLR style)
            # All pairwise similarities
            all_sim = query @ key.T() / temperature  # (N, N)

            # Numerator: exp(positive similarity)
            # Denominator: sum of exp(all similarities) for each query
            # But we need to be careful: the positive is on the diagonal

            # log_softmax over all keys for each query, then pick diagonal
            log_probs = all_sim.log_softmax(axis=-1)  # (N, N)

            # Extract diagonal (positive pairs)
            # Loss is negative log prob of positive
            diag_indices = np.arange(query.shape[0])
            loss = Tensor(-log_probs.data[diag_indices, diag_indices])
        else:
            # Explicit negatives provided
            neg_norm = (negatives ** 2).sum(axis=-1, keepdims=True) ** 0.5
            negatives = negatives / neg_norm

            # Negative similarities: (N, M)
            neg_sim = query @ negatives.T() / temperature

            # Concatenate positive and negative similarities
            # pos_sim: (N,) -> (N, 1)
            # neg_sim: (N, M)
            # all_sim: (N, 1+M)
            pos_sim_expanded = pos_sim.reshape((-1, 1))

            # For proper autograd, we need to concatenate tensors
            # Simplified: compute logsumexp manually
            all_sim_data = np.concatenate([pos_sim_expanded.data, neg_sim.data], axis=-1)
            all_sim = Tensor(all_sim_data)

            # InfoNCE: -log(exp(pos) / sum(exp(all))) = -pos + logsumexp(all)
            log_probs = all_sim.log_softmax(axis=-1)
            loss = -log_probs[:, 0]  # First column is positive

        return _reduce(loss, reduction)


# =============================================================================
# Distribution Modulees
# =============================================================================

class KLDivModule(Module):
    """
    Kullback-Leibler Divergence Module.

    Measures how one probability distribution diverges from another.

    KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
             = Σ P(x) * (log P(x) - log Q(x))

    Note: Input should be log-probabilities (from log_softmax),
    and target should be probabilities.

    Properties:
    - Not symmetric: KL(P||Q) ≠ KL(Q||P)
    - KL(P||Q) = 0 iff P = Q
    - Always non-negative

    Common uses:
    - Knowledge distillation (soft targets)
    - VAE latent space regularization
    - Policy gradient methods

    References:
        - "Information Theory, Inference and Learning Algorithms" MacKay

    Example:
        >>> loss = KLDivModule()
        >>> loss(log_probs, target_probs, reduction='batchmean')
    """

    def forward(self, log_probs: Tensor, targets: Tensor,
                log_target: bool = False, reduction: str = 'batchmean') -> Tensor:
        """
        Args:
            log_probs: (N, C) log-probabilities (from log_softmax)
            targets: (N, C) target probabilities (or log-probs if log_target=True)
            log_target: If True, targets are log-probabilities
            reduction: 'mean', 'sum', 'batchmean', or 'none'

        Returns:
            KL divergence loss
        """
        if log_target:
            # targets are log probabilities
            # KL = exp(log_p) * (log_p - log_q) = p * (log_p - log_q)
            target_probs = targets.exp()
            kl = target_probs * (targets - log_probs)
        else:
            # targets are probabilities
            # KL = p * (log(p) - log_q)
            # Need to handle p=0 case
            eps = 1e-8
            target_log = (targets + eps).log()
            kl = targets * (target_log - log_probs)

        # Sum over classes
        kl_per_sample = kl.sum(axis=-1)

        if reduction == 'batchmean':
            return kl_per_sample.sum() / log_probs.shape[0]
        else:
            return _reduce(kl_per_sample, reduction)


class DiceModule(Module):
    """
    Dice Module for segmentation.

    Optimizes the Dice coefficient (F1 score) directly.
    Good for highly imbalanced segmentation tasks.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice

    References:
        - Milletari et al. "V-Net: Fully Convolutional Neural Networks for
          Volumetric Medical Image Segmentation" (2016)

    Example:
        >>> loss = DiceModule()
        >>> loss(predictions, targets, smooth=1.0)
    """

    def forward(self, predictions: Tensor, targets: Tensor,
                smooth: float = 1.0, reduction: str = 'mean') -> Tensor:
        """
        Args:
            predictions: (N, C, H, W) or (N, H, W) predicted probabilities
            targets: Same shape as predictions, binary masks
            smooth: Smoothing factor to avoid division by zero
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Dice loss (1 - Dice coefficient)
        """
        # Flatten spatial dimensions while keeping batch and channel dims
        if predictions.data.ndim == 4:
            # (N, C, H, W) -> (N, C, H*W)
            N, C, H, W = predictions.shape
            pred_flat = predictions.reshape((N, C, -1))
            target_flat = targets.reshape((N, C, -1))

            # Compute Dice per channel
            intersection = (pred_flat * target_flat).sum(axis=-1)  # (N, C)
            pred_sum = pred_flat.sum(axis=-1)  # (N, C)
            target_sum = target_flat.sum(axis=-1)  # (N, C)

            dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
            loss = 1 - dice  # (N, C)

            # Average over channels, then reduce over batch
            loss = loss.mean(axis=-1)  # (N,)
        elif predictions.data.ndim == 3:
            # (N, H, W) -> binary segmentation
            N, H, W = predictions.shape
            pred_flat = predictions.reshape((N, -1))
            target_flat = targets.reshape((N, -1))

            intersection = (pred_flat * target_flat).sum(axis=-1)  # (N,)
            pred_sum = pred_flat.sum(axis=-1)
            target_sum = target_flat.sum(axis=-1)

            dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
            loss = 1 - dice  # (N,)
        else:
            # Assume flattened already
            intersection = (predictions * targets).sum(axis=-1)
            pred_sum = predictions.sum(axis=-1)
            target_sum = targets.sum(axis=-1)

            dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
            loss = 1 - dice

        return _reduce(loss, reduction)


# =============================================================================
# Functional Interfaces
# =============================================================================

