"""
Graph Neural Network Layers Module.

Implements various GNN layer types including GCN, GAT, GraphSAGE, and GIN.

Theory:
    GNN layers propagate information between connected nodes. Different
    layers differ in how they compute and aggregate messages:

    - GCN: Normalized mean aggregation with spectral motivation
    - GAT: Attention-weighted aggregation
    - GraphSAGE: Sample-based aggregation with various aggregators
    - GIN: Sum aggregation with MLP (maximally expressive)

Spectral vs Spatial:
    - Spectral: Define convolution via graph Fourier transform
    - Spatial: Define convolution via message passing
    Most modern methods use spatial interpretation.

References:
    - GCN: "Semi-Supervised Classification with GCNs" (Kipf & Welling, 2017)
      https://arxiv.org/abs/1609.02907
    - GAT: "Graph Attention Networks" (Veličković et al., 2018)
      https://arxiv.org/abs/1710.10903
    - GraphSAGE: "Inductive Representation Learning" (Hamilton et al., 2017)
      https://arxiv.org/abs/1706.02216
    - GIN: "How Powerful are Graph Neural Networks?" (Xu et al., 2019)
      https://arxiv.org/abs/1810.00826

Implementation Status: STUB
Complexity: Advanced
Prerequisites: foundations, nn_core
"""

import numpy as np
from typing import Tuple, Optional, List
from abc import ABC, abstractmethod

__all__ = ['GCNConv', 'GATConv', 'GraphSAGEConv', 'GINConv']


class MessagePassingLayer(ABC):
    """
    Abstract base class for message passing layers.

    Theory:
        Message passing consists of three steps:
        1. Message: Compute message from neighbor j to node i
        2. Aggregate: Combine messages from all neighbors
        3. Update: Update node representation

    The general form:
        h_i' = γ(h_i, ⊕_{j∈N(i)} φ(h_i, h_j, e_ij))
    """

    @abstractmethod
    def message(
        self,
        x_i: np.ndarray,
        x_j: np.ndarray,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute messages from j to i."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(
        self,
        messages: np.ndarray,
        index: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """Aggregate messages at each node."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        aggregated: np.ndarray,
        x: np.ndarray
    ) -> np.ndarray:
        """Update node features."""
        raise NotImplementedError


class GCNConv(MessagePassingLayer):
    """
    Graph Convolutional Network layer.

    Theory:
        GCN performs spectral convolution approximated as:
            H' = σ(D̃^{-1/2} Ã D̃^{-1/2} H W)

        Where:
        - Ã = A + I (adjacency with self-loops)
        - D̃ is degree matrix of Ã
        - W is learnable weight matrix

        Message passing view:
            h_i' = σ(W · MEAN_{j∈N(i)∪{i}} (h_j / √(d_i d_j)))

    Example:
        >>> gcn = GCNConv(in_channels=64, out_channels=32)
        >>> x_out = gcn.forward(x, edge_index)

    References:
        - "Semi-Supervised Classification with Graph Convolutional Networks"
          (Kipf & Welling, 2017) https://arxiv.org/abs/1609.02907
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True
    ):
        """
        Initialize GCN layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            add_self_loops: If True, add self-loops to adjacency
            normalize: If True, apply symmetric normalization
            bias: If True, add bias term
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        # Initialize weights
        self.W = np.random.randn(in_channels, out_channels) * 0.01
        self.b = np.zeros(out_channels) if bias else None

    def forward(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices [source, target]
            edge_weight: (E,) optional edge weights

        Returns:
            (N, out_channels) updated node features

        Implementation hints:
            1. Add self-loops if specified
            2. Compute normalization: 1/sqrt(d_i * d_j)
            3. Transform features: H = X @ W
            4. Propagate: H' = norm @ H (sparse matmul)
            5. Add bias
        """
        raise NotImplementedError(
            "Implement GCN forward pass. "
            "Symmetric normalization + linear transform."
        )

    def message(self, x_i, x_j, edge_attr=None):
        """GCN message: just the neighbor features."""
        return x_j

    def aggregate(self, messages, index, num_nodes):
        """GCN aggregation: normalized sum."""
        raise NotImplementedError("Scatter add with normalization.")

    def update(self, aggregated, x):
        """GCN update: linear transform."""
        return aggregated @ self.W + (self.b if self.b is not None else 0)


class GATConv(MessagePassingLayer):
    """
    Graph Attention Network layer.

    Theory:
        GAT uses attention to weight neighbor contributions:
            α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
            h_i' = σ(Σ_{j∈N(i)} α_ij W h_j)

        Multi-head attention:
            h_i' = ||_{k=1}^K σ(Σ_j α_ij^k W^k h_j)

        Concatenate heads during training, average at final layer.

    Example:
        >>> gat = GATConv(in_channels=64, out_channels=8, heads=8)
        >>> x_out = gat.forward(x, edge_index)  # (N, 64)

    References:
        - "Graph Attention Networks" (Veličković et al., 2018)
          https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Initialize GAT layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            heads: Number of attention heads
            concat: If True, concat heads; else average
            negative_slope: LeakyReLU negative slope
            dropout: Attention dropout rate
            bias: If True, add bias
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Initialize weights
        self.W = np.random.randn(in_channels, heads * out_channels) * 0.01
        self.a_src = np.random.randn(heads, out_channels) * 0.01
        self.a_dst = np.random.randn(heads, out_channels) * 0.01
        self.b = np.zeros(heads * out_channels if concat else out_channels) if bias else None

    def forward(
        self,
        x: np.ndarray,
        edge_index: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices

        Returns:
            (N, heads * out_channels) if concat else (N, out_channels)

        Implementation hints:
            1. Linear transform: H = X @ W, reshape to (N, heads, out)
            2. Compute attention scores: e_ij = a_src · h_i + a_dst · h_j
            3. Apply LeakyReLU
            4. Softmax over neighbors
            5. Weighted aggregation
            6. Concat or mean over heads
        """
        raise NotImplementedError(
            "Implement GAT forward with multi-head attention. "
            "Compute attention weights, then weighted sum."
        )

    def message(self, x_i, x_j, edge_attr=None):
        """GAT message: attention-weighted features."""
        raise NotImplementedError("Compute attention and weight messages.")

    def aggregate(self, messages, index, num_nodes):
        """GAT aggregation: weighted sum."""
        raise NotImplementedError("Scatter add weighted messages.")

    def update(self, aggregated, x):
        """GAT update: add bias."""
        return aggregated + (self.b if self.b is not None else 0)


class GraphSAGEConv(MessagePassingLayer):
    """
    GraphSAGE convolution layer.

    Theory:
        GraphSAGE samples and aggregates features from neighbors:
            h_N(i) = AGGREGATE({h_j : j ∈ N(i)})
            h_i' = σ(W · [h_i || h_N(i)])

        Aggregators:
        - Mean: h_N = mean(h_j)
        - LSTM: h_N = LSTM([h_j for j in N(i)])
        - Pool: h_N = max(σ(W_pool h_j + b))

    Example:
        >>> sage = GraphSAGEConv(in_channels=64, out_channels=32, aggr='mean')
        >>> x_out = sage.forward(x, edge_index)

    References:
        - "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017)
          https://arxiv.org/abs/1706.02216
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = True,
        bias: bool = True
    ):
        """
        Initialize GraphSAGE layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            aggr: Aggregation type ('mean', 'max', 'sum')
            normalize: If True, L2-normalize output
            bias: If True, add bias
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize

        # Weights for self and neighbor
        self.W_self = np.random.randn(in_channels, out_channels) * 0.01
        self.W_neigh = np.random.randn(in_channels, out_channels) * 0.01
        self.b = np.zeros(out_channels) if bias else None

    def forward(
        self,
        x: np.ndarray,
        edge_index: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices

        Returns:
            (N, out_channels) updated features

        Implementation hints:
            1. Aggregate neighbor features: h_N = AGG(h_j for j in N(i))
            2. Combine: h' = W_self @ h_i + W_neigh @ h_N
            3. Apply activation
            4. L2 normalize if specified
        """
        raise NotImplementedError(
            "Implement GraphSAGE forward. "
            "Aggregate neighbors, concat with self, transform."
        )

    def message(self, x_i, x_j, edge_attr=None):
        return x_j

    def aggregate(self, messages, index, num_nodes):
        """Aggregate based on self.aggr type."""
        raise NotImplementedError("Implement mean/max/sum aggregation.")

    def update(self, aggregated, x):
        """Combine self and aggregated."""
        raise NotImplementedError("h' = W_self @ x + W_neigh @ aggregated")


class GINConv(MessagePassingLayer):
    """
    Graph Isomorphism Network layer.

    Theory:
        GIN is the most expressive GNN under the WL test framework:
            h_i' = MLP((1 + ε) · h_i + Σ_{j∈N(i)} h_j)

        Where ε is a learnable or fixed scalar. Sum aggregation
        (not mean) is crucial for expressiveness.

    Example:
        >>> gin = GINConv(nn_module=mlp, train_eps=True)
        >>> x_out = gin.forward(x, edge_index)

    References:
        - "How Powerful are Graph Neural Networks?" (Xu et al., 2019)
          https://arxiv.org/abs/1810.00826
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        eps: float = 0.0,
        train_eps: bool = False
    ):
        """
        Initialize GIN layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            hidden_channels: Hidden dimension in MLP
            eps: Initial epsilon value
            train_eps: If True, learn epsilon
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps
        self.train_eps = train_eps

        # MLP weights
        self.W1 = np.random.randn(in_channels, hidden_channels) * 0.01
        self.b1 = np.zeros(hidden_channels)
        self.W2 = np.random.randn(hidden_channels, out_channels) * 0.01
        self.b2 = np.zeros(out_channels)

    def forward(
        self,
        x: np.ndarray,
        edge_index: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices

        Returns:
            (N, out_channels) updated features

        Implementation hints:
            1. Sum neighbor features
            2. Add scaled self: (1 + eps) * x + sum_neighbors
            3. Apply MLP: ReLU(W1 @ h + b1) @ W2 + b2
        """
        raise NotImplementedError(
            "Implement GIN forward. "
            "(1 + eps) * self + sum(neighbors), then MLP."
        )

    def message(self, x_i, x_j, edge_attr=None):
        return x_j

    def aggregate(self, messages, index, num_nodes):
        """Sum aggregation for GIN."""
        raise NotImplementedError("Scatter sum.")

    def update(self, aggregated, x):
        """Apply MLP."""
        raise NotImplementedError("MLP((1+eps)*x + agg)")


# Utility functions

def add_self_loops(
    edge_index: np.ndarray,
    num_nodes: int
) -> np.ndarray:
    """Add self-loops to edge index."""
    self_loops = np.stack([np.arange(num_nodes), np.arange(num_nodes)])
    return np.concatenate([edge_index, self_loops], axis=1)


def compute_degree(
    edge_index: np.ndarray,
    num_nodes: int
) -> np.ndarray:
    """Compute node degrees from edge index."""
    target = edge_index[1]
    return np.bincount(target, minlength=num_nodes)


def gcn_norm(
    edge_index: np.ndarray,
    num_nodes: int
) -> np.ndarray:
    """Compute GCN normalization coefficients."""
    deg = compute_degree(edge_index, num_nodes)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0

    source, target = edge_index
    norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]
    return norm
