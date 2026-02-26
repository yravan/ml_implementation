"""
Graph Neural Networks Module.

Implements graph neural network layers, pooling operations, and complete
architectures for learning on graph-structured data.

Theory:
    Graph Neural Networks (GNNs) process data with graph structure by
    iteratively aggregating information from neighboring nodes. The key
    operation is message passing:

    Message Passing Framework:
        1. Message: m_ij = φ(h_i, h_j, e_ij)
        2. Aggregate: m_i = ⊕_{j ∈ N(i)} m_ij
        3. Update: h_i' = ψ(h_i, m_i)

    Where:
    - h_i is node i's feature
    - e_ij is edge features
    - N(i) is i's neighborhood
    - ⊕ is a permutation-invariant aggregation (sum, mean, max)

Graph Representation:
    - Node features: X ∈ R^{N × F} (N nodes, F features)
    - Adjacency: A ∈ R^{N × N} or edge list
    - Edge features: E ∈ R^{M × D} (M edges, D features)

Submodules:
    - layers: GNN layer implementations (GCN, GAT, GraphSAGE, GIN)
    - pooling: Graph pooling operations (global, hierarchical)
    - models: Complete GNN architectures

References:
    - "A Comprehensive Survey on Graph Neural Networks" (Wu et al., 2020)
      https://arxiv.org/abs/1901.00596
    - "Message Passing Neural Networks" (Gilmer et al., 2017)
      https://arxiv.org/abs/1704.01212

Implementation Status: STUB
Complexity: Advanced
Prerequisites: foundations, nn_core
"""

from .layers import GCNConv, GATConv, GraphSAGEConv, GINConv
from .pooling import GlobalPooling, TopKPooling, SAGPooling
from .models import GCN, GAT, GraphSAGE, GIN

__all__ = [
    # Layers
    'GCNConv',
    'GATConv',
    'GraphSAGEConv',
    'GINConv',
    # Pooling
    'GlobalPooling',
    'TopKPooling',
    'SAGPooling',
    # Models
    'GCN',
    'GAT',
    'GraphSAGE',
    'GIN',
]
