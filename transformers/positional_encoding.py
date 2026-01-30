import numpy as np


def sinusoidal_positional_encoding(n_positions, d_model):
    """
    Generate sinusoidal positional encodings (from "Attention Is All You Need").

    The encoding uses sine and cosine functions of different frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Parameters:
        n_positions: int - Maximum sequence length (number of positions)
        d_model: int - Embedding dimension (must be even)

    Returns:
        pe: np.ndarray of shape (n_positions, d_model) - Positional encodings

    Properties to achieve:
        - Each position gets a unique encoding
        - The encoding allows the model to learn relative positions
        - sin/cos allow the model to extrapolate to longer sequences
    """
    assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"

    pe = np.zeros((n_positions, d_model))

    # TODO: Implement sinusoidal positional encoding
    #
    # Step 1: Create position indices [0, 1, 2, ..., n_positions-1]
    # positions = np.arange(n_positions)[:, np.newaxis]  # shape: (n_positions, 1)
    #
    # Step 2: Create dimension indices for the division term
    # For dimension 2i, use i in the formula
    # div_term = 10000 ** (2 * np.arange(d_model // 2) / d_model)  # shape: (d_model // 2,)
    #
    # Step 3: Compute sin for even indices, cos for odd indices
    # pe[:, 0::2] = np.sin(positions / div_term)  # Even dimensions
    # pe[:, 1::2] = np.cos(positions / div_term)  # Odd dimensions

    return pe


def learned_positional_encoding(n_positions, d_model, seed=None):
    """
    Initialize learned positional encodings (randomly initialized, then trained).

    Unlike sinusoidal encodings, these are learned parameters that get updated
    during training via backpropagation.

    Parameters:
        n_positions: int - Maximum sequence length
        d_model: int - Embedding dimension
        seed: optional int - Random seed for reproducibility

    Returns:
        pe: np.ndarray of shape (n_positions, d_model) - Initialized positional encodings

    Note: In practice, these would be nn.Parameter in PyTorch and trained.
    Here we just initialize them (e.g., from a normal distribution).
    """
    if seed is not None:
        np.random.seed(seed)

    # TODO: Initialize learned positional encodings
    # Common initialization: small random values from N(0, 0.02) or similar
    # pe = np.random.randn(n_positions, d_model) * 0.02

    pe = np.zeros((n_positions, d_model))
    return pe


def rotary_positional_encoding(x, positions):
    """
    Apply Rotary Positional Encoding (RoPE) to input embeddings.

    RoPE encodes position by rotating the embedding vectors. It's used in
    models like LLaMA and has the property that the dot product between
    two position-encoded vectors depends only on their relative position.

    Parameters:
        x: np.ndarray of shape (n, d) - Input embeddings
        positions: np.ndarray of shape (n,) - Position indices for each token

    Returns:
        x_rotated: np.ndarray of shape (n, d) - Position-encoded embeddings

    The rotation is applied to pairs of dimensions:
        [x_0, x_1] -> [x_0 * cos(θ) - x_1 * sin(θ), x_0 * sin(θ) + x_1 * cos(θ)]
    where θ depends on position and dimension index.
    """
    n, d = x.shape
    assert d % 2 == 0, "Dimension must be even for RoPE"

    x_rotated = np.zeros_like(x)

    # TODO: Implement RoPE
    #
    # Step 1: Compute rotation angles for each position and dimension pair
    # For dimension pair i, the base frequency is: θ_i = 10000^(-2i/d)
    # The angle at position p is: p * θ_i
    #
    # Step 2: Apply rotation to each pair of dimensions
    # For dimensions (2i, 2i+1):
    #   x_rotated[:, 2i]   = x[:, 2i] * cos(angle) - x[:, 2i+1] * sin(angle)
    #   x_rotated[:, 2i+1] = x[:, 2i] * sin(angle) + x[:, 2i+1] * cos(angle)

    return x_rotated


def add_positional_encoding(x, pe):
    """
    Add positional encoding to input embeddings.

    Parameters:
        x: np.ndarray of shape (n, d) - Input token embeddings
        pe: np.ndarray of shape (max_len, d) - Positional encodings

    Returns:
        x_pos: np.ndarray of shape (n, d) - Position-encoded embeddings
    """
    n = x.shape[0]
    # TODO: Add positional encoding to input
    # x_pos = x + pe[:n]

    x_pos = x
    return x_pos


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Visualize sinusoidal positional encodings
    n_positions, d_model = 100, 64
    pe = sinusoidal_positional_encoding(n_positions, d_model)

    plt.figure(figsize=(12, 4))
    plt.imshow(pe, aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.title('Sinusoidal Positional Encoding')
    plt.savefig('positional_encoding_viz.png', dpi=150)
    plt.show()
