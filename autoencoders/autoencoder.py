"""
Basic Autoencoder Implementation.

An autoencoder learns compressed representations of data by training
an encoder-decoder pair to reconstruct its input. The bottleneck layer
forces the model to learn a compact representation.

Architecture:
    Input (d) → Encoder → Latent (k) → Decoder → Output (d)

Loss: Reconstruction loss (MSE or BCE depending on data type)

Reference: MIT 6.390 Chapter 10
"""
import numpy as np


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)


def mse_loss(x_reconstructed, x_original):
    """
    Mean Squared Error reconstruction loss.

    Parameters:
        x_reconstructed: np.ndarray of shape (batch, d) - Decoder output
        x_original: np.ndarray of shape (batch, d) - Original input

    Returns:
        loss: float - Mean squared error averaged over batch and dimensions
        grad: np.ndarray of shape (batch, d) - Gradient w.r.t. x_reconstructed
    """
    loss = None
    grad = None
    return loss, grad


def bce_loss(x_reconstructed, x_original, eps=1e-8):
    """
    Binary Cross-Entropy reconstruction loss (for binary/normalized data).

    Parameters:
        x_reconstructed: np.ndarray of shape (batch, d) - Decoder output (in [0,1])
        x_original: np.ndarray of shape (batch, d) - Original input (in [0,1])
        eps: float - Small constant for numerical stability

    Returns:
        loss: float - BCE loss averaged over batch and dimensions
        grad: np.ndarray of shape (batch, d) - Gradient w.r.t. x_reconstructed
    """
    loss = None
    grad = None
    return loss, grad


class Encoder:
    """
    Encoder network: maps input x ∈ R^d to latent z ∈ R^k.

    Architecture: Linear → ReLU → Linear
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize encoder parameters.

        Parameters:
            input_dim: int (d) - Input dimension
            hidden_dim: int - Hidden layer dimension
            latent_dim: int (k) - Latent space dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Initialize weights (Xavier initialization recommended)
        self.W1 = None  # (input_dim, hidden_dim)
        self.b1 = None  # (hidden_dim,)
        self.W2 = None  # (hidden_dim, latent_dim)
        self.b2 = None  # (latent_dim,)

        # Cache for backward pass
        self.cache = {}

    def forward(self, x):
        """
        Forward pass through encoder.

        Parameters:
            x: np.ndarray of shape (batch, input_dim)

        Returns:
            z: np.ndarray of shape (batch, latent_dim) - Latent representation
        """
        z = None
        return z

    def backward(self, dz):
        """
        Backward pass through encoder.

        Parameters:
            dz: np.ndarray of shape (batch, latent_dim) - Gradient from decoder

        Returns:
            dx: np.ndarray of shape (batch, input_dim) - Gradient w.r.t. input
            grads: dict - Gradients w.r.t. parameters (dW1, db1, dW2, db2)
        """
        dx = None
        grads = {}
        return dx, grads


class Decoder:
    """
    Decoder network: maps latent z ∈ R^k to reconstruction x̂ ∈ R^d.

    Architecture: Linear → ReLU → Linear → Sigmoid (for normalized output)
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        Initialize decoder parameters.

        Parameters:
            latent_dim: int (k) - Latent space dimension
            hidden_dim: int - Hidden layer dimension
            output_dim: int (d) - Output dimension (same as input)
        """
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights
        self.W1 = None  # (latent_dim, hidden_dim)
        self.b1 = None  # (hidden_dim,)
        self.W2 = None  # (hidden_dim, output_dim)
        self.b2 = None  # (output_dim,)

        self.cache = {}

    def forward(self, z):
        """
        Forward pass through decoder.

        Parameters:
            z: np.ndarray of shape (batch, latent_dim)

        Returns:
            x_recon: np.ndarray of shape (batch, output_dim) - Reconstruction
        """
        x_recon = None
        return x_recon

    def backward(self, dx_recon):
        """
        Backward pass through decoder.

        Parameters:
            dx_recon: np.ndarray of shape (batch, output_dim) - Gradient of loss

        Returns:
            dz: np.ndarray of shape (batch, latent_dim) - Gradient w.r.t. latent
            grads: dict - Gradients w.r.t. parameters
        """
        dz = None
        grads = {}
        return dz, grads


class Autoencoder:
    """
    Complete Autoencoder combining encoder and decoder.

    Training objective: minimize reconstruction loss
        L = ||x - decode(encode(x))||²
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize autoencoder.

        Parameters:
            input_dim: int - Dimension of input data
            hidden_dim: int - Hidden layer dimension
            latent_dim: int - Dimension of latent space (bottleneck)
        """
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        """
        Full forward pass: encode then decode.

        Parameters:
            x: np.ndarray of shape (batch, input_dim)

        Returns:
            x_recon: np.ndarray of shape (batch, input_dim)
            z: np.ndarray of shape (batch, latent_dim) - Latent representation
        """
        x_recon = None
        z = None
        return x_recon, z

    def compute_loss(self, x, x_recon):
        """
        Compute reconstruction loss.

        Parameters:
            x: np.ndarray - Original input
            x_recon: np.ndarray - Reconstruction

        Returns:
            loss: float
            grad: np.ndarray - Gradient w.r.t. x_recon
        """
        return mse_loss(x_recon, x)

    def train_step(self, x, learning_rate=0.001):
        """
        Single training step: forward, compute loss, backward, update.

        Parameters:
            x: np.ndarray of shape (batch, input_dim) - Training batch
            learning_rate: float

        Returns:
            loss: float - Reconstruction loss for this batch
        """
        loss = None
        return loss

    def encode(self, x):
        """Get latent representation."""
        return self.encoder.forward(x)

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder.forward(z)

    def reconstruct(self, x):
        """Reconstruct input through bottleneck."""
        x_recon, _ = self.forward(x)
        return x_recon


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    batch_size, input_dim, hidden_dim, latent_dim = 32, 784, 256, 32

    ae = Autoencoder(input_dim, hidden_dim, latent_dim)
    x = np.random.randn(batch_size, input_dim)

    x_recon, z = ae.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
