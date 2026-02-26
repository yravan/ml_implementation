"""
Linear Regression Module
======================
Implementation of Linear Regression using Ordinary Least Squares (OLS) with multiple
solving methods: closed-form solution, gradient descent, and stochastic gradient descent.

IMPLEMENTATION STATUS
---------------------
Type: STUB
Complexity: O(n * d^2) for closed-form, O(n * d * iterations) for gradient descent
Status: Requires Implementation
Learning Focus: Fundamental supervised learning, matrix operations, optimization basics

PREREQUISITES
-------------
- Linear algebra (matrix multiplication, inversion, rank)
- Calculus (partial derivatives, chain rule)
- Probability (expectation, variance)
- Python: NumPy, typing, ABC

THEORY
------
Linear regression models the relationship between independent variables (features) and
a dependent variable (target) using a linear function. It assumes the target is a linear
combination of features plus random noise: y = X*β + ε, where β are coefficients to learn.
The Ordinary Least Squares (OLS) method minimizes the sum of squared residuals, making
it optimal for minimizing prediction error under Gaussian noise assumptions. Multiple
solving approaches exist: closed-form (analytical), gradient descent (iterative),
and stochastic gradient descent (online). The model assumes homoscedastic errors
(constant variance) and no multicollinearity between features.

MATHEMATICAL FORMULATION
------------------------
Given training data: X ∈ ℝ^(n×d), y ∈ ℝ^n
Model: ŷ = Xβ, where β ∈ ℝ^d

Loss Function (Mean Squared Error):
    L(β) = (1/2n) * ||y - Xβ||²_2
         = (1/2n) * (y - Xβ)ᵀ(y - Xβ)

Closed-form Solution (Normal Equation):
    β_opt = (XᵀX)^(-1)Xᵀy
    Derivation: ∇_β L(β) = 0 ⟹ (XᵀX)β = Xᵀy

Gradient:
    ∇_β L(β) = -(1/n)Xᵀ(y - Xβ) = -(1/n)Xᵀ(y - ŷ)

Gradient Descent Update:
    β_(t+1) = β_t - α∇_β L(β_t)
    where α > 0 is the learning rate (step size)

Predictions:
    ŷ_new = X_new * β

Performance Metrics:
    MSE = (1/n) * Σ(y_i - ŷ_i)²
    RMSE = sqrt(MSE)
    R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_i - ŷ_i)²
        and SS_tot = Σ(y_i - ȳ)²

REFERENCES
----------
[1] James et al. (2013). "An Introduction to Statistical Learning"
    https://www.statlearning.com/

[2] Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective"
    https://mitpress.mit.edu/9780262018029/machine-learning/

[3] Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"
    https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf

[4] Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning"
    Chapter 5.1.4 - Linear Regression
    https://www.deeplearningbook.org/
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np


class LinearRegression(ABC):
    """
    Abstract base class for linear regression solvers.

    This class defines the interface for different linear regression implementations
    (closed-form, gradient descent, SGD). All implementations must provide fit, predict,
    and score methods.

    Attributes
    ----------
    coefficients : Optional[np.ndarray]
        Learned regression coefficients β, shape (d,) where d is number of features
    intercept : Optional[float]
        Learned intercept term (bias)
    is_fitted : bool
        Flag indicating whether model has been fitted on training data
    """

    def __init__(self):
        """Initialize linear regression model."""
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear regression model to training data.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix, shape (n_samples, n_features)
            where n_samples >= n_features for well-posed problem
        y : np.ndarray
            Training target vector, shape (n_samples,)

        Returns
        -------
        LinearRegression
            Returns self for method chaining

        Raises
        ------
        ValueError
            If X and y have incompatible dimensions or n_samples < n_features
        """
        raise NotImplementedError(
            "Subclass must implement fit() method. "
            "Validate inputs: X.shape[0] == y.shape[0], X.ndim == 2, y.ndim == 1. "
            "Compute β and store in self.coefficients and self.intercept. "
            "Set self.is_fitted = True."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted target values, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If model has not been fitted yet (is_fitted == False)
        ValueError
            If X has incorrect number of features
        """
        raise NotImplementedError(
            "Implement prediction: check is_fitted, verify feature dimension, "
            "compute ŷ = X @ self.coefficients + self.intercept. "
            "Input shape: (n_samples, n_features), Output shape: (n_samples,)"
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² coefficient of determination on given data.

        R² measures the proportion of variance in the target explained by the model.
        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_i - ŷ_i)² (sum of squared residuals)
        and SS_tot = Σ(y_i - ȳ)² (total sum of squares)

        R² = 1.0 indicates perfect prediction, 0.0 indicates predictions at mean level,
        negative values indicate worse than baseline.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        y : np.ndarray
            Target vector, shape (n_samples,)

        Returns
        -------
        float
            R² coefficient, typically in range [-∞, 1.0]
        """
        raise NotImplementedError(
            "Implement R² score: get predictions ŷ, compute SS_res and SS_tot, "
            "return 1 - (SS_res / SS_tot). "
            "Handle edge case: SS_tot = 0 (constant y) → return 0."
        )


class LinearRegressionClosed(LinearRegression):
    """
    Linear regression using closed-form solution (Normal Equation).

    Solves the normal equation: β = (XᵀX)⁻¹Xᵀy analytically.

    COMPLEXITY
    ----------
    Time: O(n * d²) - matrix multiplication and inversion
    Space: O(n * d) - storing X and gram matrix XᵀX

    ADVANTAGES
    - Exact solution (no approximation)
    - No hyperparameters to tune
    - Single pass algorithm

    DISADVANTAGES
    - Slow for very large d (feature dimension)
    - Requires XᵀX to be invertible (full rank)
    - Sensitive to feature scaling
    - Cannot handle extremely large datasets (memory)

    NUMERICAL STABILITY CONSIDERATIONS
    - XᵀX can be ill-conditioned (nearly singular) causing numerical instability
    - Use ridge regression (add regularization) if XᵀX has small eigenvalues
    - Condition number: κ(XᵀX) = λ_max / λ_min should be < 10³ for stability
    - Consider using QR or SVD decomposition instead of direct inversion

    Example
    -------
    >>> X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y_train = np.array([2, 3, 4, 5])
    >>> model = LinearRegressionClosed()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_train)
    >>> r2 = model.score(X_train, y_train)
    """

    def __init__(self, use_svd: bool = False):
        """
        Initialize closed-form linear regression solver.

        Parameters
        ----------
        use_svd : bool, default=False
            If True, use SVD decomposition instead of direct matrix inversion.
            SVD is more numerically stable for ill-conditioned XᵀX matrices.
            Recommended for datasets with many features or multicollinearity.
        """
        super().__init__()
        self.use_svd = use_svd
        self.rank: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionClosed':
        """
        Fit linear regression using Normal Equation.

        For the standard case with intercept:
        1. Add column of ones to X for intercept term
        2. Compute X̃ = [1, X] (augmented matrix)
        3. Solve: β = (X̃ᵀX̃)⁻¹X̃ᵀy using either direct inversion or SVD

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)

        Returns
        -------
        LinearRegressionClosed
            Fitted model instance
        """
        raise NotImplementedError(
            "Implement closed-form solution: "
            "1. Validate inputs and shapes. "
            "2. Add intercept term (column of ones or track separately). "
            "3. If use_svd=True: use np.linalg.svd, solve via pseudoinverse. "
            "4. Else: compute (XᵀX)⁻¹Xᵀy using np.linalg.inv. "
            "5. Store coefficients (exclude intercept column) and intercept separately. "
            "6. Set is_fitted=True. "
            "Implementation note: Check condition number of XᵀX for warnings."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using learned parameters.

        Computes: ŷ = X @ β + intercept

        Parameters
        ----------
        X : np.ndarray
            Prediction features, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,)
        """
        raise NotImplementedError(
            "Implement prediction: check is_fitted, verify X.shape[1] == n_features. "
            "Compute ŷ = X @ self.coefficients + self.intercept. "
            "Return predictions as 1D array of shape (n_samples,)."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score on given data."""
        raise NotImplementedError(
            "Implement R² calculation: predictions, SS_res, SS_tot, return 1 - SS_res/SS_tot."
        )


class LinearRegressionGD(LinearRegression):
    """
    Linear regression using Gradient Descent optimization.

    Iteratively updates coefficients to minimize MSE loss:
    β_(t+1) = β_t - α * ∇L(β_t)

    COMPLEXITY
    ----------
    Time: O(n * d * iterations) - n samples, d features, iterations until convergence
    Space: O(n * d) - storing feature matrix

    ADVANTAGES
    - Handles large feature dimensions well
    - Easily extendable to stochastic variants (SGD)
    - Interpretable convergence behavior
    - Works with regularization (ridge, lasso)

    DISADVANTAGES
    - Requires learning rate tuning
    - Convergence speed depends on feature scaling
    - May not reach global optimum if learning rate too high
    - Slower than closed-form for small datasets

    HYPERPARAMETERS
    ---------------
    learning_rate (α) : float
        Step size for each gradient update. Typical values: 0.001 to 0.1
        Too small → slow convergence; too large → divergence
    n_iterations : int
        Maximum number of gradient updates
    tolerance : float
        Stop if loss change < tolerance (early stopping)
    fit_intercept : bool
        Whether to learn bias term separately

    Example
    -------
    >>> X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y_train = np.array([2, 3, 4, 5])
    >>> model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_train)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-4,
        fit_intercept: bool = True,
        verbose: bool = False
    ):
        """
        Initialize gradient descent linear regression solver.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Learning rate α for gradient step. Critical hyperparameter.
            Should be tuned based on data scale and problem characteristics.
        n_iterations : int, default=1000
            Maximum number of gradient steps to take
        tolerance : float, default=1e-4
            Convergence threshold - stop if |L(t) - L(t-1)| < tolerance
        fit_intercept : bool, default=True
            If True, learns intercept term separately (recommended)
        verbose : bool, default=False
            If True, print loss every 100 iterations for debugging
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.losses: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionGD':
        """
        Fit linear regression using batch gradient descent.

        Algorithm:
        1. Initialize β randomly or to zeros (shape: d,)
        2. For each iteration t = 1 to n_iterations:
           a. Compute predictions: ŷ = X @ β + intercept
           b. Compute residuals: e = y - ŷ
           c. Compute gradient: ∇L = -(1/n) * Xᵀe
           d. Update: β ← β - α * ∇L
           e. Compute loss and check convergence
        3. Store final β and intercept, set is_fitted=True

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)

        Returns
        -------
        LinearRegressionGD
            Fitted model instance
        """
        raise NotImplementedError(
            "Implement gradient descent: "
            "1. Validate inputs and initialize β (zeros or small random). "
            "2. If fit_intercept=True: compute y_mean and center y, update separately. "
            "3. Scale X for numerical stability (optional but recommended). "
            "4. Loop for n_iterations: "
            "   - Compute predictions ŷ = X @ β "
            "   - Compute MSE loss and append to self.losses "
            "   - Compute gradient ∇L = -(1/n) * Xᵀ(y - ŷ) "
            "   - Update β ← β - learning_rate * ∇L "
            "   - Check convergence condition (loss improvement < tolerance) "
            "5. Store coefficients and intercept, set is_fitted=True. "
            "6. Optional: print losses if verbose=True"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using learned coefficients.

        Parameters
        ----------
        X : np.ndarray
            Prediction features, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,)
        """
        raise NotImplementedError(
            "Implement prediction: check is_fitted, "
            "compute ŷ = X @ self.coefficients + self.intercept."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score on given data."""
        raise NotImplementedError(
            "Implement R² calculation: get predictions, compute SS_res and SS_tot, "
            "return 1 - (SS_res / SS_tot)."
        )


class LinearRegressionSGD(LinearRegression):
    """
    Linear regression using Stochastic Gradient Descent (SGD).

    Updates coefficients using mini-batches instead of full dataset, enabling
    online learning and handling of extremely large datasets.

    COMPLEXITY
    ----------
    Time: O(batch_size * d * iterations) - much smaller effective n than BGD
    Space: O(batch_size * d) - only mini-batch in memory

    ADVANTAGES
    - Handles massive datasets (streaming/online learning)
    - Often converges faster than batch GD in practice
    - Natural regularization from noise in mini-batches
    - Can escape local minima due to stochasticity

    DISADVANTAGES
    - Noisier convergence than batch GD
    - Requires learning rate scheduling for convergence
    - Less stable than batch GD
    - Need to tune batch size as additional hyperparameter

    HYPERPARAMETERS
    ---------------
    learning_rate (α) : float
        Initial learning rate, often decayed during training
    batch_size : int
        Number of samples per mini-batch
    n_epochs : int
        Number of passes through full dataset
    learning_rate_schedule : str
        'constant', 'linear' decay, or 'exponential' decay

    Example
    -------
    >>> X_train = np.random.randn(10000, 50)
    >>> y_train = np.random.randn(10000)
    >>> model = LinearRegressionSGD(batch_size=32, n_epochs=10)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_train)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 10,
        learning_rate_schedule: str = 'constant',
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize SGD linear regression solver.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Initial learning rate
        batch_size : int, default=32
            Number of samples per mini-batch. Typical: 16-256
        n_epochs : int, default=10
            Number of complete passes through dataset
        learning_rate_schedule : str, default='constant'
            How to adjust learning rate: 'constant', 'linear', or 'exponential'
        fit_intercept : bool, default=True
            Whether to learn intercept term
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool, default=False
            Whether to print progress
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate_schedule = learning_rate_schedule
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.losses: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionSGD':
        """
        Fit linear regression using stochastic gradient descent.

        Algorithm:
        1. Initialize β to zeros or small random values
        2. For each epoch e = 1 to n_epochs:
           a. Shuffle dataset randomly
           b. Split into mini-batches of size batch_size
           c. For each mini-batch:
              - Compute predictions on mini-batch
              - Compute gradient on mini-batch
              - Update β ← β - α_t * ∇L
              - Decay learning rate if using schedule
        3. Optionally track per-epoch average loss

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)

        Returns
        -------
        LinearRegressionSGD
            Fitted model instance
        """
        raise NotImplementedError(
            "Implement SGD: "
            "1. Validate inputs and initialize β. "
            "2. Set random seed if provided. "
            "3. For each epoch: "
            "   a. Create random permutation of indices (shuffle). "
            "   b. Split into mini-batches. "
            "   c. For each mini-batch: "
            "      - Extract X_batch, y_batch "
            "      - Compute predictions ŷ_batch "
            "      - Compute gradient on mini-batch "
            "      - Update β ← β - current_lr * gradient "
            "      - Decay learning rate based on schedule "
            "   d. Compute epoch loss "
            "4. Store coefficients, intercept, is_fitted=True. "
            "Learning rate decay: "
            "  - 'constant': α_t = α₀ "
            "  - 'linear': α_t = α₀ * (1 - t/T) "
            "  - 'exponential': α_t = α₀ * decay^t"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using learned coefficients.

        Parameters
        ----------
        X : np.ndarray
            Prediction features, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,)
        """
        raise NotImplementedError(
            "Implement prediction: check is_fitted, "
            "compute ŷ = X @ self.coefficients + self.intercept."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score on given data."""
        raise NotImplementedError(
            "Implement R² calculation: get predictions, compute SS_res and SS_tot, "
            "return 1 - (SS_res / SS_tot)."
        )
