"""
Lasso Regression Module
=======================
Implementation of Least Absolute Shrinkage and Selection Operator (Lasso).

Lasso regression adds L1 penalty to linear regression loss for automatic feature
selection by shrinking irrelevant coefficients exactly to zero.

IMPLEMENTATION STATUS
---------------------
Type: STUB
Complexity: O(n * d * iterations) for coordinate descent, no closed-form solution
Status: Requires Implementation
Learning Focus: L1 regularization, sparsity, feature selection, proximal methods

PREREQUISITES
-------------
- Linear regression fundamentals (linear.py)
- Ridge regression (ridge.py) for L2 regularization comparison
- Proximal gradient methods
- Coordinate descent optimization
- Python: NumPy, typing, scipy (optional)

THEORY
------
Lasso regression adds L1 penalty (sum of absolute values) to the loss function,
inducing sparsity: many coefficients become exactly zero, providing automatic
feature selection. Unlike ridge regression which shrinks all coefficients,
lasso performs "hard" shrinkage, eliminating irrelevant features entirely.
This makes Lasso invaluable for high-dimensional data where d > n.
The L1 penalty creates a non-smooth optimization landscape, but still convex
with unique minimum. No closed-form solution exists; iterative methods (coordinate
descent, proximal gradient) are necessary. The regularization parameter λ
controls sparsity: λ=0 gives OLS, larger λ produces sparser solutions.

MATHEMATICAL FORMULATION
------------------------
Given training data: X ∈ ℝ^(n×d), y ∈ ℝ^n

Lasso Loss Function (L1-Regularized MSE):
    L(β, λ) = (1/2n) * ||y - Xβ||²_2 + λ * ||β||_1
            = (1/2n) * (y - Xβ)ᵀ(y - Xβ) + λ * Σ|β_j|

    where λ ≥ 0 is regularization strength
    ||β||_1 = Σ|β_j| is L1 norm (sum of absolute values)

Note on Loss Function Form:
- Different sources use different scaling conventions
- Some use λ/2 instead of λ (affects tuning range)
- This implementation uses λ (standard in scikit-learn)

Gradient (non-differentiable due to |·|):
    ∇L(β) = -(1/n)Xᵀ(y - Xβ) + λ * sign(β)
    where sign(β) is subgradient of L1 norm

Proximal Operator (soft-thresholding):
    prox_λ(z) = sign(z) * max(|z| - λ, 0)
    = threshold each coordinate by λ

COORDINATE DESCENT ALGORITHM
----------------------------
Efficient coordinate descent for Lasso:

1. Initialize β ← 0
2. For each iteration:
   For each coordinate j = 1 to d:
     a. Compute partial residual: r_(-j) = y - X @ β + X_j * β_j
     b. Compute univariate regression: β̃_j = (1/n) * X_j ᵀ * r_(-j)
     c. Apply soft threshold: β_j ← sign(β̃_j) * max(|β̃_j| - λ, 0)

This leverages coordinate-wise separability for efficiency.

PROXIMAL GRADIENT DESCENT
-------------------------
More general approach using proximal methods:

Gradient of smooth part: ∇L_smooth(β) = -(1/n)Xᵀ(y - Xβ)
Non-smooth part: λ||β||_1 (L1 penalty)

Update:
    β_(t+1) = prox_{α*λ}(β_t - α * ∇L_smooth(β_t))
            = soft_threshold(β_t - α * ∇L_smooth(β_t), α*λ)

where α is step size

SPARSITY AND FEATURE SELECTION
------------------------------
Lasso achieves sparsity through L1 penalty:
- L1 norm ||β||_1 has sharp corners at coordinate axes
- Optimal solution often lies at corners (coefficients = 0)
- Number of non-zero coefficients typically << d

Feature Selection Property:
- Features with coefficient = 0 are "selected out"
- Provides dimensionality reduction automatically
- Interpretable: identifies relevant features

Regularization Path:
- As λ increases: more coefficients → 0
- Solution path is piecewise linear
- Can trace entire path efficiently (LARS algorithm)

ELASTIC NET CONNECTIONS
-----------------------
Elastic Net combines L1 and L2 penalties:
    L(β, λ₁, λ₂) = (1/2n)||y - Xβ||² + λ₁||β||_1 + (λ₂/2)||β||²_2

Recovers Lasso when λ₂ = 0, Ridge when λ₁ = 0.
Addresses Lasso limitations with highly correlated features.

COORDINATE-WISE SOFT THRESHOLDING
---------------------------------
For single feature regression, Lasso solution in closed form:

    β_j = soft_threshold((1/n)Xⱼᵀy, λ)
        = sign((1/n)Xⱼᵀy) * max(|(1/n)Xⱼᵀy| - λ, 0)

This shows why coordinate descent is natural for Lasso.

REFERENCES
----------
[1] Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso"
    Journal of the Royal Statistical Society, 58(1), 267-288
    https://www.jstor.org/stable/2346178

[2] Friedman, J., Hastie, T., & Tibshirani, R. (2010). "Regularization Paths for
    Generalized Linear Models via Coordinate Descent"
    https://hastie.su.domains/Papers/glmnet.pdf

[3] Boyd, S., & Parikh, N. (2014). "Proximal Algorithms"
    https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

[4] James et al. (2013). "An Introduction to Statistical Learning"
    Chapter 6.3 - The Lasso
    https://www.statlearning.com/
"""

from typing import Optional, Tuple, Union
import numpy as np


class Lasso:
    """
    Lasso Regression (L1-regularized linear regression).

    Uses L1 penalty to induce sparsity, automatically performing feature selection
    by shrinking irrelevant coefficients exactly to zero.

    COMPLEXITY
    ----------
    Time: O(n * d * iterations) using coordinate descent
    Space: O(n * d) for storing feature matrix

    ADVANTAGES
    - Automatic feature selection (sparse solutions)
    - Handles high-dimensional data (d > n)
    - Computationally efficient (coordinate descent)
    - Interpretable solutions (identifies important features)
    - Well-understood convergence properties

    DISADVANTAGES
    - Non-differentiable objective (requires specialized solvers)
    - No closed-form solution
    - Arbitrary selection among correlated features
    - Requires λ tuning (typically via cross-validation)
    - Performance depends on feature scaling

    WHEN TO USE
    - High-dimensional data with many irrelevant features
    - Need interpretable sparse solutions
    - Feature selection is important
    - Correlated features can be "ignored" (arbitrary selection okay)

    WHEN NOT TO USE
    - Need to keep highly correlated features (use Elastic Net)
    - Interpretability not important
    - All features known to be relevant
    - Extreme sparsity not desired

    Attributes
    ----------
    coefficients : Optional[np.ndarray]
        Learned Lasso coefficients β, shape (d,), many = 0
    intercept : Optional[float]
        Learned intercept term
    lambda_param : float
        L1 regularization strength λ
    n_nonzero_ : Optional[int]
        Number of non-zero coefficients (sparsity measure)
    is_fitted : bool
        Whether model has been fitted
    """

    def __init__(self, lambda_param: float = 0.1, fit_intercept: bool = True):
        """
        Initialize Lasso model.

        Parameters
        ----------
        lambda_param : float, default=0.1
            L1 regularization strength λ. Controls sparsity:
            λ = 0 → OLS (no regularization)
            λ → ∞ → all coefficients → 0 (empty model)
            Typical values: 0.001 to 1.0 (dataset-dependent)
        fit_intercept : bool, default=True
            Whether to learn intercept term separately
        """
        self.lambda_param = lambda_param
        self.fit_intercept = fit_intercept
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.n_nonzero_: Optional[int] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 10000,
        tolerance: float = 1e-4,
        verbose: bool = False
    ) -> 'Lasso':
        """
        Fit Lasso model using coordinate descent.

        Algorithm (Coordinate Descent):
        ------
        1. Initialize β ← 0
        2. If fit_intercept: compute intercept and center y
        3. For each iteration t = 1 to max_iterations:
           a. For each coordinate j = 1 to d:
              - Compute residual excluding j: r_(-j) = y - X @ β + X_j * β_j
              - Update jth coefficient:
                β̃_j = (1/n) * X_j ᵀ * r_(-j)
                β_j ← sign(β̃_j) * max(|β̃_j| - λ, 0)  [soft threshold]
           b. Check convergence: ||β_new - β_old|| < tolerance
        4. Store sparse β and intercept, set is_fitted=True

        Soft Thresholding Rule:
        -----
        The key operation: prox_λ(z) = sign(z) * max(|z| - λ, 0)
        - If |z| ≤ λ: result = 0 (threshold to exact zero)
        - If z > λ: result = z - λ > 0
        - If z < -λ: result = z + λ < 0

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)
        max_iterations : int, default=10000
            Maximum coordinate descent iterations
        tolerance : float, default=1e-4
            Convergence threshold for coefficient change
        verbose : bool, default=False
            Print iteration progress

        Returns
        -------
        Lasso
            Fitted model instance
        """
        raise NotImplementedError(
            "Implement Lasso coordinate descent: "
            "1. Validate inputs: X.shape[0] == y.shape[0]. "
            "2. Initialize β = zeros(d). "
            "3. If fit_intercept=True: "
            "   - Compute y_mean = mean(y), store as self.intercept "
            "   - Center: y_centered = y - y_mean "
            "4. For iteration in range(max_iterations): "
            "   - For each coordinate j in range(d): "
            "     a. Compute partial residual: r_neg_j = y_centered - X @ β + X[:, j] * β[j] "
            "     b. Compute univariate update: beta_tilde = (1/n) * X[:, j] ᵀ @ r_neg_j "
            "     c. Apply soft threshold: β[j] = sign(beta_tilde) * max(|beta_tilde| - lambda, 0) "
            "   - Compute coefficient change ||β_new - β_old|| "
            "   - If change < tolerance: break (convergence) "
            "5. Count non-zero coefficients: self.n_nonzero_ = sum(β != 0) "
            "6. Store self.coefficients = β, set self.is_fitted = True "
            "7. Return self"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new samples.

        Computes: ŷ = X @ β + intercept

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
            If model not fitted
        ValueError
            If feature dimension mismatch
        """
        raise NotImplementedError(
            "Implement prediction: check is_fitted, verify feature dimension, "
            "compute ŷ = X @ self.coefficients + self.intercept."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.

        R² = 1 - (SS_res / SS_tot)

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        y : np.ndarray
            Target vector, shape (n_samples,)

        Returns
        -------
        float
            R² score
        """
        raise NotImplementedError(
            "Implement R² score calculation."
        )

    def get_selected_features(self) -> np.ndarray:
        """
        Return indices of non-zero coefficients (selected features).

        Returns
        -------
        np.ndarray
            Indices j where β_j != 0, shape (n_nonzero,)
        """
        raise NotImplementedError(
            "Implement feature selection: return np.where(self.coefficients != 0)[0]."
        )


class LassoCV:
    """
    Lasso Regression with automatic λ selection via cross-validation.

    Efficiently searches for optimal λ by cross-validation without retraining
    from scratch for each λ (uses warm-starting).

    ALGORITHM
    ---------
    1. Define grid of λ values (e.g., log-spaced)
    2. For each λ:
       a. Split data into k folds
       b. For each fold:
          - Train Lasso(λ) on k-1 folds
          - Evaluate on held-out fold
       c. Compute mean CV error
    3. Select λ with minimum CV error
    4. Refit on all data with optimal λ

    ADVANTAGES
    - Automatic λ selection (no manual tuning)
    - Can warm-start from previous λ (faster)
    - Computes full regularization path
    - Cross-validation error estimates

    DISADVANTAGES
    - More computationally expensive than single fit
    - k times slower than non-CV version
    - Dependent on fold split

    Example
    -------
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = LassoCV(cv=5)
    >>> model.fit(X, y)
    >>> print(f"Optimal λ = {model.lambda_opt_}")
    >>> y_pred = model.predict(X)
    """

    def __init__(
        self,
        lambdas: Optional[np.ndarray] = None,
        cv: int = 5,
        fit_intercept: bool = True
    ):
        """
        Initialize Lasso CV model.

        Parameters
        ----------
        lambdas : np.ndarray, optional
            Candidate λ values. If None, generates log-spaced grid
        cv : int, default=5
            Number of cross-validation folds
        fit_intercept : bool, default=True
            Whether to fit intercept term
        """
        self.lambdas = lambdas if lambdas is not None else np.logspace(-4, 2, 20)
        self.cv = cv
        self.fit_intercept = fit_intercept
        self.cv_scores_: Optional[np.ndarray] = None
        self.lambda_opt_: Optional[float] = None
        self.model_: Optional[Lasso] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoCV':
        """
        Fit Lasso with automatic λ selection.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)

        Returns
        -------
        LassoCV
            Fitted model with optimal λ
        """
        raise NotImplementedError(
            "Implement Lasso CV: similar to RidgeCV but use Lasso instead. "
            "1. Create k-fold split. "
            "2. For each λ in self.lambdas: "
            "   - For each fold: train Lasso(λ) on train, evaluate on test "
            "   - Compute mean CV score "
            "3. Select optimal λ = argmin(cv_scores_). "
            "4. Refit Lasso(lambda_opt) on all data. "
            "5. Store self.lambda_opt_, self.model_, self.cv_scores_. "
            "6. Return self."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with optimal λ model."""
        raise NotImplementedError(
            "Delegate to self.model_.predict(X)."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² with optimal λ model."""
        raise NotImplementedError(
            "Delegate to self.model_.score(X, y)."
        )


class ElasticNet:
    """
    Elastic Net Regression (L1 + L2 regularization).

    Combines Lasso (L1) and Ridge (L2) penalties to balance feature selection
    with stability, especially useful when features are highly correlated.

    ELASTIC NET LOSS
    ----------------
    L(β, λ₁, λ₂) = (1/2n) * ||y - Xβ||²_2 + λ₁ * ||β||_1 + (λ₂/2) * ||β||²_2

    Mixing parameter l1_ratio ∈ [0, 1]:
    L(β, λ, l1_ratio) = (1/2n) * ||y - Xβ||²_2
                        + λ * l1_ratio * ||β||_1
                        + λ * (1 - l1_ratio) / 2 * ||β||²_2

    l1_ratio = 0 → Ridge (L2 only)
    l1_ratio = 1 → Lasso (L1 only)
    l1_ratio ∈ (0, 1) → Elastic Net (both penalties)

    ADVANTAGES
    - Feature selection like Lasso
    - Handles correlated features better (Ridge component)
    - Encourages grouped coefficients to be selected
    - More stable than pure Lasso

    DISADVANTAGES
    - Two hyperparameters to tune (λ₁, λ₂)
    - More complex than Lasso or Ridge
    - Still requires cross-validation

    Example
    -------
    >>> model = ElasticNet(lambda_param=0.1, l1_ratio=0.5)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        lambda_param: float = 0.1,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True
    ):
        """
        Initialize Elastic Net model.

        Parameters
        ----------
        lambda_param : float, default=0.1
            Total regularization strength
        l1_ratio : float, default=0.5
            Balance between L1 (Lasso) and L2 (Ridge)
            0 = pure Ridge, 1 = pure Lasso, 0.5 = equal mix
        fit_intercept : bool, default=True
            Whether to fit intercept term
        """
        self.lambda_param = lambda_param
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 10000,
        tolerance: float = 1e-4
    ) -> 'ElasticNet':
        """
        Fit Elastic Net using coordinate descent.

        Similar to Lasso but with combined L1 and L2 penalties.
        Use proximal operator:
            prox(z) = sign(z) * max(|z| - λ₁, 0) / (1 + λ₂)

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)
        max_iterations : int, default=10000
            Maximum coordinate descent iterations
        tolerance : float, default=1e-4
            Convergence threshold

        Returns
        -------
        ElasticNet
            Fitted model
        """
        raise NotImplementedError(
            "Implement Elastic Net coordinate descent: "
            "1. Compute λ₁ = lambda_param * l1_ratio (L1 weight). "
            "2. Compute λ₂ = lambda_param * (1 - l1_ratio) (L2 weight). "
            "3. Similar to Lasso but: "
            "   - Apply soft threshold with λ₁ "
            "   - Divide by (1 + λ₂) to account for L2 penalty "
            "   - β_j ← sign(beta_tilde) * max(|beta_tilde| - λ₁, 0) / (1 + λ₂) "
            "4. Otherwise same coordinate descent loop."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError(
            "Implement prediction: ŷ = X @ β + intercept."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        raise NotImplementedError(
            "Implement R² calculation."
        )
