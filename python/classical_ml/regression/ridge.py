"""
Ridge Regression Module
=======================
Implementation of Ridge Regression (L2-regularized linear regression).

Ridge regression adds L2 penalty to linear regression loss to prevent overfitting,
reduce variance, and handle multicollinear features.

IMPLEMENTATION STATUS
---------------------
Type: STUB
Complexity: O(n * d^2) for closed-form, O(n * d * iterations) for gradient descent
Status: Requires Implementation
Learning Focus: Regularization techniques, bias-variance tradeoff, hyperparameter tuning

PREREQUISITES
-------------
- Linear regression fundamentals (linear.py)
- Regularization concepts (penalty terms, constraint)
- Matrix algebra (eigenvalue decomposition)
- Cross-validation for hyperparameter tuning
- Python: NumPy, typing

THEORY
------
Ridge regression extends ordinary least squares by adding an L2 penalty (squared sum of
coefficients) to the loss function. This shrinks large coefficients toward zero, reducing
model complexity and improving generalization. Ridge regression is particularly useful
when features are highly correlated (multicollinearity) because it distributes weight
more evenly across correlated features. The method is also called Tikhonov regularization
or weight decay. The regularization strength λ controls the bias-variance tradeoff:
λ=0 recovers OLS, while larger λ increases bias but reduces variance.

MATHEMATICAL FORMULATION
------------------------
Given training data: X ∈ ℝ^(n×d), y ∈ ℝ^n

Ridge Loss Function (L2-Regularized MSE):
    L(β, λ) = (1/2n) * ||y - Xβ||²_2 + (λ/2) * ||β||²_2
            = (1/2n) * (y - Xβ)ᵀ(y - Xβ) + (λ/2) * βᵀβ

    where λ ≥ 0 is regularization parameter (lambda/alpha)

Closed-form Solution (Ridge Normal Equation):
    β_ridge = (XᵀX + λI)^(-1)Xᵀy

    Key insight: XᵀX + λI is always invertible if λ > 0 (even for singular XᵀX)
    Eigenvalue perspective: λI shifts all eigenvalues by λ

Gradient:
    ∇_β L(β, λ) = -(1/n)Xᵀ(y - Xβ) + λβ

Gradient Descent Update:
    β_(t+1) = β_t - α(-(1/n)Xᵀ(y - Xβ_t) + λβ_t)
            = β_t - α(λβ_t - (1/n)Xᵀ(y - Xβ_t))
            = (1 - αλ)β_t + (α/n)Xᵀ(y - Xβ_t)

Regularization Path:
    As λ increases: ||β||_2 monotonically decreases
    Coefficient shrinkage: more regular features shrink more
    All coefficients shrink uniformly

REGULARIZATION EFFECT
---------------------
L2 Penalty: λ/2 * Σ β_j²
- Shrinks all coefficients toward zero proportionally
- No feature selection (non-zero coefficients)
- Keeps highly correlated features

Geometric Interpretation:
- L2 penalty creates circular constraint region
- Optimization finds smallest ellipse touching constraint circle
- Ridge solution lies inside constraint region (shrunk)

BIAS-VARIANCE ANALYSIS
-----------------------
Trade-off parameter λ:
- λ = 0: OLS - high variance, low bias
- λ → ∞: Predictions → 0 - low variance, high bias
- Optimal λ balances MSE = Bias² + Variance

Bias increases with λ: E[β_ridge] ≠ β_true
Variance decreases with λ: Var[β_ridge] < Var[β_OLS]

DEGREES OF FREEDOM
------------------
Ridge regression effective degrees of freedom:
    df(λ) = Σ λ_i / (λ_i + λ)

    where λ_i are eigenvalues of XᵀX
    df(0) = d (full OLS), df(λ→∞) → 0

CROSS-VALIDATION FOR λ
----------------------
1. Split data into k folds
2. For each λ candidate:
   - For each fold:
     * Train on k-1 folds
     * Evaluate on held-out fold
   - Compute average CV error
3. Select λ with minimum CV error
4. Retrain on all data with optimal λ

REFERENCES
----------
[1] Hoerl, A. E., & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for
    Nonorthogonal Problems"
    Technometrics, 12(1), 55-67

[2] James et al. (2013). "An Introduction to Statistical Learning"
    Chapter 6.2 - Ridge Regression
    https://www.statlearning.com/

[3] Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"
    Chapter 3.4 - Ridge Regression
    https://hastie.su.domains/ElemStatLearn/

[4] Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective"
    Chapter 7.5.2 - Ridge Regression
    https://mitpress.mit.edu/9780262018029/machine-learning/
"""

from typing import Optional, Tuple, Union
import numpy as np


class RidgeRegression:
    """
    Ridge Regression (L2-regularized linear regression).

    Adds L2 penalty λ * ||β||² to OLS loss to control model complexity and
    improve generalization, especially with multicollinear features.

    COMPLEXITY
    ----------
    Time: O(n * d²) for closed-form (matrix inversion)
    Space: O(n * d + d²) for storing X and gram matrix

    ADVANTAGES
    - Simple and interpretable
    - Closed-form solution available
    - Handles multicollinearity well
    - Monotonic coefficient shrinkage

    DISADVANTAGES
    - All coefficients remain non-zero (no feature selection)
    - Requires tuning λ (typically via cross-validation)
    - Performance sensitive to feature scaling
    - Less interpretable than original features (scaling changes λ)

    WHEN TO USE
    - Multicollinear features present
    - Want to reduce variance at cost of bias
    - Need stable predictions on new data
    - Features have natural interpretation

    WHEN NOT TO USE
    - Need automatic feature selection (use Lasso instead)
    - Computational resources extremely limited
    - Interpretability critical (all features in model)

    Attributes
    ----------
    coefficients : Optional[np.ndarray]
        Learned ridge coefficients β, shape (d,)
    intercept : Optional[float]
        Learned intercept term
    lambda_param : float
        Regularization strength λ
    is_fitted : bool
        Whether model has been fitted
    """

    def __init__(self, lambda_param: float = 1.0, fit_intercept: bool = True):
        """
        Initialize Ridge Regression model.

        Parameters
        ----------
        lambda_param : float, default=1.0
            L2 regularization strength λ. Controls shrinkage:
            λ = 0 → OLS (no regularization)
            λ → ∞ → all coefficients → 0
            Typical values: 0.001 to 100 (dataset-dependent, tune via CV)
        fit_intercept : bool, default=True
            Whether to learn intercept term separately (recommended)
        """
        self.lambda_param = lambda_param
        self.fit_intercept = fit_intercept
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """
        Fit Ridge regression model using closed-form solution.

        Solves the ridge normal equation: β = (XᵀX + λI)⁻¹Xᵀy

        Algorithm:
        1. Validate inputs (shapes, dimensions)
        2. If fit_intercept=True: center y by subtracting mean
        3. Compute gram matrix G = XᵀX (d × d matrix)
        4. Add regularization: G_ridge = G + λI
        5. Compute right-hand side: h = Xᵀy
        6. Solve linear system: β = (G_ridge)⁻¹h using np.linalg.solve
        7. If fit_intercept was True: intercept = mean(y)
        8. Set is_fitted=True

        Numerical Considerations:
        - XᵀX + λI is always well-conditioned for λ > 0
        - Use np.linalg.solve for efficiency (LU decomposition)
        - Avoid explicit inversion (less stable)
        - Condition number improves as λ increases

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix, shape (n_samples, n_features)
        y : np.ndarray
            Training target vector, shape (n_samples,)

        Returns
        -------
        RidgeRegression
            Fitted model instance

        Raises
        ------
        ValueError
            If shapes incompatible or n_samples < 1
        """
        raise NotImplementedError(
            "Implement Ridge fitting: "
            "1. Validate inputs: X.shape[0] == y.shape[0], X.ndim == 2, y.ndim == 1. "
            "2. If fit_intercept=True: compute y_mean = mean(y), center y = y - y_mean. "
            "3. Compute gram matrix: G = Xᵀ @ X (shape d×d). "
            "4. Add L2 penalty: G_ridge = G + lambda_param * I. "
            "5. Compute RHS: h = Xᵀ @ y (shape d,). "
            "6. Solve system: β = solve(G_ridge, h) using np.linalg.solve. "
            "7. Store self.coefficients = β (shape d,). "
            "8. If fit_intercept: store self.intercept = y_mean. "
            "9. Set self.is_fitted = True. "
            "Return self for chaining."
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
            If model not fitted (is_fitted == False)
        ValueError
            If X.shape[1] != n_features (feature dimension mismatch)
        """
        raise NotImplementedError(
            "Implement prediction: "
            "1. Check is_fitted == True, raise RuntimeError if not. "
            "2. Verify X.shape[1] == len(self.coefficients), raise ValueError if not. "
            "3. Compute predictions: ŷ = X @ self.coefficients + self.intercept. "
            "4. Return 1D array of shape (n_samples,)."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² coefficient of determination.

        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_i - ŷ_i)² and SS_tot = Σ(y_i - ȳ)²

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        y : np.ndarray
            Target vector, shape (n_samples,)

        Returns
        -------
        float
            R² score, typically in range [-∞, 1.0]
            1.0 = perfect predictions
            0.0 = predicts mean
            negative = worse than mean
        """
        raise NotImplementedError(
            "Implement R² score: "
            "1. Get predictions ŷ = predict(X). "
            "2. Compute residuals: e = y - ŷ. "
            "3. Compute SS_res = sum(e²). "
            "4. Compute SS_tot = sum((y - mean(y))²). "
            "5. Handle edge case: if SS_tot == 0 (constant y), return 0. "
            "6. Return 1 - (SS_res / SS_tot)."
        )


class RidgeRegressionCV:
    """
    Ridge Regression with built-in cross-validation for λ tuning.

    Automatically finds optimal λ by cross-validation, eliminating manual
    hyperparameter tuning. Trains on all data with optimal λ.

    ALGORITHM
    ---------
    1. Define candidate λ values (log-spaced range)
    2. For each λ:
       a. Split data into k folds
       b. For each fold:
          - Train Ridge(λ) on k-1 folds
          - Evaluate on held-out fold (compute MSE or R²)
       c. Compute mean CV error across folds
    3. Select λ with minimum CV error
    4. Refit on all data with optimal λ
    5. Store optimal λ for future predictions

    ADVANTAGES
    - Automatic λ selection (data-driven)
    - Reduces overfitting compared to manual tuning
    - Provides cross-validation error estimate
    - Standard ML workflow

    DISADVANTAGES
    - More computationally expensive (k times slower than single Ridge)
    - k-fold setup requires parameter tuning
    - Assumes data is i.i.d. (fold stratification needed for imbalanced data)

    Example
    -------
    >>> X = np.random.randn(100, 20)
    >>> y = np.random.randn(100)
    >>> model = RidgeRegressionCV(lambdas=np.logspace(-3, 3, 7), cv=5)
    >>> model.fit(X, y)
    >>> print(f"Optimal λ = {model.lambda_opt_}")
    >>> y_pred = model.predict(X)

    Attributes
    ----------
    lambdas : np.ndarray
        Candidate λ values to evaluate
    cv_scores_ : np.ndarray
        Mean cross-validation errors for each λ, shape (len(lambdas),)
    lambda_opt_ : float
        Optimal λ selected by CV
    model_ : RidgeRegression
        Final model trained on all data with optimal λ
    """

    def __init__(
        self,
        lambdas: Optional[np.ndarray] = None,
        cv: int = 5,
        scoring: str = 'r2'
    ):
        """
        Initialize Ridge CV model.

        Parameters
        ----------
        lambdas : np.ndarray, optional
            Candidate λ values. If None, uses log-spaced range [10^-3, 10^3]
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='r2'
            Metric to optimize: 'mse' (minimize) or 'r2' (maximize)
        """
        self.lambdas = lambdas if lambdas is not None else np.logspace(-3, 3, 7)
        self.cv = cv
        self.scoring = scoring
        self.cv_scores_: Optional[np.ndarray] = None
        self.lambda_opt_: Optional[float] = None
        self.model_: Optional[RidgeRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressionCV':
        """
        Fit Ridge regression with automatic λ selection via cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)

        Returns
        -------
        RidgeRegressionCV
            Fitted model with optimal λ
        """
        raise NotImplementedError(
            "Implement Ridge CV: "
            "1. Validate inputs. "
            "2. Initialize cv_scores array of length len(lambdas). "
            "3. Create k-fold split (use np.array_split on indices). "
            "4. For each λ in self.lambdas: "
            "   a. Initialize scores list. "
            "   b. For each fold i in range(cv): "
            "      - Get test fold indices. "
            "      - Get train fold indices (all except test). "
            "      - Extract X_train, y_train, X_test, y_test. "
            "      - Fit Ridge(λ) on train fold. "
            "      - Evaluate on test fold using scoring metric. "
            "      - Append score to fold scores. "
            "   c. Compute mean CV score: cv_scores_[λ_idx] = mean(fold_scores). "
            "5. Select optimal λ: "
            "   - If scoring='r2': argmax(cv_scores_). "
            "   - If scoring='mse': argmin(cv_scores_). "
            "6. Refit on all data with optimal λ: "
            "   - self.model_ = RidgeRegression(lambda_opt).fit(X, y). "
            "7. Return self."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using model with optimal λ.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,)
        """
        raise NotImplementedError(
            "Implement prediction: delegate to self.model_.predict(X)."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² using optimal model."""
        raise NotImplementedError(
            "Implement scoring: delegate to self.model_.score(X, y)."
        )


class RidgeRegressionIterative:
    """
    Ridge Regression using iterative optimization (gradient descent).

    Trains Ridge regression by iteratively updating coefficients via gradient
    descent instead of closed-form solution. Useful for very large d or
    integration with other iterative algorithms.

    GRADIENT DESCENT FORMULATION
    ----------------------------
    Gradient of ridge loss:
        ∇L(β) = -(1/n)Xᵀ(y - Xβ) + λβ
               = -(1/n)Xᵀe + λβ

    Update rule:
        β_(t+1) = β_t - α * ∇L(β_t)
                = β_t - α * (-(1/n)Xᵀe + λβ_t)
                = β_t + (α/n)Xᵀe - αλβ_t
                = (1 - αλ)β_t + (α/n)Xᵀe

    Note: Factor (1 - αλ) provides implicit regularization per iteration

    CONVERGENCE
    -----------
    Ridge has unique minimum (strongly convex loss).
    Convergence guaranteed with sufficiently small learning rate.
    Rate depends on condition number of Hessian (2XᵀX + 2λI).

    Example
    -------
    >>> model = RidgeRegressionIterative(lambda_param=1.0, learning_rate=0.01)
    >>> model.fit(X_train, y_train, n_iterations=1000)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        lambda_param: float = 1.0,
        learning_rate: float = 0.01,
        fit_intercept: bool = True,
        verbose: bool = False
    ):
        """
        Initialize iterative Ridge regression.

        Parameters
        ----------
        lambda_param : float, default=1.0
            L2 regularization strength
        learning_rate : float, default=0.01
            Gradient descent step size
        fit_intercept : bool, default=True
            Whether to learn intercept term
        verbose : bool, default=False
            Whether to print loss during training
        """
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.losses: list = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 1000,
        tolerance: float = 1e-4
    ) -> 'RidgeRegressionIterative':
        """
        Fit Ridge regression using gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)
        n_iterations : int, default=1000
            Maximum gradient descent iterations
        tolerance : float, default=1e-4
            Stop if loss change < tolerance

        Returns
        -------
        RidgeRegressionIterative
            Fitted model
        """
        raise NotImplementedError(
            "Implement Ridge GD: "
            "1. Validate inputs and initialize β. "
            "2. If fit_intercept: compute and store y_mean, center y. "
            "3. For each iteration: "
            "   a. Compute predictions ŷ = X @ β. "
            "   b. Compute ridge loss: L = (1/2n)||y - ŷ||² + (λ/2)||β||². "
            "   c. Compute gradient ∇L = -(1/n)Xᵀ(y - ŷ) + λβ. "
            "   d. Update β ← β - α * ∇L. "
            "   e. Check convergence: |L_new - L_old| < tolerance. "
            "4. Store β, intercept, set is_fitted=True."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using learned coefficients."""
        raise NotImplementedError(
            "Implement prediction: ŷ = X @ β + intercept."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        raise NotImplementedError(
            "Implement R² calculation."
        )
