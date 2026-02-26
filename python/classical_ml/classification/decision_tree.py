"""
Decision Tree Classifier Module

Implementation Status: Stub - Educational Framework
Complexity: O(m*d*log(m)) for training, O(log(m)) for prediction
Prerequisites: Information theory, recursion, tree data structures

This module provides implementations of Decision Trees using the CART
(Classification and Regression Trees) algorithm for classification.

THEORY:
========
Decision trees recursively partition the feature space into increasingly pure
regions. Each split is chosen to maximize information gain or minimize impurity.
The algorithm builds a tree top-down in a greedy manner: at each node, it selects
the feature and threshold that best separates the classes.

Decision trees are highly interpretable (produces human-readable rules), handle
non-linear relationships naturally, and require minimal data preprocessing.
However, they tend to overfit without proper constraints like maximum depth.

The CART algorithm produces binary splits (each node has exactly 2 children),
making it simpler than alternatives like ID3 (which can produce multi-way splits).

MATHEMATICAL FOUNDATION:
=========================
Impurity Measures:

1. GINI IMPURITY:
   Gini(S) = 1 - Σ_k p_k²

   Where p_k is proportion of class k in set S.
   Measures probability of misclassifying random sample if labeled randomly
   by class distribution in set.

   Range: [0, 1]
   - 0: Pure node (single class)
   - 0.5: Maximum impurity (two classes equally mixed)

2. ENTROPY (Information):
   H(S) = -Σ_k p_k * log₂(p_k)

   Measures information content or uncertainty.
   In bits: 1 bit of information for 2 equally likely outcomes.

   Range: [0, log₂(K)] where K is number of classes
   - 0: Pure node
   - log₂(K): Maximum entropy

   Cross-entropy for probabilistic classification:
   CE = -Σ_k p_k * log(p_k)  [using natural log]

Information Gain:
  IG(S, A) = I(S) - Σ (|S_v|/|S|) * I(S_v)

  Where:
  - A: attribute to split on
  - S_v: subset of S where A takes value v
  - I: impurity function (Gini or Entropy)

  IG measures reduction in impurity after split. Higher gain = better split.

Gain Ratio (corrects bias toward high-cardinality attributes):
  GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)

  Where:
  SplitInfo(S, A) = -Σ (|S_v|/|S|) * log₂(|S_v|/|S|)

ALGORITHM (CART with Gini):
=============================
1. Start with root node containing all training samples
2. For each node:
   a. If node is pure (single class) or stopping criterion met:
      - Create leaf node with class label (majority class)
   b. Otherwise:
      - For each feature j and each possible split value t:
        * Split into left (x_j ≤ t) and right (x_j > t)
        * Compute weighted Gini after split:
          Gini_split = (n_left/n) * Gini(left) + (n_right/n) * Gini(right)
      - Choose feature j and threshold t that minimize Gini_split
   c. Recursively apply to left and right subsets

Split Selection for Numerical Features:
========================================
For feature x_j with values [x_1, x_2, ..., x_m]:

Candidate thresholds:
  1. Midpoints between sorted values: (x_i + x_{i+1}) / 2
  2. Or try all unique values

For each threshold t:
  - Left: samples where x_j ≤ t
  - Right: samples where x_j > t
  - Compute impurity reduction

This is O(m log m) per feature (sorting) and O(m) for evaluation.

STOPPING CRITERIA & PRUNING:
============================
Early Stopping (prevent overfitting):
  - max_depth: maximum tree depth
  - min_samples_split: minimum samples required to split a node
  - min_samples_leaf: minimum samples required at leaf nodes
  - max_leaf_nodes: maximum number of leaf nodes

Cost Complexity Pruning (post-pruning):
  - Grow full tree, then remove nodes that don't improve validation score
  - More effective than early stopping but computationally expensive

NUMERICAL STABILITY GOTCHAS:
=============================
1. DEEP TREES WITH SMALL DATASETS:
   Problem: Overfitting - perfect classification on training data
   Solution: Use max_depth, min_samples_split, or min_samples_leaf

2. CLASS IMBALANCE:
   Problem: Minority class ignored, pure leaves contain majority class
   Solution: Use class weights to make minority class splits more valuable

3. FEATURE SCALING NOT NEEDED:
   Advantage: Trees are scale-invariant (use thresholds, not distances)
   Disadvantage: If using other algorithms later, scale anyway

4. MISSING SPLITS FOR CATEGORICAL FEATURES:
   Problem: If feature has C categories, need C-1 levels in tree
   Solution: Encode categorical features properly or use one-hot encoding

5. NUMERICAL PRECISION IN THRESHOLD SELECTION:
   Problem: With floating point, x_j ≤ t and x_j > t may overlap
   Solution: Use robust comparison (avoid exact equality)

ADVANTAGES vs DISADVANTAGES:
=============================
Advantages:
  + Easy to understand and interpret
  + Requires minimal data preprocessing (no scaling needed)
  + Handles both numerical and categorical features
  + Can capture non-linear relationships
  + Fast prediction O(log m)
  + Feature importance measures

Disadvantages:
  - Prone to overfitting without proper constraints
  - Greedy algorithm may not find global optimum
  - Unstable (small data changes → large tree changes)
  - Biased toward high-cardinality features without normalization
  - Requires all paths through tree for instance to classify

RANDOM FORESTS & BOOSTING:
============================
To mitigate overfitting:
  - Ensemble: combine many trees
  - Random Forests: parallel trees with bootstrap + random features
  - Gradient Boosting: sequential trees, each corrects previous errors
  - Extremely Randomized Trees: use random thresholds instead of optimal

REFERENCES:
============
1. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984).
   Classification and Regression Trees. Chapman and Hall.
   https://www.routledge.com/Classification-and-Regression-Trees/Breiman/p/book/9780412048418

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
   Statistical Learning. https://web.stanford.edu/~hastie/ElemStatLearn/

3. Quinlan, J. R. (1993). C4.5: Programs for Machine Learning.
   Morgan Kaufmann. (Classic reference for ID3/C4.5 algorithm)

4. Decision Trees - Scikit-learn Documentation
   https://scikit-learn.org/stable/modules/tree.html

5. Information Theory & Information Gain
   https://en.wikipedia.org/wiki/Information_gain_(decision_tree)

6. Cost Complexity Pruning
   https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
"""

from typing import Optional, Tuple, List
import numpy as np


class TreeNode:
    """
    Node in decision tree.

    Parameters
    ----------
    feature : int, optional
        Feature index to split on (None for leaf nodes)

    threshold : float, optional
        Threshold value for split (None for leaf nodes)

    left : TreeNode, optional
        Left child node

    right : TreeNode, optional
        Right child node

    class_label : int, optional
        Predicted class for leaf node

    samples : int
        Number of training samples in this node

    impurity : float
        Impurity (Gini or Entropy) of this node
    """

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['TreeNode'] = None,
        right: Optional['TreeNode'] = None,
        class_label: Optional[int] = None,
        samples: int = 0,
        impurity: float = 0.0,
    ):
        """Initialize tree node."""
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_label = class_label
        self.samples = samples
        self.impurity = impurity

    def is_leaf(self) -> bool:
        """Check if node is a leaf (makes predictions)."""
        return self.class_label is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using CART Algorithm.

    Builds binary decision tree for classification by recursively selecting
    splits that minimize Gini impurity. Supports early stopping via depth
    and sample constraints to prevent overfitting.

    Parameters
    ----------
    criterion : str, default='gini'
        Function to measure split quality: 'gini' or 'entropy'

    max_depth : int, optional
        Maximum tree depth. None = unlimited (may lead to overfitting)

    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
        Larger value → simpler tree, less overfitting

    min_samples_leaf : int, default=1
        Minimum number of samples required at leaf node.
        Larger value → smoother decision boundary

    max_leaf_nodes : int, optional
        Maximum number of leaf nodes. Grows by "best" splits.
        If None: unlimited

    random_state : int, optional
        Random seed for reproducibility (in case of ties)

    Attributes
    ----------
    tree : TreeNode
        Root node of fitted decision tree

    classes : np.ndarray
        Unique class labels seen during training

    n_features : int
        Number of features seen during training

    depth : int
        Actual depth of fitted tree

    n_leaves : int
        Number of leaves in fitted tree

    feature_importances : np.ndarray, shape (n_features,)
        Relative importance of each feature

    Examples
    --------
    >>> from classical_ml.classification import DecisionTreeClassifier
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> tree = DecisionTreeClassifier(max_depth=5)
    >>> tree.fit(X, y)
    >>> predictions = tree.predict(X)
    """

    def __init__(
        self,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_leaf_nodes: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize Decision Tree Classifier."""
        if criterion not in ['gini', 'entropy']:
            raise ValueError(f"Unknown criterion: {criterion}")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_leaf_nodes is not None and max_leaf_nodes < 1:
            raise ValueError("max_leaf_nodes must be >= 1")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

        self.tree: Optional[TreeNode] = None
        self.classes: Optional[np.ndarray] = None
        self.n_features: Optional[int] = None
        self.depth: int = 0
        self.n_leaves: int = 0
        self.feature_importances: Optional[np.ndarray] = None

    def _gini(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity.

        Gini(S) = 1 - Σ_k p_k²

        Parameters
        ----------
        y : np.ndarray
            Class labels

        Returns
        -------
        float
            Gini impurity in range [0, 1]
        """
        raise NotImplementedError(
            "Compute Gini impurity: "
            "1. Get unique classes and their proportions: p_k = count(y==k) / len(y) "
            "2. Return 1 - sum(p_k²) "
            "Hint: Use np.unique(y, return_counts=True)"
        )

    def _entropy(self, y: np.ndarray) -> float:
        """
        Compute Shannon entropy.

        H(S) = -Σ_k p_k * log₂(p_k)

        Parameters
        ----------
        y : np.ndarray
            Class labels

        Returns
        -------
        float
            Entropy in bits (log base 2)

        Notes
        -----
        Handle edge case: 0 * log(0) = 0 (using L'Hôpital's rule)
        Use np.where or similar to avoid log(0)
        """
        raise NotImplementedError(
            "Compute Shannon entropy: "
            "1. Get proportions p_k "
            "2. Return -Σ p_k * log₂(p_k), handling p_k=0 case "
            "Hint: Use np.where(p > 0, -p * np.log2(p), 0)"
        )

    def _impurity(self, y: np.ndarray) -> float:
        """
        Compute impurity using selected criterion.

        Parameters
        ----------
        y : np.ndarray
            Class labels

        Returns
        -------
        float
            Impurity value
        """
        raise NotImplementedError(
            "Implement _impurity: "
            "If criterion == 'gini': return self._gini(y) "
            "Otherwise: return self._entropy(y)"
        )

    def _information_gain(
        self, parent: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> float:
        """
        Compute information gain from split.

        IG = I(parent) - (|left|/|parent|) * I(left) - (|right|/|parent|) * I(right)

        Parameters
        ----------
        parent : np.ndarray
            Class labels before split

        left : np.ndarray
            Class labels in left child

        right : np.ndarray
            Class labels in right child

        Returns
        -------
        float
            Information gain (higher is better)
        """
        raise NotImplementedError(
            "Compute information gain: "
            "1. Compute parent impurity: I_parent "
            "2. Compute weighted child impurity: I_children = (n_left/n_parent)*I_left + ... "
            "3. Return I_parent - I_children"
        )

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """
        Find best feature and threshold to split on.

        For each feature:
          - For each unique value or midpoint:
            - Compute information gain
            - Track best split

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Feature matrix

        y : np.ndarray, shape (n,)
            Class labels

        Returns
        -------
        best_feature : int or None
            Feature index to split on (None if no valid split)

        best_threshold : float or None
            Threshold value for split

        best_gain : float or None
            Information gain from best split
        """
        raise NotImplementedError(
            "Implement _find_best_split: "
            "1. For each feature j in range(n_features): "
            "   a. Get sorted unique values or midpoints as candidates "
            "   b. For each threshold t: "
            "      - Split: left = y[X[:, j] <= t], right = y[X[:, j] > t] "
            "      - Compute gain using _information_gain "
            "   c. Track best_feature, best_threshold, best_gain "
            "2. Return (best_feature, best_threshold, best_gain)"
        )

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> TreeNode:
        """
        Recursively build decision tree.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Feature matrix for this node

        y : np.ndarray, shape (n,)
            Class labels for this node

        depth : int
            Current depth in tree

        Returns
        -------
        TreeNode
            Subtree rooted at current node

        Notes
        -----
        Stopping criteria:
        1. All samples have same class (pure node)
        2. n_samples < min_samples_split
        3. depth >= max_depth
        4. max_leaf_nodes reached
        5. No valid split found (all features give same split)

        When stopping, create leaf node with majority class.
        """
        raise NotImplementedError(
            "Implement _build_tree recursively: "
            "1. Check stopping criteria (pure, depth, min_samples_split, etc.) "
            "2. Find best split "
            "3. If no valid split or stopping: create leaf with majority class "
            "4. Otherwise: split and recursively build left/right subtrees "
            "5. Return TreeNode with all information "
            "Hint: Track depth and number of leaves"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Fit decision tree to training data.

        Parameters
        ----------
        X : np.ndarray, shape (m, d)
            Training feature matrix

        y : np.ndarray, shape (m,)
            Training labels

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator

        Notes
        -----
        Builds tree top-down using greedy CART algorithm.
        Computes feature importances based on total information gain.
        """
        raise NotImplementedError(
            "Implement fit method: "
            "1. Store self.classes = np.unique(y) "
            "2. Store self.n_features = X.shape[1] "
            "3. Build tree using self._build_tree(X, y) "
            "4. Compute tree depth and number of leaves "
            "5. Compute feature importances "
            "6. Return self"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        For each sample, traverse tree to leaf and return class distribution
        of training samples at that leaf.

        Parameters
        ----------
        X : np.ndarray, shape (m, d)
            Feature matrix

        Returns
        -------
        proba : np.ndarray, shape (m, n_classes)
            Class probabilities
        """
        raise NotImplementedError(
            "Implement predict_proba: "
            "1. For each sample: traverse tree to leaf "
            "2. Get class distribution at leaf "
            "3. Return probabilities as (m, n_classes) matrix"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (m, d)
            Feature matrix

        Returns
        -------
        predictions : np.ndarray, shape (m,)
            Predicted class labels
        """
        raise NotImplementedError(
            "Implement predict: "
            "1. For each sample: traverse tree to leaf "
            "2. Return class_label from leaf node"
        )

    def _traverse_tree(self, x: np.ndarray) -> TreeNode:
        """
        Traverse tree to find leaf for single sample.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Single sample feature vector

        Returns
        -------
        TreeNode
            Leaf node reached
        """
        raise NotImplementedError(
            "Implement _traverse_tree: "
            "Starting from self.tree root: "
            "While node is not a leaf: "
            "  if x[node.feature] <= node.threshold: "
            "    go to node.left "
            "  else: "
            "    go to node.right "
            "Return leaf node"
        )

    def get_feature_importances(self) -> np.ndarray:
        """
        Compute feature importances based on information gain.

        Importance = total information gain from splits using feature / total gain

        Returns
        -------
        importances : np.ndarray, shape (n_features,)
            Relative importance of each feature
        """
        raise NotImplementedError(
            "Compute feature importances: "
            "1. Traverse tree collecting information gain at each split "
            "2. Accumulate gains per feature "
            "3. Normalize by total gain "
            "4. Return importance vector"
        )
