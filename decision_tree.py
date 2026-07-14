"""
Decision Tree & Random Forest
==============================
A decision tree partitions the feature space by recursively selecting the
feature and threshold that best separates the classes (or reduces variance
for regression).  Random Forest builds many trees on bootstrap samples and
averages their predictions to reduce over-fitting.

This script demonstrates:
1. Decision tree classification from scratch (ID3-style, information gain).
2. Decision tree and random forest classifiers with scikit-learn on the
   Wine dataset.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# 1. From-scratch decision tree (binary splits, Gini impurity)
# ---------------------------------------------------------------------------

def _gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - float(np.sum(probs ** 2))


class _Node:
    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: "_Node | None" = None,
        right: "_Node | None" = None,
        value: int | None = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # set for leaf nodes

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeScratch:
    """CART-style decision tree classifier using Gini impurity."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: _Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeScratch":
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        # Stopping criteria
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return _Node(value=int(np.bincount(y).argmax()))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return _Node(value=int(np.bincount(y).argmax()))

        left_mask = X[:, feature] <= threshold
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int | None, float | None]:
        best_gain = -1.0
        best_feature, best_threshold = None, None
        parent_gini = _gini(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                if left_mask.sum() == 0 or (~left_mask).sum() == 0:
                    continue
                gain = parent_gini - (
                    left_mask.sum() / len(y) * _gini(y[left_mask])
                    + (~left_mask).sum() / len(y) * _gini(y[~left_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _traverse(self, x: np.ndarray, node: _Node) -> int:
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root) for x in X])


# ---------------------------------------------------------------------------
# 2. scikit-learn Decision Tree & Random Forest (Wine dataset)
# ---------------------------------------------------------------------------

def sklearn_demo() -> None:
    """Train and compare Decision Tree and Random Forest on the Wine dataset."""
    dataset = load_wine()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))

    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    print("=== scikit-learn Decision Tree & Random Forest (Wine dataset) ===")
    print(f"  Decision Tree accuracy : {dt_acc:.4f}")
    print(f"  Random Forest accuracy : {rf_acc:.4f}")
    print()
    print("Random Forest classification report:")
    print(
        classification_report(
            y_test, rf.predict(X_test), target_names=dataset.target_names
        )
    )


def scratch_demo() -> None:
    """Train and evaluate the from-scratch decision tree."""
    dataset = load_wine()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalise to help the simple splitter
    mean, std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    model = DecisionTreeScratch(max_depth=5)
    model.fit(X_train_n, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_n))

    print("=== From-scratch Decision Tree (Wine dataset) ===")
    print(f"  Accuracy: {acc:.4f}")


if __name__ == "__main__":
    scratch_demo()
    print()
    sklearn_demo()
