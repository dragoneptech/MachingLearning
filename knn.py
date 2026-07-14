"""
K-Nearest Neighbors (KNN)
=========================
KNN is a non-parametric algorithm that classifies a sample by majority vote
among its K closest training examples (using Euclidean distance by default).

This script demonstrates:
1. KNN classifier implemented from scratch using NumPy.
2. KNN classifier using scikit-learn on the Breast Cancer dataset, with a
   simple search for the best value of K.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. From-scratch KNN classifier
# ---------------------------------------------------------------------------

class KNNScratch:
    """K-Nearest Neighbors classifier."""

    def __init__(self, k: int = 5):
        self.k = k
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNScratch":
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> int:
        distances = np.linalg.norm(self._X_train - x, axis=1)
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self._y_train[k_indices]
        return int(np.bincount(k_labels).argmax())


# ---------------------------------------------------------------------------
# 2. scikit-learn KNN with K selection (Breast Cancer dataset)
# ---------------------------------------------------------------------------

def sklearn_demo() -> None:
    """Search for the best K and report results on the test set."""
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_k, best_acc = 1, 0.0
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        if acc > best_acc:
            best_acc, best_k = acc, k

    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    print("=== scikit-learn KNN (Breast Cancer dataset) ===")
    print(f"  Best K  : {best_k}")
    print(f"  Accuracy: {best_acc:.4f}")
    print()
    print(
        classification_report(
            y_test, y_pred, target_names=dataset.target_names
        )
    )


def scratch_demo() -> None:
    """Train and evaluate the from-scratch KNN classifier."""
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNNScratch(k=7)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== From-scratch KNN (Breast Cancer dataset, K=7) ===")
    print(f"  Accuracy: {acc:.4f}")


if __name__ == "__main__":
    scratch_demo()
    print()
    sklearn_demo()
