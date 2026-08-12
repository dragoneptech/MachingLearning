"""
Logistic Regression
===================
Logistic regression is a classification algorithm that predicts the
probability of a binary (or multi-class) outcome using the sigmoid function.

This script demonstrates:
1. Binary logistic regression implemented from scratch using NumPy.
2. Multi-class logistic regression using scikit-learn on the Iris dataset.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. From-scratch binary logistic regression (sigmoid + gradient descent)
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionScratch:
    """Binary logistic regression via gradient descent."""

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            linear = X @ self.weights + self.bias
            y_pred = _sigmoid(linear)

            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 2. scikit-learn multi-class logistic regression (Iris dataset)
# ---------------------------------------------------------------------------

def sklearn_demo() -> None:
    """Train and evaluate a scikit-learn logistic regression model."""
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== scikit-learn Logistic Regression (Iris dataset) ===")
    print(f"  Accuracy: {acc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=load_iris().target_names))


def scratch_demo() -> None:
    """Train and evaluate the from-scratch binary logistic regression model."""
    rng = np.random.default_rng(42)

    # Create a linearly separable binary dataset
    X0 = rng.standard_normal((100, 2)) + np.array([-2, -2])
    X1 = rng.standard_normal((100, 2)) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== From-scratch Logistic Regression (binary synthetic data) ===")
    print(f"  Accuracy: {acc:.4f}")


if __name__ == "__main__":
    scratch_demo()
    print()
    sklearn_demo()
