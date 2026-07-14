"""
Neural Network (Multi-Layer Perceptron)
========================================
A neural network learns hierarchical feature representations by stacking
layers of weighted connections followed by non-linear activation functions.
Training uses back-propagation and gradient descent to minimise a loss
function.

This script demonstrates:
1. A single-hidden-layer neural network implemented from scratch with NumPy
   (ReLU hidden activation, sigmoid output, binary cross-entropy loss).
2. A scikit-learn MLPClassifier on the MNIST digits dataset.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. From-scratch neural network (1 hidden layer, binary classification)
# ---------------------------------------------------------------------------

def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class NeuralNetworkScratch:
    """Single-hidden-layer neural network for binary classification."""

    def __init__(
        self,
        hidden_size: int = 16,
        learning_rate: float = 0.01,
        n_iterations: int = 2000,
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkScratch":
        rng = np.random.default_rng(42)
        n_samples, n_features = X.shape

        # He initialisation
        self.W1 = rng.standard_normal((n_features, self.hidden_size)) * np.sqrt(
            2.0 / n_features
        )
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.standard_normal((self.hidden_size, 1)) * np.sqrt(
            2.0 / self.hidden_size
        )
        self.b2 = np.zeros(1)

        y_col = y.reshape(-1, 1).astype(float)

        for iteration in range(self.n_iterations):
            # Forward pass
            Z1 = X @ self.W1 + self.b1
            A1 = _relu(Z1)
            Z2 = A1 @ self.W2 + self.b2
            A2 = _sigmoid(Z2)

            # Backward pass
            dZ2 = A2 - y_col                              # (n, 1)
            dW2 = (A1.T @ dZ2) / n_samples
            db2 = dZ2.mean(axis=0)

            dA1 = dZ2 @ self.W2.T                         # (n, hidden)
            dZ1 = dA1 * _relu_deriv(Z1)
            dW1 = (X.T @ dZ1) / n_samples
            db1 = dZ1.mean(axis=0)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        A1 = _relu(X @ self.W1 + self.b1)
        return _sigmoid(A1 @ self.W2 + self.b2).ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 2. scikit-learn MLP on the MNIST digits dataset
# ---------------------------------------------------------------------------

def sklearn_demo() -> None:
    """Train and evaluate a scikit-learn MLP on the digits dataset."""
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== scikit-learn MLP (Digits dataset, 10 classes) ===")
    print(f"  Accuracy: {acc:.4f}")
    print()
    print(classification_report(y_test, y_pred))


def scratch_demo() -> None:
    """Train and evaluate the from-scratch neural network (binary task)."""
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = NeuralNetworkScratch(hidden_size=32, learning_rate=0.01, n_iterations=2000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    print("=== From-scratch Neural Network (Breast Cancer, binary) ===")
    print(f"  Accuracy: {acc:.4f}")


if __name__ == "__main__":
    scratch_demo()
    print()
    sklearn_demo()
