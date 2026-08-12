"""
Linear Regression
=================
Linear regression models the relationship between a continuous target variable
and one or more input features by fitting a straight line (or hyperplane).

This script demonstrates:
1. Linear regression implemented from scratch using NumPy (gradient descent).
2. Linear regression using scikit-learn on a real dataset.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. From-scratch implementation
# ---------------------------------------------------------------------------

class LinearRegressionScratch:
    """Ordinary least-squares linear regression via gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


# ---------------------------------------------------------------------------
# 2. scikit-learn implementation on the Diabetes dataset
# ---------------------------------------------------------------------------

def sklearn_demo() -> None:
    """Train and evaluate a scikit-learn linear regression model."""
    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== scikit-learn Linear Regression (Diabetes dataset) ===")
    print(f"  MSE : {mse:.2f}")
    print(f"  R²  : {r2:.4f}")


def scratch_demo() -> None:
    """Train and evaluate the from-scratch linear regression model."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 1))
    y = 3.0 * X.ravel() + 5.0 + rng.standard_normal(200) * 0.5

    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== From-scratch Linear Regression (synthetic data) ===")
    print(f"  Learned weight : {model.weights[0]:.4f}  (true: 3.0)")
    print(f"  Learned bias   : {model.bias:.4f}  (true: 5.0)")
    print(f"  MSE            : {mse:.4f}")
    print(f"  R²             : {r2:.4f}")


if __name__ == "__main__":
    scratch_demo()
    print()
    sklearn_demo()
