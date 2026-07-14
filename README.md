# Machine Learning Study

A collection of machine learning algorithms implemented **from scratch** (using NumPy) alongside equivalent **scikit-learn** examples.  
Each script is self-contained and can be run independently.

---

## Algorithms covered

| File | Algorithm | Dataset used |
|------|-----------|--------------|
| `linear_regression.py` | Linear Regression | Synthetic data, Diabetes |
| `logistic_regression.py` | Logistic Regression | Synthetic data, Iris |
| `decision_tree.py` | Decision Tree & Random Forest | Wine |
| `knn.py` | K-Nearest Neighbors | Breast Cancer |
| `neural_network.py` | Multi-Layer Perceptron | Breast Cancer, Digits |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the examples

```bash
python linear_regression.py
python logistic_regression.py
python decision_tree.py
python knn.py
python neural_network.py
```

---

## Key concepts

- **Linear Regression** – fit a line to continuous data using gradient descent.
- **Logistic Regression** – binary/multi-class classification via the sigmoid function.
- **Decision Tree / Random Forest** – recursive feature-based splitting; ensembling reduces variance.
- **K-Nearest Neighbors** – classify by majority vote among the K closest training points.
- **Neural Network (MLP)** – stacked layers with ReLU activations trained by back-propagation.
