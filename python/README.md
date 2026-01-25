# Python Bindings for ML Library

Complete Python bindings for the C++ ML library using pybind11.

## Features

### Regression Models
- **Linear Regression** - Linear regression with gradient descent and regularization
- Supports MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError loss functions
- L1/L2 regularization

### Classification Models
- **Logistic Regression** - Binary classification with sigmoid activation
- **Support Vector Machines (SVM)** - Multiple kernels (Linear, RBF, Polynomial, Sigmoid)
- **K-Nearest Neighbors (KNN)** - Instance-based learning
- **Decision Trees** - Tree-based classification with Gini/Entropy impurity
- **Random Forests** - Ensemble of decision trees

### Clustering
- **K-Means Clustering** - Unsupervised clustering algorithm

### Core Components
- **Loss Functions**: MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, BinaryCrossEntropy
- **Optimizers**: Batch, Stochastic, Mini-batch gradient descent
- **Regularizers**: L1 (Lasso), L2 (Ridge), None
- **Matrix**: NumPy-compatible matrix operations
- **Metrics**: Accuracy, Precision, Recall, F1, MSE, MAE, RMSE, R²

## Installation

```bash
# Install dependencies
pip install pybind11 numpy

# Navigate to python directory
cd python

# Build
mkdir build && cd build
cmake ..
cmake --build .

# Run examples
cd ..
PYTHONPATH=build python example.py
```

## Quick Start Examples

### Linear Regression

```python
import numpy as np
import ml_lib

# Generate data
X_train = np.random.randn(100, 3)
y_train = (2.5 * X_train[:, 0] + 1.5 * X_train[:, 1]).reshape(-1, 1)

# Create model
loss = ml_lib.MeanSquaredErrorLoss()
optimizer = ml_lib.BatchOptimizer(learning_rate=0.01)
regularizer = ml_lib.L2Regularizer(lambda_=0.01)

model = ml_lib.LinearRegression(
    input_dim=3,
    loss=loss,
    optimizer=optimizer,
    regularizer=regularizer
)

# Forward pass
predictions = model.forward(X_train)
```

### Logistic Regression

```python
import numpy as np
import ml_lib

# Generate binary classification data
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(float).reshape(-1, 1)

# Create model
loss = ml_lib.BinaryCrossEntropyLoss()
optimizer = ml_lib.BatchOptimizer(learning_rate=0.01)
regularizer = ml_lib.L2Regularizer(lambda_=0.01)

model = ml_lib.LogisticRegression(
    input_dim=2,
    loss=loss,
    optimizer=optimizer,
    regularizer=regularizer
)

# Get probabilities
probabilities = model.predict_proba(X_train)

# Get class predictions
predictions = model.predict(X_train)
```

### Support Vector Machine

```python
import numpy as np
import ml_lib

# Generate data (SVM expects {-1, 1} labels)
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(float).reshape(-1, 1)
y_train = 2 * y_train - 1  # Convert {0,1} to {-1,1}

# Create SVM with RBF kernel
svm = ml_lib.SupportVectorMachine(
    C=1.0,
    gamma=0.1,
    kernel=ml_lib.Kernel.RBF,
    max_iter=1000
)

svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
```

### K-Nearest Neighbors

```python
import numpy as np
import ml_lib

# Training data
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(float).reshape(-1, 1)

# Create and train
knn = ml_lib.KNearestNeighbors(k=5)
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)

# Evaluate
accuracy = ml_lib.metrics.accuracy(y_test, predictions)
```

### Decision Tree & Random Forest

```python
import numpy as np
import ml_lib

# Decision Tree
tree = ml_lib.DecisionTree()
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

# Random Forest
rf = ml_lib.RandomForest()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

### K-Means Clustering

```python
import numpy as np
import ml_lib

# Generate data
X = np.random.randn(300, 2)

# Cluster
kmeans = ml_lib.KMeansClustering(k=3, max_iter=100)
kmeans.fit(X, np.zeros((X.shape[0], 1)))  # y is required but not used
cluster_labels = kmeans.predict(X)
```

## Complete API Reference

### Models

#### `LinearRegression(input_dim, loss, optimizer, regularizer)`
**Parameters:**
- `input_dim` (int): Number of input features
- `loss` (LossFunction): Loss function instance (MeanSquaredErrorLoss, MeanAbsoluteErrorLoss, RootMeanSquaredErrorLoss)
- `optimizer` (Optimizer): Optimizer instance
- `regularizer` (Regularizer): Regularizer instance

**Methods:**
- `forward(X)`: Forward pass, returns predictions

---

#### `LogisticRegression(input_dim, loss, optimizer, regularizer)`
**Parameters:**
- `input_dim` (int): Number of input features
- `loss` (LossFunction): Loss function (BinaryCrossEntropyLoss recommended)
- `optimizer` (Optimizer): Optimizer instance
- `regularizer` (Regularizer): Regularizer instance

**Methods:**
- `predict(X)`: Returns class labels (0 or 1)
- `predict_proba(X)`: Returns class probabilities
- `forward(X)`: Forward pass

---

#### `SupportVectorMachine(C=1.0, gamma=0.1, kernel=LINEAR, degree=3, tolerance=1e-3, max_iter=1000, coef0=0.0)`
**Parameters:**
- `C` (float): Regularization parameter
- `gamma` (float): Kernel coefficient
- `kernel` (Kernel): KERNEL.LINEAR, KERNEL.RBF, KERNEL.POLYNOMIAL, KERNEL.SIGMOID
- `degree` (int): Degree for polynomial kernel
- `tolerance` (float): Tolerance for stopping criterion
- `max_iter` (int): Maximum iterations
- `coef0` (float): Independent term in kernel function

**Methods:**
- `fit(X, y)`: Train the model (expects y in {-1, 1})
- `predict(X)`: Make predictions (returns {-1, 1})

---

#### `KNearestNeighbors(k=3)`
**Parameters:**
- `k` (int): Number of neighbors to consider

**Methods:**
- `fit(X, y)`: Store training data
- `predict(X)`: Classify based on k nearest neighbors

---

#### `DecisionTree()`
Decision tree classifier using Gini impurity or Entropy.

**Methods:**
- `fit(X, y)`: Build decision tree
- `predict(X)`: Make predictions

---

#### `RandomForest()`
Ensemble of decision trees with bootstrap aggregation.

**Methods:**
- `fit(X, y)`: Build forest
- `predict(X)`: Make predictions (majority vote)

---

#### `KMeansClustering(k=3, threshold=3, max_iter=100)`
**Parameters:**
- `k` (int): Number of clusters
- `threshold` (int): Convergence threshold
- `max_iter` (int): Maximum iterations

**Methods:**
- `fit(X, y)`: Find cluster centroids (y parameter required but not used)
- `predict(X)`: Assign samples to nearest cluster

### Loss Functions

```python
ml_lib.MeanSquaredErrorLoss()       # Mean Squared Error
ml_lib.MeanAbsoluteErrorLoss()      # Mean Absolute Error
ml_lib.RootMeanSquaredErrorLoss()   # Root Mean Squared Error
ml_lib.BinaryCrossEntropyLoss()     # Binary Cross Entropy
```

### Optimizers

```python
ml_lib.BatchOptimizer(learning_rate=0.01)        # Batch gradient descent
ml_lib.StochasticOptimizer(learning_rate=0.01)   # Stochastic GD
ml_lib.MiniBatchOptimizer(learning_rate=0.01)    # Mini-batch GD
```

**Methods:**
- `set_learning_rate(lr)`: Update learning rate
- `get_learning_rate()`: Get current learning rate

### Regularizers

```python
ml_lib.L1Regularizer(lambda_=0.01)   # L1 (Lasso) regularization
ml_lib.L2Regularizer(lambda_=0.01)   # L2 (Ridge) regularization
```

### Metrics

All metrics available in `ml_lib.metrics`:

```python
# Classification metrics
ml_lib.metrics.accuracy(y_true, y_pred)
ml_lib.metrics.precision(y_true, y_pred)
ml_lib.metrics.recall(y_true, y_pred)
ml_lib.metrics.f1_score(y_true, y_pred)

# Regression metrics
ml_lib.metrics.mse(y_true, y_pred)
ml_lib.metrics.mae(y_true, y_pred)
ml_lib.metrics.rmse(y_true, y_pred)
ml_lib.metrics.r2_score(y_true, y_pred)
```

### Enums

```python
# Loss types
ml_lib.LossType.MEAN_ABSOLUTE_ERROR
ml_lib.LossType.MEAN_SQUARED_ERROR
ml_lib.LossType.ROOT_MEAN_SQUARED_ERROR
ml_lib.LossType.BINARY_CROSS_ENTROPY

# Optimizer types
ml_lib.OptimizerType.BATCH
ml_lib.OptimizerType.STOCHASTIC
ml_lib.OptimizerType.MINI_BATCH

# Regularizer types
ml_lib.RegularizerType.NONE
ml_lib.RegularizerType.L1
ml_lib.RegularizerType.L2

# SVM Kernels
ml_lib.Kernel.LINEAR
ml_lib.Kernel.POLYNOMIAL
ml_lib.Kernel.RBF
ml_lib.Kernel.SIGMOID

# Decision Tree Impurity
ml_lib.Impurity.GINI
ml_lib.Impurity.ENTROPY

# K-Means Initialization
ml_lib.ClusterInit.RANDOM
ml_lib.ClusterInit.SMART
```

## Important Notes

### Data Format
- All input arrays should be NumPy arrays with dtype `float64`
- Target arrays (y) must be 2D column vectors: shape `(n_samples, 1)`
  ```python
  y = y.reshape(-1, 1)  # Convert 1D to 2D
  ```

### SVM Labels
- SVM expects labels in {-1, 1}, not {0, 1}
  ```python
  y_svm = 2 * y - 1  # Convert {0,1} to {-1,1}
  ```

### Memory Management
- Loss, Optimizer, and Regularizer objects are managed by the model
- Use `py::keep_alive` (handled automatically in bindings)
- Don't delete these objects while model is in use

### Gradient Models
- Linear and Logistic Regression use gradient-based training
- They require manual training loops (forward, backward, update)
- Or use the C++ fit method implementation if available

## Troubleshooting

### Import Error
```python
import sys
sys.path.insert(0, '/path/to/build/directory')
import ml_lib
```

### pybind11 Not Found
```bash
pip install pybind11

# Or via package manager:
# Ubuntu/Debian: sudo apt install pybind11-dev
# macOS: brew install pybind11
```

### CMake Not Finding pybind11
```bash
pip install "pybind11[global]"
```

## Examples

See [example.py](example.py) for complete working examples of all models.

```bash
cd python
PYTHONPATH=build python example.py
```

## License

This project is available for educational purposes.
