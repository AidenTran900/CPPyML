# ML Models

A C++ machine learning library built from the ground up. Implementing various ML algorithms and models using fundamental linear algebra and optimization techniques.

## Features

### Models
- **Linear Regression** with gradient descent optimization
- **Logistic Regression** for binary classification
- **K-Nearest Neighbors (KNN)** with Euclidean and Manhattan distance metrics
- **Support Vector Machines (SVM)** with multiple kernels (Linear, Polynomial, RBF, Sigmoid)
- **Decision Trees** with Gini and Entropy impurity measures
- **Random Forests** with bootstrap aggregation
- **K-Means Clustering** for unsupervised learning
- **Neural Networks** with backpropagation and configurable layers
- **Perceptron** for binary classification

### Core Components
- **Matrix Operations**: Addition, multiplication, transpose, inverse, Hadamard product, determinant
- **Activation Functions**: ReLU, Sigmoid, Tanh, Linear, Softplus, Softmax, Step, Sign
- **Loss Functions**: MSE, MAE, RMSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Optimizers**: SGD, Mini-Batch GD, Momentum, AdaGrad, RMSProp, Adam
- **Normalization**: Layer Norm, RMS Norm
- **Regularization**: L1 (Lasso) & L2 (Ridge)
- **Metrics**:
  - Regression: R², Adjusted R², MSE, MAE, RMSE
  - Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, AUC

### NLP Components
- **Tokenizer**: Word, Character, BPE (Byte Pair Encoding), and Sentence tokenization
- **Embedding Layer**: Trainable word embeddings

### Language Bindings
- **Python bindings** via pybind11 with NumPy array support

## Prerequisites

- C++17 or higher
- CMake 3.16+
- A C++ compiler (GCC, Clang, or MSVC)

## Building

### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/ProdigiousPersonn/ML-Models
cd ML-Models

# Create and enter build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# Run the executable
./Build
```

### Windows

```bash
# Clone the repository
git clone https://github.com/ProdigiousPersonn/ML-Models
cd ML-Models

# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build . --config Release

# Run the executable
.\Release\Build.exe
```

## Project Structure

```
LinearModel/
├── source/
│   ├── main.cpp           # Entry point
│   ├── math/              # Matrix operations
│   ├── core/              # Loss, optimizer, regularizer, metrics, tokenizer, embedding
│   ├── models/            # ML model implementations
│   └── utils/             # CSV utilities
├── include/ml_lib/        # Public headers
├── examples/
│   ├── c++/               # C++ examples
│   │   └── linear-regression/housing/
│   ├── python/            # Python examples
│   └── datasets/          # Example datasets
├── python/                # Python bindings (pybind11)
├── tests/                 # Unit tests (doctest)
├── external/              # Dependencies (fmt, spdlog, doctest)
├── csv-parser/            # CSV parsing library
├── pybind11/              # Python bindings library
└── CMakeLists.txt         # Build configuration
```

## Examples

### Housing Price Prediction (Linear Regression)

A complete example demonstrating linear regression on a real-world housing dataset (https://www.kaggle.com/datasets/yasserh/housing-prices-dataset):

- Dataset: 545 housing samples with 12 features (area, bedrooms, bathrooms, etc.)
- Features: Z-score normalization
- Model: Linear regression with L2 regularization
- Optimizer: Batch gradient descent
- Metrics: MSE, RMSE, MAE, R²

### Heart Disease Prediction (Logistic Regression)

A binary classification example using logistic regression on the Framingham Heart Study dataset (https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression):

- Dataset: Framingham Heart Study - 10 Year CHD Risk
- Features: 15 clinical features (age, sex, cholesterol, blood pressure, BMI, etc.)
- Preprocessing: Z-score normalization
- Model: Logistic regression with L2 regularization
- Loss: Binary Cross-Entropy (BCE)
- Optimizer: Batch gradient descent
- Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, AUC

Run the examples:
```bash
./Build
```

## Roadmap

### Regression [X]
- [x] **Linear Regression**
- [x] **Evaluation Metrics (Regression):** MSE, MAE, RMSE, R-squared
- [x] **Regularization:** L1 (Lasso) & L2 (Ridge)

### Classification [X]
- [x] **Logistic Regression**
- [x] **Evaluation Metrics (Classification):**
    - [x] Accuracy, Precision, Recall, FPR, F1-Score
    - [x] Confusion Matrix
    - [x] ROC Curve and AUC
- [x] **K-Nearest Neighbors (KNN)**
- [x] **Support Vector Machines (SVMs)**

### Tree-Based Models [X]
- [x] **Decision Trees**
- [x] **Random Forests**

### Unsupervised Learning [X]
- [x] **K-Means Clustering**

### Deep Learning [In Progress]
- [x] **Neural Networks (Feedforward)**
- [x] **Backpropagation**
- [x] **Activation Functions:** ReLU, Sigmoid, Tanh, Linear, Softplus, Softmax, Step, Sign
- [x] **Optimizers:**
    - [x] Mini-Batch Gradient Descent
    - [x] Adam Optimizer
    - [x] RMSProp
    - [x] AdaGrad
    - [x] Momentum SGD
- [ ] **Model Serialization**
- [ ] **Batch Normalization**
- [x] **Layer Normalization**
- [x] **RMS Normalization**
- [ ] **Dropout Regularization**

### NLP [In Progress]
- [x] **Tokenizer:** Word, Character, BPE, Sentence
- [x] **Embedding Layer**
- [ ] **Attention Mechanisms**
- [ ] **Transformers**
- [ ] **Language Models (Basic LLM architecture)**

### DL Architectures [ ]
- [ ] **Convolutional Neural Networks (CNNs)** (For images)
- [ ] **Recurrent Neural Networks (RNNs)** (For sequences)

## License

This project is available for educational purposes.
