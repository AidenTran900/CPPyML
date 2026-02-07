# ML Models

A C++ machine learning library built from the ground up implementing various ML algorithms and models.

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
- **Residual Networks (ResNet)** with skip connections
- **Transformer** with multi-head self-attention, KV cache, and autoregressive generation
- **Perceptron** for binary classification

### Core Components
- **Matrix Operations**: Addition, multiplication, transpose, inverse, Hadamard product, determinant — templated for `float` and `double` (`Matrix<float>` / `Matrix<double>`)
- **Activation Functions**: ReLU, Sigmoid, Tanh, Linear, Softplus, Softmax, Step, Sign
- **Loss Functions**: MSE, MAE, RMSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Optimizers**: SGD, Mini-Batch GD, Momentum, AdaGrad, RMSProp, Adam
- **Normalization**: Layer Norm, RMS Norm
- **Regularization**: L1 (Lasso) & L2 (Ridge)
- **Metrics**:
  - Regression: R², Adjusted R², MSE, MAE, RMSE
  - Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, AUC

### NLP / Transformer Components
- **Tokenizer**: Word, Character, BPE (Byte Pair Encoding), and Sentence tokenization
- **Embedding Layer**: Trainable word embeddings
- **Multi-Head Attention**: Scaled dot-product attention with KV cache for efficient inference
- **Positional Encoding**: Sinusoidal and Rotary (RoPE)
- **Transformer Blocks**: Pre-norm architecture with residual connections

### Precision Support
All core classes are templated on scalar type (`template<typename T = double>`), enabling both `float` (f32) and `double` (f64) precision:
- `Matrix<float>` / `MatrixF32` for memory-efficient inference
- `Matrix<double>` / `MatrixF64` for training precision (default)
- Classical ML models default to `double`; the transformer stack supports both

### Language Bindings
- **Python bindings** via pybind11 with NumPy array support (both `float32` and `float64`)

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
│   ├── logistic-regression/ # Heart disease classification example
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

### NLP / Transformers [In Progress]
- [x] **Tokenizer:** Word, Character, BPE, Sentence
- [x] **Embedding Layer**
- [x] **Attention Mechanisms:** Multi-head self-attention with KV cache
- [x] **Positional Encoding:** Sinusoidal, Rotary (RoPE)
- [x] **Transformer Blocks:** Pre-norm with residual connections
- [x] **Transformer Model:** Autoregressive generation with token sampling
- [ ] **Language Models (Basic LLM architecture / GGUF loading)**

### Precision [ ]
- [x] **f64 (double):** Default precision for all operations
- [x] **f32 (float):** Template support across the full stack
- [ ] **f16 / Quantization:** For efficient model loading

### DL Architectures [ ]
- [ ] **Convolutional Neural Networks (CNNs)** (For images)
- [ ] **Recurrent Neural Networks (RNNs)** (For sequences)
