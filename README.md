# ML Models

A C++ machine learning library built from the ground up. Implementing various ML algorithms and models using fundamental linear algebra and optimization techniques.

## Features

- **Linear regression** with gradient descent optimization
- **Logistic regression** for binary classification
- **K-Nearest Neighbors (KNN)** for classification
- **Support Vector Machines (SVM)** for classification
- **Decision Trees** for classification with Gini and Entropy impurity measures
- **Random Forests** for ensemble learning with bootstrap aggregation
- **K-Means Clustering** for unsupervised learning and data segmentation
- **Neural Networks** with backpropagation and multiple activation functions (ReLU, Sigmoid, Tanh, Linear, Softplus)
- Matrix operations library (addition, multiplication, transpose, inverse, element-wise operations)
- Loss functions: MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, L1, L2, BinaryCrossEntropy
- Evaluation metrics:
  - Regression: R2, Adjusted R2, MSE, MAE, RMSE
  - Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, AUC
- Regularization: L1 (Lasso) & L2 (Ridge)
- Real-world examples:
  - Housing price prediction (Linear Regression)
  - Heart disease prediction (Logistic Regression)

## Prerequisites

- C++14 or higher
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
│   ├── core/              # Loss, optimizer, regularizer, metrics
│   └── models/            # ML model implementations (Linear/Logistic Regression, KNN, SVM)
├── include/
│   └── ml_lib/            # Public headers
├── examples/
│   ├── linear-regression/
│   │   └── housing/       # Housing price prediction example
│   └── logistic-regression/
│       └── h-disease-example.cpp  # Heart disease prediction example
├── tests/                 # Unit tests
├── external/              # Dependencies (fmt, spdlog, doctest)
└── CMakeLists.txt        # Build configuration
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
- [X] **Regularization:** L1 (Lasso) & L2 (Ridge)

### Classification [X]
- [x] **Logistic Regression**
- [x] **Evaluation Metrics (Classification):**
    - [x] Accuracy, Precision, Recall, FPR, F1-Score
    - [X] Confusion Matrix
    - [X] ROC Curve and AUC
- [X] **K-Nearest Neighbors (KNN)**
- [X] **Support Vector Machines (SVMs)**

### Tree-Based Models [X]
- [X] **Decision Trees**
- [X] **Random Forests**

### Unsupervised Learning [X]
- [X] **K-Means Clustering**

### Deep Learning [In Progress]
- [X] **Neural Networks (Feedforward)**
- [X] **Backpropagation**
- [X] **Activation Functions:** ReLU, Sigmoid, Tanh, Linear, Softplus, Step, Sign
- [ ] **Optimizers:**
    - [ ] **Mini-Batch Gradient Descent**
    - [ ] **Adam Optimizer**
    - [ ] **RMSProp**
- [ ] **Model Serialization**
- [ ] **Batch Normalization**
- [ ] **Dropout Regularization**

### DL Architectures [ ]
- [ ] **Convolutional Neural Networks (CNNs)** (For images)
- [ ] **Recurrent Neural Networks (RNNs)** (For sequences)

### Modern NLP [ ]
- [ ] **Embeddings (Word2Vec, GloVe)**
- [ ] **Attention Mechanisms**
- [ ] **Transformers**
- [ ] **Language Models (Basic LLM architecture)**

## License

This project is available for educational purposes.
