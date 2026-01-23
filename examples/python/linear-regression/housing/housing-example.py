#!/usr/bin/env python3
"""
Housing Price Prediction - Linear Regression Example

This example demonstrates linear regression for predicting housing prices
using the housing dataset. It uses the same data processing as the C++ version.
"""

import sys
import os

# Add processing path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../processing/python'))

import numpy as np
from housing import load_housing_data, process_housing_data


class LinearRegression:
    """Simple Linear Regression with gradient descent."""

    def __init__(self, n_features, learning_rate=0.01, l2_lambda=0.01):
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self._X = None
        self._y_pred = None

    def forward(self, X):
        self._X = X
        self._y_pred = X @ self.weights + self.bias
        return self._y_pred

    def compute_loss(self, y_pred, y_true):
        mse = np.mean((y_pred - y_true) ** 2)
        l2_reg = self.l2_lambda * np.sum(self.weights ** 2)
        return mse + l2_reg

    def backward(self, y_true):
        n_samples = self._X.shape[0]
        error = self._y_pred - y_true

        self._d_weights = (2 / n_samples) * (self._X.T @ error) + 2 * self.l2_lambda * self.weights
        self._d_bias = (2 / n_samples) * np.sum(error)

    def update(self):
        self.weights -= self.learning_rate * self._d_weights
        self.bias -= self.learning_rate * self._d_bias


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def run_housing_example():
    print("-" * 60)
    print("Housing Price Prediction - Linear Regression")
    print("-" * 60)

    # Load and process data using shared processing module
    file_path = "examples/datasets/housing.csv"
    df = load_housing_data(file_path)

    if df is None:
        print("Error: No data loaded!")
        return 1

    X_train, y_train, X_test, y_test = process_housing_data(df)
    print("Features normalized using z-score normalization\n")

    print("-" * 60)
    print("Dataset Split:")
    print("-" * 60)
    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Test samples:     {X_test.shape[0]}")
    print(f"    Features:         {X_train.shape[1]}")
    print("    Feature names: area, bedrooms, bathrooms, stories,")
    print("        mainroad, guestroom, basement, hotwaterheating,")
    print("        airconditioning, parking, prefarea, furnishing\n")

    # Create and train model
    model = LinearRegression(
        n_features=X_train.shape[1],
        learning_rate=0.01,
        l2_lambda=0.01
    )

    print("-" * 60)
    print("Training Progress:")
    print("-" * 60)

    epochs = 2000
    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        loss = model.compute_loss(y_pred, y_train)
        model.backward(y_train)
        model.update()

        if epoch % 100 == 0 or epoch == epochs - 1:
            r2_score = r2(y_train, y_pred)
            print(f"Epoch {epoch:4d}: Loss = {loss:.6f}, R² = {r2_score:.4f}")

    # Training metrics
    print(f"\n{'-' * 60}")
    print("Training Set Metrics:")
    print("-" * 60)
    y_train_pred = model.forward(X_train)

    print(f"MSE:  {mse(y_train, y_train_pred):.4f} (million²)")
    print(f"RMSE: {rmse(y_train, y_train_pred):.4f} million")
    print(f"MAE:  {mae(y_train, y_train_pred):.4f} million")
    print(f"R²:   {r2(y_train, y_train_pred):.4f}")

    # Test metrics
    print(f"\n{'-' * 60}")
    print("Test Set Metrics:")
    print("-" * 60)
    y_test_pred = model.forward(X_test)

    print(f"MSE:  {mse(y_test, y_test_pred):.4f} (million²)")
    print(f"RMSE: {rmse(y_test, y_test_pred):.4f} million")
    print(f"MAE:  {mae(y_test, y_test_pred):.4f} million")
    print(f"R²:   {r2(y_test, y_test_pred):.4f}")

    # Sample predictions
    print(f"\n{'-' * 60}")
    print("Sample Predictions (Test Set - First 10):")
    print("-" * 60)
    print(f"{'Actual (M)':<15} {'Predicted (M)':<15} {'Error (M)':<15} {'Error %':<15}")
    print("-" * 60)

    display_count = min(10, len(y_test))
    for i in range(display_count):
        actual = y_test[i, 0]
        predicted = y_test_pred[i, 0]
        error = actual - predicted
        error_pct = (abs(error) / actual) * 100
        print(f"{actual:<15.3f} {predicted:<15.3f} {error:<15.3f} {error_pct:<15.2f}")

    # Model summary
    print(f"\n{'-' * 60}")
    print("Model Summary:")
    print("-" * 60)
    print("Loss Function: Mean Squared Error (MSE)")
    print("Optimizer: Batch Gradient Descent (lr=0.01)")
    print("Regularization: L2 (lambda=0.01)")
    print(f"Training Epochs: {epochs}")
    print("-" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(run_housing_example())
