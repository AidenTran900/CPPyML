#include "ml_lib/models/logistic-regression.h"
#include <cmath>

LogisticRegression::LogisticRegression(int input_dim, LossFunction* loss, Optimizer* opt, Regularizer* reg)
    : LinearRegression(input_dim, loss, opt, reg), weights(input_dim, 1, 0.01), bias(1, 1, 0.0),
      grad_w(input_dim, 1, 0.0), grad_b(1, 1, 0.0) {}

// y^​=XW+b
Matrix LogisticRegression::forward(const Matrix &X)
{
    last_input = X;
    Matrix result = X * weights + bias;

    for (int i = 0; i < result.rows(); i++) {
        result(i, 0) = 1 / (1 + pow(std::exp(1.0), -(result(i, 0))) );
    }

    last_output = result;
    return result;
}

void LogisticRegression::backward(const Matrix& y_true) {
    int m = last_input.rows();
    if (m == 0) return;

    Matrix predictions = last_output;
    Matrix error = loss_func->gradient(predictions, y_true);
    Matrix reg_vals = regularizer->gradient(weights);

    grad_w = last_input.transpose() * error + reg_vals;

    double grad_b_sum = 0.0;
    for (int j = 0; j < error.rows(); j++) {
        grad_b_sum += error(j, 0);
    }
    grad_b = Matrix(bias.rows(), bias.cols(), grad_b_sum);
}

void LogisticRegression::update() {
    optimizer->step(weights, grad_w);
    optimizer->step(bias, grad_b);
}

Matrix LogisticRegression::predict(const Matrix& y) {
    int y_rows = y.rows();
    int y_cols = y.cols();

    Matrix predictions = Matrix(y_rows, y_cols, 0);
    for (int i = 0; i < y_rows; i++) {
        for (int j = 0; j < y_cols; j++) {
            predictions(i, j) = (y(i, j) > this->threshold) ? 1 : 0;
        }
    }
    return predictions;
}
