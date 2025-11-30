#include "ml_lib/models/perceptron.h"
#include <cmath>

Perceptron::Perceptron(int input_dim, LossFunction* loss, Optimizer* opt, Regularizer* reg)
    : GradientModel(loss, opt, reg), weights(input_dim, 1, 0.01), bias(1, 1, 0.0),
      grad_w(input_dim, 1, 0.0), grad_b(1, 1, 0.0) {}

Matrix Perceptron::forward(const Matrix &X)
{
    last_input = X;
    Matrix result = X.multiply(weights);

    // Step func
    for (int i = 0; i < result.rows(); i++) {
        double linear_output = result(i, 0) + bias(0, 0);
        result(i, 0) = (linear_output >= 0.0) ? 1.0 : 0.0;
    }

    last_output = result;
    return result;
}

void Perceptron::backward(const Matrix& y_true) {
    int m = last_input.rows();
    if (m == 0) return;

    Matrix predictions = last_output;
    Matrix error = loss_func->gradient(predictions, y_true);
    Matrix reg_vals = regularizer->gradient(weights);

    grad_w = last_input.transpose().multiply(error).add(reg_vals);

    double grad_b_sum = 0.0;
    for (int j = 0; j < error.rows(); j++) {
        grad_b_sum += error(j, 0);
    }
    grad_b = Matrix(bias.rows(), bias.cols(), grad_b_sum);
}

void Perceptron::update() {
    optimizer->step(weights, grad_w);
    optimizer->step(bias, grad_b);
}

Matrix Perceptron::predict(const Matrix& X) {
    return forward(X);
}
