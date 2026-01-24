#include "ml_lib/models/linear-regression.h"

LinearRegression::LinearRegression(int input_dim, LossFunction* loss, Optimizer* opt, Regularizer* reg)
    : GradientModel(loss, opt, reg)
{
    weights = Matrix(input_dim, 1);
    bias = Matrix(1, 1);
    grad_w = Matrix(input_dim, 1);
    grad_b = Matrix(1, 1);
}

// y^​=XW+b
Matrix LinearRegression::forward(const Matrix &X)
{
    last_input = X;
    Matrix result = X * weights;

    for (int i = 0; i < result.rows(); i++) {
        result(i, 0) = result(i, 0) + bias(0, 0);
    }

    last_output = result;
    return result;
}

void LinearRegression::backward(const Matrix& y_true)
{
    int m = last_input.rows();
    if (m == 0) return;

    Matrix predictions = last_output;
    Matrix error = loss_func->gradient(predictions, y_true);
    Matrix reg_vals = regularizer->gradient(weights);

    // Weights
    grad_w = last_input.transpose() * error + reg_vals;

    // Bias
    double grad_b_sum = 0.0;
    for (int j = 0; j < error.rows(); j++) {
        grad_b_sum += error(j, 0);
    }
    grad_b = Matrix(bias.rows(), bias.cols(), grad_b_sum);
}

void LinearRegression::update()
{
    optimizer->step(weights, grad_w);
    optimizer->step(bias, grad_b);
}
