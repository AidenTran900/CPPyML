#include "ml_lib/models/linear-regression.h"

LinearRegression::LinearRegression(int input_dim, LossFunction* loss, Optimizer* opt)
    : BaseModel(loss, opt), weights(input_dim, 1, 0.01), bias(1, 1, 0.0),
      grad_w(input_dim, 1, 0.0), grad_b(1, 1, 0.0) {}

// y^​=XW+b
Matrix LinearRegression::forward(const Matrix &X)
{
    last_input = X;
    Matrix result = X.multiply(weights);

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

    grad_w = last_input.transpose().multiply(error);

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