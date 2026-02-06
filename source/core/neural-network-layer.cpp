#include "ml_lib/core/neural-network-layer.h"
#include "ml_lib/core/softmax.h"
#include <cmath>

NeuralNetworkLayer::NeuralNetworkLayer(int input_dim, int output_dim, ACTIVATION_FUNC act)
{
    weights = Matrix(input_dim, output_dim);
    bias = Matrix(output_dim, 1);
    grad_w = Matrix(input_dim, output_dim);
    grad_b = Matrix(output_dim, 1);
    activation = act;
}

double NeuralNetworkLayer::applyActivation(const double x) {
    switch (activation) {
        case (ACTIVATION_FUNC::LINEAR):
            return x;
        case (ACTIVATION_FUNC::RELU):
            return std::max(0.0, x);
        case (ACTIVATION_FUNC::SIGMOID):
            return 1 / (1 + pow(exp(1.0), -x) );
        case (ACTIVATION_FUNC::SOFTMAX):
            return x;
        case (ACTIVATION_FUNC::SIGN):
            return (x > 0)? 1 : -1;
        case (ACTIVATION_FUNC::SOFTPLUS):
            return log(1 + pow(exp(1.0), x) );
        case (ACTIVATION_FUNC::STEP):
            return (x > 0)? 1 : 0;
        case (ACTIVATION_FUNC::TANH):
            return (2 / (1 + pow(exp(1.0), -2*x) )) - 1;
    }
}

double NeuralNetworkLayer::applyActivationDerivative(const double x) {
    switch (activation) {
        case (ACTIVATION_FUNC::LINEAR):
            return 1.0;
        case (ACTIVATION_FUNC::RELU):
            return (x > 0) ? 1.0 : 0.0;
        case (ACTIVATION_FUNC::SIGMOID): {
            double sig = 1.0 / (1.0 + pow(exp(1.0), -x));
            return sig * (1.0 - sig);
        }
        case (ACTIVATION_FUNC::SIGN):
            return 0.0;
        case (ACTIVATION_FUNC::SOFTPLUS):
            return 1.0 / (1.0 + pow(exp(1.0), -x));
        case (ACTIVATION_FUNC::STEP):
            return 0.0;
        case (ACTIVATION_FUNC::TANH): {
            double tanh_x = (2.0 / (1.0 + pow(exp(1.0), -2*x))) - 1.0;
            return 1.0 - tanh_x * tanh_x;
        }
        case (ACTIVATION_FUNC::SOFTMAX):
            return 1.0;
    }
}

Matrix NeuralNetworkLayer::forward(const Matrix &X)
{
    last_input = X;

    Matrix linear = X * weights;

    last_pre_activation = Matrix(linear.rows(), linear.cols());
    for (int i = 0; i < linear.rows(); i++) {
        for (int j = 0; j < linear.cols(); j++) {
            last_pre_activation(i, j) = linear(i, j) + bias(j, 0);
        }
    }

    Matrix result;
    if (activation == ACTIVATION_FUNC::SOFTMAX) {
        result = Softmax::apply(last_pre_activation);
    } else {
        result = Matrix(last_pre_activation.rows(), last_pre_activation.cols());
        for (int i = 0; i < last_pre_activation.rows(); i++) {
            for (int j = 0; j < last_pre_activation.cols(); j++) {
                result(i, j) = applyActivation(last_pre_activation(i, j));
            }
        }
    }

    last_output = result;
    return result;
}

Matrix NeuralNetworkLayer::backward(const Matrix &grad_output)
{
    Matrix grad;
    if (activation == ACTIVATION_FUNC::SOFTMAX) {
        grad = Softmax::derivative(last_output, grad_output);
    } else {
        Matrix activation_deriv = Matrix(last_pre_activation.rows(), last_pre_activation.cols());
        for (int i = 0; i < last_pre_activation.rows(); i++) {
            for (int j = 0; j < last_pre_activation.cols(); j++) {
                activation_deriv(i, j) = applyActivationDerivative(last_pre_activation(i, j));
            }
        }
        grad = grad_output.hadamard(activation_deriv);
    }

    grad_w = last_input.transpose() * grad;

    grad_b = Matrix(grad.cols(), 1, 0.0);
    for (int i = 0; i < grad.rows(); i++) {
        for (int j = 0; j < grad.cols(); j++) {
            grad_b(j, 0) += grad(i, j);
        }
    }

    Matrix grad_input = grad * weights.transpose();

    return grad_input;
}

void NeuralNetworkLayer::update(Optimizer *opt)
{
    opt->step(weights, grad_w);
    opt->step(bias, grad_b);
}
