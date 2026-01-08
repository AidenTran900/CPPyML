#include "neural-network-layer.h"
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
    }
}

Matrix NeuralNetworkLayer::forward(const Matrix &X)
{
    last_input = X;
    last_pre_activation = X * weights + bias;

    Matrix result = Matrix(last_pre_activation.rows(), last_pre_activation.cols());
    for (int i = 0; i < last_pre_activation.rows(); i++) {
        result(i, 0) = applyActivation(last_pre_activation(i,0));
    }

    last_output = result;
    return result;
}

Matrix NeuralNetworkLayer::backward(const Matrix &grad_output)
{
    Matrix activation_deriv = Matrix(last_pre_activation.rows(), last_pre_activation.cols());
    for (int i = 0; i < last_pre_activation.rows(); i++) {
        activation_deriv(i, 0) = applyActivationDerivative(last_pre_activation(i, 0));
    }

    Matrix grad = grad_output.hadamard(activation_deriv);

    grad_w = last_input.transpose() * grad;
    grad_b = grad;

    Matrix grad_input = grad * weights.transpose();

    return grad_input;
}

void NeuralNetworkLayer::update(Optimizer *opt)
{
    weights = weights - grad_w * opt->getLearningRate();
    bias = bias - grad_b * opt->getLearningRate();
}
