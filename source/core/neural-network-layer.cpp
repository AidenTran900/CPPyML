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

// Softmax Stuff
Matrix NeuralNetworkLayer::applySoftmax(const Matrix& x) {
    Matrix result(x.rows(), x.cols());

    double max_val = x(0,0);
    for (int i = 0; i < x.rows(); i++) {
        if (x(i,0) > max_val) {
            max_val = x(i,0);  
        }
    }

    // Exponentiation
    double sum = 0.0;
    for (int i = 0; i < x.rows(); i++) {
        result(i, 0) = std::exp(x(i, 0) - max_val);
        sum += result(i, 0);
    }

    // Normalization
    for (int i = 0; i < x.rows(); i++) {
        result(i, 0) /= sum;
    }

    return result;

}

Matrix NeuralNetworkLayer::applySoftmaxDerivative(const Matrix& softmax_output, const Matrix& grad_output) {
    // Jacobian of softmax: dS_i/dx_j = S_i * (delta_ij - S_j)
    // For backprop: grad_input_i = sum_j(grad_output_j * dS_j/dx_i)
    //             = sum_j(grad_output_j * S_j * (delta_ji - S_i))
    //             = S_i * grad_output_i - S_i * sum_j(grad_output_j * S_j)
    //             = S_i * (grad_output_i - dot(grad_output, S))

    int n = softmax_output.rows();
    Matrix result(n, 1);

    double dot_product = 0.0;
    for (int i = 0; i < n; i++) {
        dot_product += grad_output(i, 0) * softmax_output(i, 0);
    }

    for (int i = 0; i < n; i++) {
        result(i, 0) = softmax_output(i, 0) * (grad_output(i, 0) - dot_product);
    }

    return result;
}

        
// Main methods
Matrix NeuralNetworkLayer::forward(const Matrix &X)
{
    last_input = X;
    last_pre_activation = X * weights + bias;

    Matrix result;
    if (activation == ACTIVATION_FUNC::SOFTMAX) {
        result = applySoftmax(last_pre_activation);
    } else {
        result = Matrix(last_pre_activation.rows(), last_pre_activation.cols());
        for (int i = 0; i < last_pre_activation.rows(); i++) {
            result(i, 0) = applyActivation(last_pre_activation(i,0));
        }
    }

    last_output = result;
    return result;
}

Matrix NeuralNetworkLayer::backward(const Matrix &grad_output)
{
    Matrix grad;
    if (activation == ACTIVATION_FUNC::SOFTMAX) {
        grad = applySoftmaxDerivative(last_output, grad_output);
    } else {
        Matrix activation_deriv = Matrix(last_pre_activation.rows(), last_pre_activation.cols());
        for (int i = 0; i < last_pre_activation.rows(); i++) {
            activation_deriv(i, 0) = applyActivationDerivative(last_pre_activation(i, 0));
        }
        grad = grad_output.hadamard(activation_deriv);
    }

    grad_w = last_input.transpose() * grad;
    grad_b = grad;

    Matrix grad_input = grad * weights.transpose();

    return grad_input;
}

void NeuralNetworkLayer::update(Optimizer *opt)
{
    opt->step(weights, grad_w);
    opt->step(bias, grad_b);
}
