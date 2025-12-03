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


Matrix NeuralNetworkLayer::forward(const Matrix &X)
{
    last_input = X;
    Matrix result = X.multiply(weights).add(bias);

    for (int i = 0; i < result.rows(); i++) {
        result(i, 0) = applyActivation(result(i,0));
    }

    last_output = result;
    return result;
}

void NeuralNetworkLayer::backward(const Matrix &grad_output, double learning_rate)
{
    
}

void NeuralNetworkLayer::update(Optimizer *opt)
{

}
