#include "ml_lib/core/neural-network-layer.h"
#include "ml_lib/core/softmax.h"
#include <cmath>

template<typename T>
NeuralNetworkLayer<T>::NeuralNetworkLayer(int input_dim, int output_dim, ACTIVATION_FUNC act)
{
    weights = Matrix<T>(input_dim, output_dim);
    bias = Matrix<T>(output_dim, 1);
    grad_w = Matrix<T>(input_dim, output_dim);
    grad_b = Matrix<T>(output_dim, 1);
    activation = act;
}

template<typename T>
T NeuralNetworkLayer<T>::applyActivation(T x) {
    switch (activation) {
        case (ACTIVATION_FUNC::LINEAR):
            return x;
        case (ACTIVATION_FUNC::RELU):
            return std::max(static_cast<T>(0.0), x);
        case (ACTIVATION_FUNC::SIGMOID):
            return static_cast<T>(1.0) / (static_cast<T>(1.0) + pow(exp(static_cast<T>(1.0)), -x) );
        case (ACTIVATION_FUNC::SOFTMAX):
            return x;
        case (ACTIVATION_FUNC::SIGN):
            return (x > static_cast<T>(0.0))? static_cast<T>(1.0) : static_cast<T>(-1.0);
        case (ACTIVATION_FUNC::SOFTPLUS):
            return log(static_cast<T>(1.0) + pow(exp(static_cast<T>(1.0)), x) );
        case (ACTIVATION_FUNC::STEP):
            return (x > static_cast<T>(0.0))? static_cast<T>(1.0) : static_cast<T>(0.0);
        case (ACTIVATION_FUNC::TANH):
            return (static_cast<T>(2.0) / (static_cast<T>(1.0) + pow(exp(static_cast<T>(1.0)), static_cast<T>(-2.0)*x) )) - static_cast<T>(1.0);
    }
}

template<typename T>
T NeuralNetworkLayer<T>::applyActivationDerivative(T x) {
    switch (activation) {
        case (ACTIVATION_FUNC::LINEAR):
            return static_cast<T>(1.0);
        case (ACTIVATION_FUNC::RELU):
            return (x > static_cast<T>(0.0)) ? static_cast<T>(1.0) : static_cast<T>(0.0);
        case (ACTIVATION_FUNC::SIGMOID): {
            T sig = static_cast<T>(1.0) / (static_cast<T>(1.0) + pow(exp(static_cast<T>(1.0)), -x));
            return sig * (static_cast<T>(1.0) - sig);
        }
        case (ACTIVATION_FUNC::SIGN):
            return static_cast<T>(0.0);
        case (ACTIVATION_FUNC::SOFTPLUS):
            return static_cast<T>(1.0) / (static_cast<T>(1.0) + pow(exp(static_cast<T>(1.0)), -x));
        case (ACTIVATION_FUNC::STEP):
            return static_cast<T>(0.0);
        case (ACTIVATION_FUNC::TANH): {
            T tanh_x = (static_cast<T>(2.0) / (static_cast<T>(1.0) + pow(exp(static_cast<T>(1.0)), static_cast<T>(-2.0)*x))) - static_cast<T>(1.0);
            return static_cast<T>(1.0) - tanh_x * tanh_x;
        }
        case (ACTIVATION_FUNC::SOFTMAX):
            return static_cast<T>(1.0);
    }
}

template<typename T>
Matrix<T> NeuralNetworkLayer<T>::forward(const Matrix<T> &X)
{
    last_input = X;

    Matrix<T> linear = X * weights;

    last_pre_activation = Matrix<T>(linear.rows(), linear.cols());
    for (int i = 0; i < linear.rows(); i++) {
        for (int j = 0; j < linear.cols(); j++) {
            last_pre_activation(i, j) = linear(i, j) + bias(j, 0);
        }
    }

    Matrix<T> result;
    if (activation == ACTIVATION_FUNC::SOFTMAX) {
        result = Softmax::apply<T>(last_pre_activation);
    } else {
        result = Matrix<T>(last_pre_activation.rows(), last_pre_activation.cols());
        for (int i = 0; i < last_pre_activation.rows(); i++) {
            for (int j = 0; j < last_pre_activation.cols(); j++) {
                result(i, j) = applyActivation(last_pre_activation(i, j));
            }
        }
    }

    last_output = result;
    return result;
}

template<typename T>
Matrix<T> NeuralNetworkLayer<T>::backward(const Matrix<T> &grad_output)
{
    Matrix<T> grad;
    if (activation == ACTIVATION_FUNC::SOFTMAX) {
        grad = Softmax::derivative<T>(last_output, grad_output);
    } else {
        Matrix<T> activation_deriv = Matrix<T>(last_pre_activation.rows(), last_pre_activation.cols());
        for (int i = 0; i < last_pre_activation.rows(); i++) {
            for (int j = 0; j < last_pre_activation.cols(); j++) {
                activation_deriv(i, j) = applyActivationDerivative(last_pre_activation(i, j));
            }
        }
        grad = grad_output.hadamard(activation_deriv);
    }

    grad_w = last_input.transpose() * grad;

    grad_b = Matrix<T>(grad.cols(), 1, 0.0);
    for (int i = 0; i < grad.rows(); i++) {
        for (int j = 0; j < grad.cols(); j++) {
            grad_b(j, 0) += grad(i, j);
        }
    }

    Matrix<T> grad_input = grad * weights.transpose();

    return grad_input;
}

template<typename T>
void NeuralNetworkLayer<T>::update(Optimizer<T> *opt)
{
    opt->step(weights, grad_w);
    opt->step(bias, grad_b);
}

template class NeuralNetworkLayer<float>;
template class NeuralNetworkLayer<double>;
