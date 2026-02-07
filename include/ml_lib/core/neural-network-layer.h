#pragma once
#include "../models/perceptron.h"

// y^​=activation(XW+b)
// y is our approximation
// X is our design matrix
// W is our WEIGHT matrix / slope
// b is our BIAS / y intercept
// activation is the step function
enum ACTIVATION_FUNC {
    STEP,
        // Step func
        // Acts like perceptron
    SIGN,
    LINEAR,
        // Acts like linear regression
    SIGMOID,
        // Acts like logistic regression
    TANH,
    RELU,
    SOFTPLUS,
    SOFTMAX
};

template<typename T = double>
class NeuralNetworkLayer {
    private:
        Matrix<T> weights;
        Matrix<T> bias;

        Matrix<T> grad_w;
        Matrix<T> grad_b;

        Matrix<T> last_input;
        Matrix<T> last_output;
        Matrix<T> last_pre_activation;

        ACTIVATION_FUNC activation;

    public:
        NeuralNetworkLayer(int input_dim, int output_dim, ACTIVATION_FUNC act);

        T applyActivation(T x);
        T applyActivationDerivative(T x);

        Matrix<T> forward(const Matrix<T>& X);
        Matrix<T> backward(const Matrix<T> &grad_output);
        void update(Optimizer<T>* opt);
};
