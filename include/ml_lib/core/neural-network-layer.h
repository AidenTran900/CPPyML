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

class NeuralNetworkLayer {
    private:
        Matrix weights;
        Matrix bias;

        Matrix grad_w;
        Matrix grad_b;

        Matrix last_input;
        Matrix last_output;
        Matrix last_pre_activation;

        ACTIVATION_FUNC activation;

    public:
        NeuralNetworkLayer(int input_dim, int output_dim, ACTIVATION_FUNC act);

        double applyActivation(const double x);
        double applyActivationDerivative(const double x);
        Matrix applySoftmax(const Matrix& x);
        Matrix applySoftmaxDerivative(const Matrix& softmax_output, const Matrix& grad_output);

        Matrix forward(const Matrix& X);
        Matrix backward(const Matrix &grad_output);
        void update(Optimizer* opt);
};