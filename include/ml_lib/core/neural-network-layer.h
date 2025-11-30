#include "../models/perceptron.h";

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
    PIECEWISE,
    SIGMOID,
        // Acts like logistic regression
    TANH,
    RELU,
    SOFTPLUS
};

class NeuralNetworkLayer {
    private:
        Matrix weights;
        Matrix bias;

        Matrix grad_w;
        Matrix grad_b;

        ACTIVATION_FUNC activation;

    public:
        NeuralNetworkLayer(int input_dim, int output_dim, ACTIVATION_FUNC act);

        Matrix forward(const Matrix& X);
        void backward(const Matrix& grad_output);
        void update(Optimizer* opt);
}