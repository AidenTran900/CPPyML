#pragma once
#include "../math/matrix.h"
#include "gradient-model.h"

// y^​=activation(XW+b)
// y is our approximation
// X is our design matrix
// W is our WEIGHT matrix / slope
// b is our BIAS / y intercept
// activation is the step function
class Perceptron : public GradientModel {
    private:
        Matrix weights;
        Matrix bias;

        Matrix grad_w;
        Matrix grad_b;

    public:
        Perceptron(int input_dim, LossFunction* loss, Optimizer* opt, Regularizer* reg);

        Matrix forward(const Matrix& X) override;
        void backward(const Matrix& y_true) override;
        void update() override;

        Matrix predict(const Matrix& X);
};