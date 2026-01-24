#pragma once
#include "../math/matrix.h"
#include "gradient-model.h"

// y^​=XW+b
// y is our approximation
// X is our design matrix
// W is our WEIGHT matrix / slope
// b is our BIAS / y intercept
class LinearRegression : public GradientModel {
    private:
        Matrix weights;
        Matrix bias;

        Matrix grad_w;
        Matrix grad_b;

    public:
        LinearRegression(int input_dim,
                         std::unique_ptr<LossFunction> loss,
                         std::unique_ptr<Optimizer> opt,
                         std::unique_ptr<Regularizer> reg);

        Matrix forward(const Matrix& X) override;
        void backward(const Matrix& y_true) override;
        void update() override;
};