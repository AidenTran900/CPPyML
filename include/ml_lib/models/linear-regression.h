#pragma once
#include "../math/matrix.h"
#include "base-model.h"

// y^​=XW+b
// y is our approximation
// X is our design matrix
// W is our WEIGHT matrix / slope
// b is our BIAS / y intercept
class LinearRegression : public BaseModel {
    private:
        Matrix weights;
        Matrix bias;

        Matrix grad_w;
        Matrix grad_b;

    public:
        LinearRegression(int input_dim, LossFunction* loss, Optimizer* opt);

        Matrix forward(const Matrix& X) override;
        void backward(const Matrix& y_true) override;
        void update() override;
};