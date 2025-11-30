#pragma once
#include "../math/matrix.h"
#include "gradient-model.h"

class NeuralNetwork : public GradientModel {
    private:
        Matrix weights;
        Matrix bias;

        Matrix grad_w;
        Matrix grad_b;

    public:
        NeuralNetwork(int input_dim, LossFunction* loss, Optimizer* opt, Regularizer* reg);

        Matrix forward(const Matrix& X) override;
        void backward(const Matrix& y_true) override;
        void update() override;
};