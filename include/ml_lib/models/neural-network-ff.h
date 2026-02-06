#pragma once
#include "../math/matrix.h"
#include "gradient-model.h"
#include "../core/neural-network-layer.h"

class NeuralNetworkFF : public GradientModel {
    private:
        std::vector<NeuralNetworkLayer> layers;

        Matrix last_input;
        Matrix last_output;

    public:
        NeuralNetworkFF(int input_dim, LossFunction* loss, Optimizer* opt, Regularizer* reg);

        void addLayer(int input_dim, int output_dim, ACTIVATION_FUNC act);

        Matrix forward(const Matrix& X) override;
        void backward(const Matrix& y_true) override;
        void update() override;
};