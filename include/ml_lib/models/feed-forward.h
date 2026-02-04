#pragma once
#include "../math/matrix.h"
#include "../core/neural-network-layer.h"

class FeedForward {
    private:
        NeuralNetworkLayer layer1;
        NeuralNetworkLayer layer2;
    public:
        FeedForward(int input_dim, int hidden_dim, int output_dim, ACTIVATION_FUNC act1, ACTIVATION_FUNC act2);
        virtual Matrix forward(const Matrix& input);
        virtual ~FeedForward() = default;
};