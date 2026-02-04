#include "feed-forward.h"
#include "../core/neural-network-layer.h"

FeedForward::FeedForward(int input_dim, int hidden_dim, int output_dim, ACTIVATION_FUNC act1, ACTIVATION_FUNC act2)
    : layer1(input_dim, hidden_dim, act1),
      layer2(hidden_dim, output_dim, act2)
{}

Matrix FeedForward::forward(const Matrix& input) {
    Matrix hidden = layer1.forward(input);
    Matrix output = layer2.forward(hidden);
    return output;
}


FeedForward::~FeedForward() {}