#include "neural-network-ff.h"
#include "neural-network-layer.h"
#include <algorithm>

NeuralNetworkFF::NeuralNetworkFF(int input_dim, LossFunction *loss, Optimizer *opt, Regularizer *reg)
    : GradientModel(loss, opt, reg)
{}

void NeuralNetworkFF::addLayer(int input_dim, int output_dim, ACTIVATION_FUNC act)
{
    this->layers.push_back(NeuralNetworkLayer(input_dim, output_dim, act));
}

Matrix NeuralNetworkFF::forward(const Matrix &X)
{
    last_input = X;
    for (auto& layer : layers) {
        last_input = layer.forward(last_input);
    }
    last_output = last_input;

    return last_output;
}

void NeuralNetworkFF::backward(const Matrix &y_true)
{
    Matrix last_gradient = loss_func->gradient(last_output, y_true);
    for (int i = layers.size() - 1; i >= 0; i--) {
        last_gradient = layers[i].backward(last_gradient);
    }
}

void NeuralNetworkFF::update()
{
    for (auto& layer : layers) {
        layer.update(optimizer.get());
    }
}
