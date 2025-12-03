#include "neural-network.h"
#include "neural-network-layer.h"

NeuralNetwork::NeuralNetwork(int input_dim, LossFunction *loss, Optimizer *opt, Regularizer *reg)
    : GradientModel(loss, opt, reg)
{}

void NeuralNetwork::addLayer(int input_dim, int output_dim, ACTIVATION_FUNC act)
{
    this->layers.push_back(NeuralNetworkLayer(input_dim, output_dim, act));
}

Matrix NeuralNetwork::forward(const Matrix &X)
{
    for (auto& layer : layers) {
        last_input = layer.forward(last_input);
    }
    last_output = last_input;
}

void NeuralNetwork::backward(const Matrix &y_true)
{
}

void NeuralNetwork::update()
{
}
