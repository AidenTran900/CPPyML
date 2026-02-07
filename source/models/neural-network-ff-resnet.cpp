#include "neural-network-resnet.h"
#include "neural-network-layer.h"
#include <algorithm>

NeuralNetworkResNet::NeuralNetworkResNet(int input_dim, std::unique_ptr<LossFunction<>> loss, std::unique_ptr<Optimizer<>> opt, std::unique_ptr<Regularizer<>> reg, int skips)
    : GradientModel<>(std::move(loss), std::move(opt), std::move(reg)), skips(skips)
{}

void NeuralNetworkResNet::addLayer(int input_dim, int output_dim, ACTIVATION_FUNC act)
{
    this->layers.push_back(NeuralNetworkLayer<>(input_dim, output_dim, act));
}

Matrix<> NeuralNetworkResNet::forward(const Matrix<> &X)
{
    last_input = X;
    Matrix<> residual = X;

    for (int i = 0; i < layers.size(); i++) {
        last_input = layers[i].forward(last_input);

        if ((i + 1) % skips == 0 && last_input.rows() == residual.rows() && last_input.cols() == residual.cols()) {
            last_input = last_input + residual;
            residual = last_input;
        }
    }
    last_output = last_input;

    return last_output;
}

void NeuralNetworkResNet::backward(const Matrix<> &y_true)
{
    Matrix<> last_gradient = loss_func->gradient(last_output, y_true);
    for (int i = layers.size() - 1; i >= 0; i--) {
        Matrix<> layer_grad = layers[i].backward(last_gradient);

        if ((i + 1) % skips == 0) {
            last_gradient = layer_grad + last_gradient;
        } else {
            last_gradient = layer_grad;
        }
    }
}

void NeuralNetworkResNet::update()
{
    for (auto& layer : layers) {
        layer.update(optimizer.get());
    }
}
