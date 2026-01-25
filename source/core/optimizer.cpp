#include "ml_lib/core/optimizer.h"
#include <cmath>

// Stochastic
void StochasticOptimizer::step(Matrix& param, const Matrix& grad)
{
    param = param - grad * learning_rate;
}


// Mini Batch
void MiniBatchOptimizer::step(Matrix& param, const Matrix& grad)
{
    Matrix* key = &param;
    if (accumulated_grads.find(key) == accumulated_grads.end()) {
        accumulated_grads[key] = Matrix(grad.rows(), grad.cols(), 0.0);
    }
    accumulated_grads[key] = accumulated_grads[key] + grad;
    current_step++;

    if (current_step >= batch_size) {
        for (auto& [param_ptr, acc_grad] : accumulated_grads) {
            *param_ptr = *param_ptr - acc_grad * (learning_rate / batch_size);
        }
        accumulated_grads.clear();
        current_step = 0;
    }
}

std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, double lr)
{
    switch (type) {
        case OptimizerType::STOCHASTIC:
            return std::make_unique<StochasticOptimizer>(lr);
        case OptimizerType::MINI_BATCH:
            return std::make_unique<MiniBatchOptimizer>(lr);
        default:
            return std::make_unique<StochasticOptimizer>(lr);
    }
}
