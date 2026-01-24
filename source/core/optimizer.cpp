#include "ml_lib/core/optimizer.h"

// Batch
void BatchOptimizer::step(Matrix& param, const Matrix& grad)
{
    param = param - grad * learning_rate;
}


// Stochastic
void StochasticOptimizer::step(Matrix& param, const Matrix& grad)
{
    param = param - grad * learning_rate;
}


// Mini Batch
void MiniBatchOptimizer::step(Matrix& param, const Matrix& grad)
{
    param = param - grad * learning_rate;
}

std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, double lr)
{
    switch (type) {
        case OptimizerType::BATCH:
            return std::make_unique<BatchOptimizer>(lr);
        case OptimizerType::STOCHASTIC:
            return std::make_unique<StochasticOptimizer>(lr);
        case OptimizerType::MINI_BATCH:
            return std::make_unique<MiniBatchOptimizer>(lr);
        default:
            return std::make_unique<BatchOptimizer>(lr);
    }
}
