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

// Adam
void AdamOptimizer::step(Matrix& param, const Matrix& grad)
{
    const Matrix* key = &param;

    if (first_moment.find(key) == first_moment.end()) {
        first_moment[key] = Matrix(param.rows(), param.cols(), 0.0);
        second_moment[key] = Matrix(param.rows(), param.cols(), 0.0);
        timesteps[key] = 0;
    }

    Matrix& first_moment_estimate = first_moment[key];
    Matrix& second_moment_estimate = second_moment[key];
    timesteps[key]++;
    int current_timestep = timesteps[key];

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            first_moment_estimate(i, j) = beta1 * first_moment_estimate(i, j) + (1.0 - beta1) * grad(i, j);
            second_moment_estimate(i, j) = beta2 * second_moment_estimate(i, j) + (1.0 - beta2) * grad(i, j) * grad(i, j);

            double bias_corrected_first = first_moment_estimate(i, j) / (1.0 - std::pow(beta1, current_timestep));
            double bias_corrected_second = second_moment_estimate(i, j) / (1.0 - std::pow(beta2, current_timestep));

            param(i, j) = param(i, j) - learning_rate * bias_corrected_first / (std::sqrt(bias_corrected_second) + epsilon);
        }
    }
}

// Momentum
void MomentumOptimizer::step(Matrix& param, const Matrix& grad)
{
    const Matrix* key = &param;

    if (velocity.find(key) == velocity.end()) {
        velocity[key] = Matrix(param.rows(), param.cols(), 0.0);
    }

    Matrix& velocity_estimate = velocity[key];

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            velocity_estimate(i, j) = momentum * velocity_estimate(i, j) - learning_rate * grad(i, j);
            param(i, j) = param(i, j) + velocity_estimate(i, j);
        }
    }
}

// AdaGrad
void AdaGradOptimizer::step(Matrix& param, const Matrix& grad)
{
    const Matrix* key = &param;

    if (accumulated_squared_grads.find(key) == accumulated_squared_grads.end()) {
        accumulated_squared_grads[key] = Matrix(param.rows(), param.cols(), 0.0);
    }

    Matrix& accumulated_squared = accumulated_squared_grads[key];

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            accumulated_squared(i, j) = accumulated_squared(i, j) + grad(i, j) * grad(i, j);
            param(i, j) = param(i, j) - learning_rate * grad(i, j) / (std::sqrt(accumulated_squared(i, j)) + epsilon);
        }
    }
}

// RMSprop
void RMSpropOptimizer::step(Matrix& param, const Matrix& grad)
{
    const Matrix* key = &param;

    if (mean_squared_grads.find(key) == mean_squared_grads.end()) {
        mean_squared_grads[key] = Matrix(param.rows(), param.cols(), 0.0);
    }

    Matrix& mean_squared = mean_squared_grads[key];

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            mean_squared(i, j) = decay_rate * mean_squared(i, j) + (1.0 - decay_rate) * grad(i, j) * grad(i, j);
            param(i, j) = param(i, j) - learning_rate * grad(i, j) / (std::sqrt(mean_squared(i, j)) + epsilon);
        }
    }
}


std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, double lr)
{
   switch (type) {
       case OptimizerType::STOCHASTIC:
           return std::make_unique<StochasticOptimizer>(lr);
       case OptimizerType::MINI_BATCH:
           return std::make_unique<MiniBatchOptimizer>(lr);
       case OptimizerType::MOMENTUM:
           return std::make_unique<MomentumOptimizer>(lr);
       case OptimizerType::ADAGRAD:
           return std::make_unique<AdaGradOptimizer>(lr);
       case OptimizerType::RMSPROP:
           return std::make_unique<RMSpropOptimizer>(lr);
       case OptimizerType::ADAM:
           return std::make_unique<AdamOptimizer>(lr);
       default:
           return std::make_unique<BatchOptimizer>(lr);
   }
}
