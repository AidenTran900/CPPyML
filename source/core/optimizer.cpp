#include "ml_lib/core/optimizer.h"
#include <cmath>

// Stochastic
template<typename T>
void StochasticOptimizer<T>::step(Matrix<T>& param, const Matrix<T>& grad)
{
    param = param - grad * this->learning_rate;
}


// Mini Batch
template<typename T>
void MiniBatchOptimizer<T>::step(Matrix<T>& param, const Matrix<T>& grad)
{
    Matrix<T>* key = &param;
    if (accumulated_grads.find(key) == accumulated_grads.end()) {
        accumulated_grads[key] = Matrix<T>(grad.rows(), grad.cols(), 0.0);
    }
    accumulated_grads[key] = accumulated_grads[key] + grad;
    current_step++;

    if (current_step >= batch_size) {
        for (auto& [param_ptr, acc_grad] : accumulated_grads) {
            *param_ptr = *param_ptr - acc_grad * (this->learning_rate / batch_size);
        }
        accumulated_grads.clear();
        current_step = 0;
    }
}

// Adam
template<typename T>
void AdamOptimizer<T>::step(Matrix<T>& param, const Matrix<T>& grad)
{
    const Matrix<T>* key = &param;

    if (first_moment.find(key) == first_moment.end()) {
        first_moment[key] = Matrix<T>(param.rows(), param.cols(), 0.0);
        second_moment[key] = Matrix<T>(param.rows(), param.cols(), 0.0);
        timesteps[key] = 0;
    }

    Matrix<T>& first_moment_estimate = first_moment[key];
    Matrix<T>& second_moment_estimate = second_moment[key];
    timesteps[key]++;
    int current_timestep = timesteps[key];

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            double g = static_cast<double>(grad(i, j));
            double m = static_cast<double>(first_moment_estimate(i, j));
            double v = static_cast<double>(second_moment_estimate(i, j));

            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * g * g;

            first_moment_estimate(i, j) = static_cast<T>(m);
            second_moment_estimate(i, j) = static_cast<T>(v);

            double bias_corrected_first = m / (1.0 - std::pow(beta1, current_timestep));
            double bias_corrected_second = v / (1.0 - std::pow(beta2, current_timestep));

            param(i, j) = static_cast<T>(static_cast<double>(param(i, j)) - this->learning_rate * bias_corrected_first / (std::sqrt(bias_corrected_second) + epsilon));
        }
    }
}

// Momentum
template<typename T>
void MomentumOptimizer<T>::step(Matrix<T>& param, const Matrix<T>& grad)
{
    const Matrix<T>* key = &param;

    if (velocity.find(key) == velocity.end()) {
        velocity[key] = Matrix<T>(param.rows(), param.cols(), 0.0);
    }

    Matrix<T>& velocity_estimate = velocity[key];

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            double v = momentum * static_cast<double>(velocity_estimate(i, j)) - this->learning_rate * static_cast<double>(grad(i, j));
            velocity_estimate(i, j) = static_cast<T>(v);
            param(i, j) = static_cast<T>(static_cast<double>(param(i, j)) + v);
        }
    }
}

// AdaGrad
template<typename T>
void AdaGradOptimizer<T>::step(Matrix<T>& param, const Matrix<T>& grad)
{
    const Matrix<T>* key = &param;

    if (accumulated_squared_grads.find(key) == accumulated_squared_grads.end()) {
        accumulated_squared_grads[key] = Matrix<T>(param.rows(), param.cols(), 0.0);
    }

    Matrix<T>& accumulated_squared = accumulated_squared_grads[key];

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            double g = static_cast<double>(grad(i, j));
            double acc = static_cast<double>(accumulated_squared(i, j)) + g * g;
            accumulated_squared(i, j) = static_cast<T>(acc);
            param(i, j) = static_cast<T>(static_cast<double>(param(i, j)) - this->learning_rate * g / (std::sqrt(acc) + epsilon));
        }
    }
}

// RMSprop
template<typename T>
void RMSpropOptimizer<T>::step(Matrix<T>& param, const Matrix<T>& grad)
{
    const Matrix<T>* key = &param;

    if (mean_squared_grads.find(key) == mean_squared_grads.end()) {
        mean_squared_grads[key] = Matrix<T>(param.rows(), param.cols(), 0.0);
    }

    Matrix<T>& mean_squared = mean_squared_grads[key];

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            double g = static_cast<double>(grad(i, j));
            double ms = decay_rate * static_cast<double>(mean_squared(i, j)) + (1.0 - decay_rate) * g * g;
            mean_squared(i, j) = static_cast<T>(ms);
            param(i, j) = static_cast<T>(static_cast<double>(param(i, j)) - this->learning_rate * g / (std::sqrt(ms) + epsilon));
        }
    }
}


template<typename T>
std::unique_ptr<Optimizer<T>> createOptimizer(OptimizerType type, double lr)
{
   switch (type) {
       case OptimizerType::STOCHASTIC:
           return std::make_unique<StochasticOptimizer<T>>(lr);
       case OptimizerType::MINI_BATCH:
           return std::make_unique<MiniBatchOptimizer<T>>(lr);
       case OptimizerType::MOMENTUM:
           return std::make_unique<MomentumOptimizer<T>>(lr);
       case OptimizerType::ADAGRAD:
           return std::make_unique<AdaGradOptimizer<T>>(lr);
       case OptimizerType::RMSPROP:
           return std::make_unique<RMSpropOptimizer<T>>(lr);
       case OptimizerType::ADAM:
           return std::make_unique<AdamOptimizer<T>>(lr);
       default:
           return std::make_unique<StochasticOptimizer<T>>(lr);
   }
}

// Explicit template instantiation
template class StochasticOptimizer<float>;
template class StochasticOptimizer<double>;
template class MiniBatchOptimizer<float>;
template class MiniBatchOptimizer<double>;
template class AdamOptimizer<float>;
template class AdamOptimizer<double>;
template class MomentumOptimizer<float>;
template class MomentumOptimizer<double>;
template class AdaGradOptimizer<float>;
template class AdaGradOptimizer<double>;
template class RMSpropOptimizer<float>;
template class RMSpropOptimizer<double>;
template std::unique_ptr<Optimizer<float>> createOptimizer<float>(OptimizerType, double);
template std::unique_ptr<Optimizer<double>> createOptimizer<double>(OptimizerType, double);
