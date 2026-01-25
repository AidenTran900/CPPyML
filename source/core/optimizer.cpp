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

   if (m.find(key) == m.end()) {
       m[key] = Matrix(param.rows(), param.cols(), 0.0);
       v[key] = Matrix(param.rows(), param.cols(), 0.0);
       t[key] = 0;
   }

   Matrix& m_t = m[key];
   Matrix& v_t = v[key];
   t[key]++;
   int timestep = t[key];

   for (int i = 0; i < param.rows(); i++) {
       for (int j = 0; j < param.cols(); j++) {
           m_t(i, j) = beta1 * m_t(i, j) + (1.0 - beta1) * grad(i, j);
           v_t(i, j) = beta2 * v_t(i, j) + (1.0 - beta2) * grad(i, j) * grad(i, j);

           double m_hat = m_t(i, j) / (1.0 - std::pow(beta1, timestep));
           double v_hat = v_t(i, j) / (1.0 - std::pow(beta2, timestep));

           param(i, j) = param(i, j) - learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
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
        case OptimizerType::ADAM:
           return std::make_unique<AdamOptimizer>(lr);
        default:
            return std::make_unique<StochasticOptimizer>(lr);
    }
}
