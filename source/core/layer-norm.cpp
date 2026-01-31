#include "ml_lib/core/layer-norm.h"
#include <cmath>

LayerNorm::LayerNorm(int features, double epsilon)
{
    this->features = features;
    this->epsilon = epsilon;
    gamma = Matrix(features, 1, 1.0);
    beta = Matrix(features, 1, 0.0);
    grad_gamma = Matrix(features, 1, 0.0);
    grad_beta = Matrix(features, 1, 0.0);
}

Matrix LayerNorm::forward(const Matrix &input)
{
    int rows = input.rows();
    int cols = input.cols();

    normalized_cache = Matrix(rows, cols);
    std_cache.resize(rows);
    Matrix output(rows, cols);

    for (int i = 0; i < rows; i++) {
        double mean = 0.0;
        for (int j = 0; j < cols; j++) {
            mean += input(i, j);
        }
        mean /= cols;

        double variance = 0.0;
        for (int j = 0; j < cols; j++) {
            variance += (input(i, j) - mean) * (input(i, j) - mean);
        }
        variance /= cols;

        double std = std::sqrt(variance + epsilon);
        std_cache[i] = std;

        for (int j = 0; j < cols; j++) {
            double normalized = (input(i, j) - mean) / std;
            normalized_cache(i, j) = normalized;
            output(i, j) = normalized * gamma(j, 0) + beta(j, 0);
        }
    }
    return output;
}

Matrix LayerNorm::backward(const Matrix &grad_output)
{
    int rows = grad_output.rows();
    int cols = grad_output.cols();
    Matrix grad_input(rows, cols);

    for (int i = 0; i < rows; i++) {
        // normalized gradients
        std::vector<double> grad_norm(cols);
        for (int j = 0; j < cols; j++) {
            grad_norm[j] = grad_output(i, j) * gamma(j, 0);
        }

        // accumulate parameter gradients
        for (int j = 0; j < cols; j++) {
            grad_gamma(j, 0) += grad_output(i, j) * normalized_cache(i, j);
            grad_beta(j, 0) += grad_output(i, j);
        }

        // compute means
        double mean_gn = 0.0;
        double mean_gn_x_n = 0.0;

        for (int j = 0; j < cols; j++) {
            mean_gn += grad_norm[j];
            mean_gn_x_n += grad_norm[j] * normalized_cache(i, j);
        }
        mean_gn /= cols;
        mean_gn_x_n /= cols;

        // grad input
        for (int j = 0; j < cols; j++) {
            grad_input(i, j) = (grad_norm[j] - mean_gn - normalized_cache(i, j) * mean_gn_x_n) / std_cache[i];
        }
    }
    return grad_input;
}

void LayerNorm::update(Optimizer *opt)
{
    opt->step(gamma, grad_gamma);
    opt->step(beta, grad_beta);

    // reset gradients
    grad_gamma = Matrix(features, 1, 0.0);
    grad_beta = Matrix(features, 1, 0.0);
}
