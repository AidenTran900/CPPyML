#include "ml_lib/core/rms-norm.h"
#include <cmath>

RMSNorm::RMSNorm(int features, double epsilon)
{
    this->features = features;
    this->epsilon = epsilon;
    gamma = Matrix(features, 1, 1.0);
    grad_gamma = Matrix(features, 1, 0.0);
}

Matrix RMSNorm::forward(const Matrix &input)
{
    int rows = input.rows();
    int cols = input.cols();

    normalized_cache = Matrix(rows, cols);
    rms_cache.resize(rows);
    Matrix output(rows, cols);

    for (int i = 0; i < rows; i++) {
        double sum_squared = 0.0;
        for (int j = 0; j < cols; j++) {
            sum_squared += input(i, j) * input(i, j);
        }
        sum_squared /= cols;

        double rms = std::sqrt(sum_squared + epsilon);
        rms_cache[i] = rms;

        for (int j = 0; j < cols; j++) {
            double normalized = input(i, j) / rms;
            normalized_cache(i, j) = normalized;
            output(i, j) = normalized * gamma(j, 0);
        }
    }
    return output;
}

Matrix RMSNorm::backward(const Matrix &grad_output)
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
        }

        // compute means
        double mean_gn_x_n = 0.0;

        for (int j = 0; j < cols; j++) {
            mean_gn_x_n += grad_norm[j] * normalized_cache(i, j);
        }
        mean_gn_x_n /= cols;

        // grad input
        for (int j = 0; j < cols; j++) {
            grad_input(i, j) = (grad_norm[j] - normalized_cache(i, j) * mean_gn_x_n) / rms_cache[i];
        }
    }
    return grad_input;
}

void RMSNorm::update(Optimizer *opt)
{
    opt->step(gamma, grad_gamma);

    // reset gradient
    grad_gamma = Matrix(features, 1, 0.0);
}
