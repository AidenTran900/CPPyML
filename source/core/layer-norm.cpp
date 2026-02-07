#include "ml_lib/core/layer-norm.h"
#include <cmath>

template<typename T>
LayerNorm<T>::LayerNorm(int features, double epsilon)
{
    this->features = features;
    this->epsilon = epsilon;
    gamma = Matrix<T>(features, 1, 1.0);
    beta = Matrix<T>(features, 1, 0.0);
    grad_gamma = Matrix<T>(features, 1, 0.0);
    grad_beta = Matrix<T>(features, 1, 0.0);
}

template<typename T>
Matrix<T> LayerNorm<T>::forward(const Matrix<T> &input)
{
    int rows = input.rows();
    int cols = input.cols();

    normalized_cache = Matrix<T>(rows, cols);
    std_cache.resize(rows);
    Matrix<T> output(rows, cols);

    for (int i = 0; i < rows; i++) {
        T mean = static_cast<T>(0.0);
        for (int j = 0; j < cols; j++) {
            mean += input(i, j);
        }
        mean /= static_cast<T>(cols);

        T variance = static_cast<T>(0.0);
        for (int j = 0; j < cols; j++) {
            variance += (input(i, j) - mean) * (input(i, j) - mean);
        }
        variance /= static_cast<T>(cols);

        T std = static_cast<T>(std::sqrt(static_cast<double>(variance) + epsilon));
        std_cache[i] = std;

        for (int j = 0; j < cols; j++) {
            T normalized = (input(i, j) - mean) / std;
            normalized_cache(i, j) = normalized;
            output(i, j) = normalized * gamma(j, 0) + beta(j, 0);
        }
    }
    return output;
}

template<typename T>
Matrix<T> LayerNorm<T>::backward(const Matrix<T> &grad_output)
{
    int rows = grad_output.rows();
    int cols = grad_output.cols();
    Matrix<T> grad_input(rows, cols);

    for (int i = 0; i < rows; i++) {
        // normalized gradients
        std::vector<T> grad_norm(cols);
        for (int j = 0; j < cols; j++) {
            grad_norm[j] = grad_output(i, j) * gamma(j, 0);
        }

        // accumulate parameter gradients
        for (int j = 0; j < cols; j++) {
            grad_gamma(j, 0) += grad_output(i, j) * normalized_cache(i, j);
            grad_beta(j, 0) += grad_output(i, j);
        }

        // compute means
        T mean_gn = static_cast<T>(0.0);
        T mean_gn_x_n = static_cast<T>(0.0);

        for (int j = 0; j < cols; j++) {
            mean_gn += grad_norm[j];
            mean_gn_x_n += grad_norm[j] * normalized_cache(i, j);
        }
        mean_gn /= static_cast<T>(cols);
        mean_gn_x_n /= static_cast<T>(cols);

        // grad input
        for (int j = 0; j < cols; j++) {
            grad_input(i, j) = (grad_norm[j] - mean_gn - normalized_cache(i, j) * mean_gn_x_n) / std_cache[i];
        }
    }
    return grad_input;
}

template<typename T>
void LayerNorm<T>::update(Optimizer<T> *opt)
{
    opt->step(gamma, grad_gamma);
    opt->step(beta, grad_beta);

    // reset gradients
    grad_gamma = Matrix<T>(features, 1, 0.0);
    grad_beta = Matrix<T>(features, 1, 0.0);
}

template class LayerNorm<float>;
template class LayerNorm<double>;
