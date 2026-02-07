#include "ml_lib/core/rms-norm.h"
#include <cmath>

template<typename T>
RMSNorm<T>::RMSNorm(int features, double epsilon)
{
    this->features = features;
    this->epsilon = epsilon;
    gamma = Matrix<T>(features, 1, 1.0);
    grad_gamma = Matrix<T>(features, 1, 0.0);
}

template<typename T>
Matrix<T> RMSNorm<T>::forward(const Matrix<T> &input)
{
    int rows = input.rows();
    int cols = input.cols();

    normalized_cache = Matrix<T>(rows, cols);
    rms_cache.resize(rows);
    Matrix<T> output(rows, cols);

    for (int i = 0; i < rows; i++) {
        T sum_squared = static_cast<T>(0.0);
        for (int j = 0; j < cols; j++) {
            sum_squared += input(i, j) * input(i, j);
        }
        sum_squared /= static_cast<T>(cols);

        T rms = static_cast<T>(std::sqrt(static_cast<double>(sum_squared) + epsilon));
        rms_cache[i] = rms;

        for (int j = 0; j < cols; j++) {
            T normalized = input(i, j) / rms;
            normalized_cache(i, j) = normalized;
            output(i, j) = normalized * gamma(j, 0);
        }
    }
    return output;
}

template<typename T>
Matrix<T> RMSNorm<T>::backward(const Matrix<T> &grad_output)
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
        }

        // compute means
        T mean_gn_x_n = static_cast<T>(0.0);

        for (int j = 0; j < cols; j++) {
            mean_gn_x_n += grad_norm[j] * normalized_cache(i, j);
        }
        mean_gn_x_n /= static_cast<T>(cols);

        // grad input
        for (int j = 0; j < cols; j++) {
            grad_input(i, j) = (grad_norm[j] - normalized_cache(i, j) * mean_gn_x_n) / rms_cache[i];
        }
    }
    return grad_input;
}

template<typename T>
void RMSNorm<T>::update(Optimizer<T> *opt)
{
    opt->step(gamma, grad_gamma);

    // reset gradient
    grad_gamma = Matrix<T>(features, 1, 0.0);
}

template class RMSNorm<float>;
template class RMSNorm<double>;
