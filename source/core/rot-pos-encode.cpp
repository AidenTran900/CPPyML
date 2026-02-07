#include "ml_lib/core/rotational-pos-encode.h"
#include <cmath>
#include <stdexcept>

template<typename T>
RotPositionalEncoding<T>::RotPositionalEncoding(int features, int max_seq_len)
{
    if (features % 2 != 0) {
        throw std::invalid_argument("Features must be even for rotational positional encoding");
    }

    this->features = features;
    this->max_seq_len = max_seq_len;

    int half_features = features / 2;
    cos_matrix = Matrix<T>(max_seq_len, half_features);
    sin_matrix = Matrix<T>(max_seq_len, half_features);

    // precompute positional encodings
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_features; i++) {
            double theta = pos / std::pow(period, (2.0 * i) / features);
            cos_matrix(pos, i) = static_cast<T>(std::cos(theta));
            sin_matrix(pos, i) = static_cast<T>(std::sin(theta));
        }
    }
}

template<typename T>
Matrix<T> RotPositionalEncoding<T>::forward(const Matrix<T> &input)
{
    Matrix<T> output(input.rows(), input.cols());
    int half_features = features / 2;

    // apply rotation to each position
    for (int pos = 0; pos < input.rows(); pos++) {
        for (int i = 0; i < half_features; i++) {
            T cos_val = cos_matrix(pos, i);
            T sin_val = sin_matrix(pos, i);

            T x = input(pos, 2 * i);
            T y = input(pos, 2 * i + 1);

            output(pos, 2 * i)     = x * cos_val - y * sin_val;
            output(pos, 2 * i + 1) = x * sin_val + y * cos_val;
        }
    }

    return output;
}

template class RotPositionalEncoding<float>;
template class RotPositionalEncoding<double>;
