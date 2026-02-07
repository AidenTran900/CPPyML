#include "ml_lib/core/sin-pos-encode.h"
#include <cmath>

template<typename T>
SinPositionalEncoding<T>::SinPositionalEncoding(int features, int max_seq_len)
{
    this->features = features;
    this->max_seq_len = max_seq_len;
    pe_matrix = Matrix<T>(max_seq_len, features);

    // precompute positional encodings
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < features; i++) {
            double angle = pos / std::pow(period, (2.0 * (i / 2)) / features);
            if (i % 2 == 0) {
                pe_matrix(pos, i) = static_cast<T>(std::sin(angle));
            } else {
                pe_matrix(pos, i) = static_cast<T>(std::cos(angle));
            }
        }
    }
}

template<typename T>
Matrix<T> SinPositionalEncoding<T>::forward(const Matrix<T> &input)
{
    Matrix<T> output(input.rows(), input.cols());

    // add positional encoding to input
    for (size_t i = 0; i < input.rows(); i++) {
        for (size_t j = 0; j < input.cols(); j++) {
            output(i, j) = input(i, j) + pe_matrix(i, j);
        }
    }
    return output;
}

template<typename T>
Matrix<T> SinPositionalEncoding<T>::forward(const Matrix<T> &input, int position)
{
    Matrix<T> output(input.rows(), input.cols());

    for (int j = 0; j < input.cols(); j++) {
        output(0, j) = input(0, j) + pe_matrix(position, j);
    }
    return output;
}

template class SinPositionalEncoding<float>;
template class SinPositionalEncoding<double>;
