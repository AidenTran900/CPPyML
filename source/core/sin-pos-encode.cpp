#include "ml_lib/core/sin-pos-encode.h"
#include <cmath>

SinPositionalEncoding::SinPositionalEncoding(int features, int max_seq_len)
{
    this->features = features;
    this->max_seq_len = max_seq_len;
    pe_matrix = Matrix(max_seq_len, features);

    // precompute positional encodings
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < features; i++) {
            double angle = pos / std::pow(period, (2.0 * (i / 2)) / features);
            if (i % 2 == 0) {
                pe_matrix(pos, i) = std::sin(angle);
            } else {
                pe_matrix(pos, i) = std::cos(angle);
            }
        }
    }
}

Matrix SinPositionalEncoding::forward(const Matrix &input)
{
    Matrix output(input.rows(), input.cols());

    // add positional encoding to input
    for (size_t i = 0; i < input.rows(); i++) {
        for (size_t j = 0; j < input.cols(); j++) {
            output(i, j) = input(i, j) + pe_matrix(i, j);
        }
    }
    return output;
}

