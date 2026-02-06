#include "ml_lib/core/rotational-pos-encode.h"
#include <cmath>
#include <stdexcept>

RotPositionalEncoding::RotPositionalEncoding(int features, int max_seq_len)
{
    if (features % 2 != 0) {
        throw std::invalid_argument("Features must be even for rotational positional encoding");
    }

    this->features = features;
    this->max_seq_len = max_seq_len;

    int half_features = features / 2;
    cos_matrix = Matrix(max_seq_len, half_features);
    sin_matrix = Matrix(max_seq_len, half_features);

    // precompute positional encodings
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_features; i++) {
            double theta = pos / std::pow(period, (2.0 * i) / features);
            cos_matrix(pos, i) = std::cos(theta);
            sin_matrix(pos, i) = std::sin(theta);
        }
    }
}

Matrix RotPositionalEncoding::forward(const Matrix &input)
{
    Matrix output(input.rows(), input.cols());
    int half_features = features / 2;

    // apply rotation to each position
    for (int pos = 0; pos < input.rows(); pos++) {
        for (int i = 0; i < half_features; i++) {
            double cos_val = cos_matrix(pos, i);
            double sin_val = sin_matrix(pos, i);

            double x = input(pos, 2 * i);
            double y = input(pos, 2 * i + 1);

            output(pos, 2 * i)     = x * cos_val - y * sin_val;
            output(pos, 2 * i + 1) = x * sin_val + y * cos_val;
        }
    }

    return output;
}
