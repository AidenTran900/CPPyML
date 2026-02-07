#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>

template<typename T = double>
class RotPositionalEncoding {
    private:
        int features; // Size of the feature dimension (must be even)
        int max_seq_len; // Maximum sequence length
        double period = 10000.0; // Period for the rotation frequencies
        Matrix<T> cos_matrix; // Cosine values (max_seq_len x features/2)
        Matrix<T> sin_matrix; // Sine values (max_seq_len x features/2)
    public:
        RotPositionalEncoding(int features, int max_seq_len);

        Matrix<T> forward(const Matrix<T> &input);
};
