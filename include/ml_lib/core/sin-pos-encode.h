#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>

template<typename T = double>
class SinPositionalEncoding {
    private:
        int features; // Size of the feature dimension
        int max_seq_len; // Maximum sequence length
        double period = 10000.0; // Period for the sine and cosine functions
        Matrix<T> pe_matrix; // Positional encoding matrix (max_seq_len x features)
    public:
        SinPositionalEncoding(int features, int max_seq_len);

        Matrix<T> forward(const Matrix<T> &input);
        Matrix<T> forward(const Matrix<T> &input, int position);
};
