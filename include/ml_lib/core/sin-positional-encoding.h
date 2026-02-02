#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>

class PositionalEncoding {
    private:
        int features; // Size of the feature dimension
        int max_seq_len; // Maximum sequence length
        Matrix pe_matrix; // Positional encoding matrix (max_seq_len x features)
    public:
        PositionalEncoding(int features, int max_seq_len);

        Matrix forward(const Matrix &input);
};