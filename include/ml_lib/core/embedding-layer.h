#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>
#include <string>

class EmbeddingLayer {
    private:
        int vocab_size;
        int embedding_dim;
        Matrix weights;
        Matrix grad_weights;
        std::vector<int> input_cache;

    public:
        EmbeddingLayer(int vocab_size, int embedding_dim);

        Matrix forward(const std::vector<int> &input);
        void backward(const Matrix &grad_output);
        void update(Optimizer *opt);
};