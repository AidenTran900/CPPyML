#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>
#include <string>

template<typename T = double>
class EmbeddingLayer {
    private:
        int vocab_size;
        int embedding_dim;
        Matrix<T> weights;
        Matrix<T> grad_weights;
        std::vector<int> input_cache;

    public:
        EmbeddingLayer(int vocab_size, int embedding_dim);

        Matrix<T> forward(const std::vector<int> &input);
        void backward(const Matrix<T> &grad_output);
        void update(Optimizer<T> *opt);

        void loadWeights(const Matrix<T>& w) { weights = w; }

#ifdef ML_USE_CUDA
        const Matrix<T>& getWeights() const { return weights; }
#endif
};
