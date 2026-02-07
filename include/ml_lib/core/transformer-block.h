#pragma once
#include "../math/matrix.h"
#include "attention-layer.h"
#include "layer-norm.h"
#include "neural-network-layer.h"
#include "optimizer.h"

template<typename T = double>
class TransformerBlock {
    private:
        AttentionLayer<T> attention;
        LayerNorm<T> norm1;
        LayerNorm<T> norm2;
        NeuralNetworkLayer<T> ff1;
        NeuralNetworkLayer<T> ff2;

        Matrix<T> attention_input_cache;
        Matrix<T> ff_input_cache;

    public:
        TransformerBlock(int embed_dim, int num_heads, int ff_dim);

        Matrix<T> forward(const Matrix<T>& input);
        Matrix<T> forward_cached(const Matrix<T>& input);
        void clear_cache();
        Matrix<T> backward(const Matrix<T>& grad_output);
        void update(Optimizer<T>* opt);
};
