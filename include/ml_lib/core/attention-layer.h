#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>
#include <cmath>

template<typename T = double>
class AttentionLayer {
    private:
        int embed_dim;      // Input embedding dimension
        int num_heads;      // Number of attention heads
        int head_dim;       // Dimension per head (embed_dim / num_heads)

        // Projection weights for Q, K, V
        Matrix<T> W_q;
        Matrix<T> W_k;
        Matrix<T> W_v;
        Matrix<T> W_o;  // Output projection

        // Gradients
        Matrix<T> grad_W_q;
        Matrix<T> grad_W_k;
        Matrix<T> grad_W_v;
        Matrix<T> grad_W_o;

        // Cached values for backward pass
        Matrix<T> input_cache;
        Matrix<T> Q_cache;
        Matrix<T> K_cache;
        Matrix<T> V_cache;
        std::vector<Matrix<T>> attention_weights_cache;

        // KV cache for inference
        Matrix<T> kv_K_cache;
        Matrix<T> kv_V_cache;

    public:
        AttentionLayer(int embed_dim, int num_heads);

        Matrix<T> forward(const Matrix<T>& input);
        Matrix<T> forward_cached(const Matrix<T>& input);
        void clear_cache();
        Matrix<T> backward(const Matrix<T>& grad_output);
        void update(Optimizer<T>* opt);
};
