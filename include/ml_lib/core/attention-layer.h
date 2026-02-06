#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>
#include <cmath>

class AttentionLayer {
    private:
        int embed_dim;      // Input embedding dimension
        int num_heads;      // Number of attention heads
        int head_dim;       // Dimension per head (embed_dim / num_heads)

        // Projection weights for Q, K, V
        Matrix W_q;
        Matrix W_k;
        Matrix W_v;
        Matrix W_o;  // Output projection

        // Gradients
        Matrix grad_W_q;
        Matrix grad_W_k;
        Matrix grad_W_v;
        Matrix grad_W_o;

        // Cached values for backward pass
        Matrix input_cache;
        Matrix Q_cache;
        Matrix K_cache;
        Matrix V_cache;
        std::vector<Matrix> attention_weights_cache;

        // KV cache for inference
        Matrix kv_K_cache;
        Matrix kv_V_cache;

    public:
        AttentionLayer(int embed_dim, int num_heads);

        Matrix forward(const Matrix& input);
        Matrix forward_cached(const Matrix& input);
        void clear_cache();
        Matrix backward(const Matrix& grad_output);
        void update(Optimizer* opt);
};
