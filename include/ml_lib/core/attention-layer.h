#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>
#include <cmath>

template<typename T = double>
class AttentionLayer {
    private:
        int embed_dim;
        int num_heads;
        int num_kv_heads;
        int head_dim;
        int kv_dim;        // num_kv_heads * head_dim
        int heads_per_group; // num_heads / num_kv_heads

        Matrix<T> W_q;  // (embed_dim, embed_dim)
        Matrix<T> W_k;  // (embed_dim, kv_dim)
        Matrix<T> W_v;  // (embed_dim, kv_dim)
        Matrix<T> W_o;  // (embed_dim, embed_dim)

        Matrix<T> grad_W_q;
        Matrix<T> grad_W_k;
        Matrix<T> grad_W_v;
        Matrix<T> grad_W_o;

        Matrix<T> input_cache;
        Matrix<T> Q_cache;   // (seq_len, embed_dim)
        Matrix<T> K_cache;   // (seq_len, kv_dim)
        Matrix<T> V_cache;   // (seq_len, kv_dim)
        std::vector<Matrix<T>> attention_weights_cache;

        Matrix<T> kv_K_cache;
        Matrix<T> kv_V_cache;

        // Optional RoPE: precomputed cos/sin tables (max_seq_len x head_dim/2)
        bool rope_enabled = false;
        Matrix<T> rope_cos;
        Matrix<T> rope_sin;
        int cached_pos = 0;

        void applyRoPE(Matrix<T>& Q, Matrix<T>& K, int start_pos);

    public:
        AttentionLayer(int embed_dim, int num_heads);
        AttentionLayer(int embed_dim, int num_heads, int num_kv_heads);

        void enableRoPE(int max_seq_len, double theta = 10000.0);

        Matrix<T> forward(const Matrix<T>& input);
        Matrix<T> forward_cached(const Matrix<T>& input);
        void clear_cache();
        Matrix<T> backward(const Matrix<T>& grad_output);
        void update(Optimizer<T>* opt);

        void loadWeights(const Matrix<T>& q, const Matrix<T>& k,
                         const Matrix<T>& v, const Matrix<T>& o) {
            W_q = q; W_k = k; W_v = v; W_o = o;
        }
};
