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
        int head_dim;

        Matrix<T> W_q;
        Matrix<T> W_k;
        Matrix<T> W_v;
        Matrix<T> W_o;

        Matrix<T> grad_W_q;
        Matrix<T> grad_W_k;
        Matrix<T> grad_W_v;
        Matrix<T> grad_W_o;

        Matrix<T> input_cache;
        Matrix<T> Q_cache;
        Matrix<T> K_cache;
        Matrix<T> V_cache;
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

        void enableRoPE(int max_seq_len);

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
