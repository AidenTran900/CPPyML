#include "attention-layer.h"
#include "softmax.h"
#include <cmath>
#include <stdexcept>

AttentionLayer::AttentionLayer(int embed_dim, int num_heads)
{
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }

    this->embed_dim = embed_dim;
    this->num_heads = num_heads;
    this->head_dim = embed_dim / num_heads;

    W_q = Matrix(embed_dim, embed_dim);
    W_k = Matrix(embed_dim, embed_dim);
    W_v = Matrix(embed_dim, embed_dim);
    W_o = Matrix(embed_dim, embed_dim);

    grad_W_q = Matrix(embed_dim, embed_dim);
    grad_W_k = Matrix(embed_dim, embed_dim);
    grad_W_v = Matrix(embed_dim, embed_dim);
    grad_W_o = Matrix(embed_dim, embed_dim);

    double scale = std::sqrt(2.0 / (embed_dim + embed_dim));
    for (int i = 0; i < embed_dim; i++) {
        for (int j = 0; j < embed_dim; j++) {
            W_q(i, j) = ((double)rand() / RAND_MAX * 2 - 1) * scale;
            W_k(i, j) = ((double)rand() / RAND_MAX * 2 - 1) * scale;
            W_v(i, j) = ((double)rand() / RAND_MAX * 2 - 1) * scale;
            W_o(i, j) = ((double)rand() / RAND_MAX * 2 - 1) * scale;
        }
    }
}

Matrix AttentionLayer::forward(const Matrix& input)
{
    int seq_len = input.rows();

    input_cache = input;

    Q_cache = input * W_q; 
    K_cache = input * W_k;
    V_cache = input * W_v;

    // multi-head attention
    Matrix output(seq_len, embed_dim);
    attention_weights_cache.clear();
    attention_weights_cache.resize(num_heads);

    double scale = 1.0 / std::sqrt((double)head_dim);

    for (int h = 0; h < num_heads; h++) {
        int head_start = h * head_dim;

        Matrix Q_h(seq_len, head_dim);
        Matrix K_h(seq_len, head_dim);
        Matrix V_h(seq_len, head_dim);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                Q_h(i, j) = Q_cache(i, head_start + j);
                K_h(i, j) = K_cache(i, head_start + j);
                V_h(i, j) = V_cache(i, head_start + j);
            }
        }

        // compute attention scores
        Matrix K_h_T = K_h.transpose();
        Matrix scores = Q_h * K_h_T;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                scores(i, j) *= scale;
            }
        }

        Matrix attn_weights = Softmax::apply(scores);
        attention_weights_cache[h] = attn_weights;

        Matrix head_output = attn_weights * V_h;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                output(i, head_start + j) = head_output(i, j);
            }
        }
    }

    return output * W_o;
}

Matrix AttentionLayer::backward(const Matrix& grad_output)
{
    int seq_len = grad_output.rows();
    double scale = 1.0 / std::sqrt((double)head_dim);

    // distribute error and update output projection
    Matrix grad_concat = grad_output * W_o.transpose();

    Matrix concat_output(seq_len, embed_dim);
    for (int h = 0; h < num_heads; h++) {
        int head_start = h * head_dim;
        Matrix V_h(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                V_h(i, j) = V_cache(i, head_start + j);
            }
        }
        Matrix head_output = attention_weights_cache[h] * V_h;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                concat_output(i, head_start + j) = head_output(i, j);
            }
        }
    }
    grad_W_o = concat_output.transpose() * grad_output;

    Matrix grad_Q(seq_len, embed_dim);
    Matrix grad_K(seq_len, embed_dim);
    Matrix grad_V(seq_len, embed_dim);

    for (int h = 0; h < num_heads; h++) {
        int head_start = h * head_dim;

        Matrix grad_head(seq_len, head_dim);
        Matrix Q_h(seq_len, head_dim);
        Matrix K_h(seq_len, head_dim);
        Matrix V_h(seq_len, head_dim);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                grad_head(i, j) = grad_concat(i, head_start + j);
                Q_h(i, j) = Q_cache(i, head_start + j);
                K_h(i, j) = K_cache(i, head_start + j);
                V_h(i, j) = V_cache(i, head_start + j);
            }
        }

        Matrix attn_weights = attention_weights_cache[h];

        // gradient V
        // paid attenttion to the right word but info was useless
        Matrix grad_V_h = attn_weights.transpose() * grad_head;  

        // gradient attention
        // paid attention to the wrong word
        Matrix grad_attn = grad_head * V_h.transpose();

        // gradient softmax
        // too much attention paid to one word
        Matrix grad_scores = Softmax::derivative(attn_weights, grad_attn);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                grad_scores(i, j) *= scale;
            }
        }

        // gradient Q
        // looked at the wrong key
        Matrix grad_Q_h = grad_scores * K_h;

        // gradient K
        // key representation was poor
        Matrix grad_K_h = grad_scores.transpose() * Q_h; 

        // accumulate into gradients
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                grad_Q(i, head_start + j) = grad_Q_h(i, j);
                grad_K(i, head_start + j) = grad_K_h(i, j);
                grad_V(i, head_start + j) = grad_V_h(i, j);
            }
        }
    }

    grad_W_q = input_cache.transpose() * grad_Q;
    grad_W_k = input_cache.transpose() * grad_K;
    grad_W_v = input_cache.transpose() * grad_V;

    return grad_Q * W_q.transpose() + grad_K * W_k.transpose() + grad_V * W_v.transpose();
}

void AttentionLayer::update(Optimizer* opt)
{
    opt->step(W_q, grad_W_q);
    opt->step(W_k, grad_W_k);
    opt->step(W_v, grad_W_v);
    opt->step(W_o, grad_W_o);

    grad_W_q = Matrix(embed_dim, embed_dim);
    grad_W_k = Matrix(embed_dim, embed_dim);
    grad_W_v = Matrix(embed_dim, embed_dim);
    grad_W_o = Matrix(embed_dim, embed_dim);
}
