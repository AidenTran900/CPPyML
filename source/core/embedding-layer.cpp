#include "ml_lib/core/embedding-layer.h"

EmbeddingLayer::EmbeddingLayer(int vocab_size, int embedding_dim)
{
    this->vocab_size = vocab_size;
    this->embedding_dim = embedding_dim;
    weights = Matrix(vocab_size, embedding_dim);
    grad_weights = Matrix(vocab_size, embedding_dim);

    //randomly initialize weights
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            weights(i, j) = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        }
    }
}

Matrix EmbeddingLayer::forward(const std::vector<int> &input)
{
    input_cache = input;
    Matrix output(input.size(), embedding_dim);

    for (size_t i = 0; i < input.size(); ++i) {
        for (int j = 0; j < embedding_dim; j++) {
            output(i, j) = weights(input[i], j);
        }
    }
    return output;
}

void EmbeddingLayer::backward(const Matrix &grad_output)
{
    grad_weights = Matrix(vocab_size, embedding_dim);
    for (size_t i = 0; i < grad_output.rows(); ++i) {
        int token_id = input_cache[i];
        for (int j = 0; j < embedding_dim; ++j) {
            grad_weights(token_id, j) += grad_output(i, j);
        }
    }
}

void EmbeddingLayer::update(Optimizer *opt)
{
    opt->step(weights, grad_weights);
}
