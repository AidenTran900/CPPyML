#include "ml_lib/core/embedding-layer.h"

template<typename T>
EmbeddingLayer<T>::EmbeddingLayer(int vocab_size, int embedding_dim)
{
    this->vocab_size = vocab_size;
    this->embedding_dim = embedding_dim;
    weights = Matrix<T>(vocab_size, embedding_dim);
    grad_weights = Matrix<T>(vocab_size, embedding_dim);

    //randomly initialize weights
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            weights(i, j) = static_cast<T>(((double)rand() / RAND_MAX) * 0.2 - 0.1);
        }
    }
}

template<typename T>
Matrix<T> EmbeddingLayer<T>::forward(const std::vector<int> &input)
{
    input_cache = input;
    Matrix<T> output(input.size(), embedding_dim);

    for (size_t i = 0; i < input.size(); ++i) {
        for (int j = 0; j < embedding_dim; j++) {
            output(i, j) = weights(input[i], j);
        }
    }
    return output;
}

template<typename T>
void EmbeddingLayer<T>::backward(const Matrix<T> &grad_output)
{
    grad_weights = Matrix<T>(vocab_size, embedding_dim);
    for (size_t i = 0; i < grad_output.rows(); ++i) {
        int token_id = input_cache[i];
        for (int j = 0; j < embedding_dim; ++j) {
            grad_weights(token_id, j) += grad_output(i, j);
        }
    }
}

template<typename T>
void EmbeddingLayer<T>::update(Optimizer<T> *opt)
{
    opt->step(weights, grad_weights);
}

template class EmbeddingLayer<float>;
template class EmbeddingLayer<double>;
