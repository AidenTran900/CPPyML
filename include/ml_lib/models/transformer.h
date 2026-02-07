#pragma once
#include "../math/matrix.h"
#include "../core/embedding-layer.h"
#include "../core/sin-pos-encode.h"
#include "../core/transformer-block.h"
#include "../core/neural-network-layer.h"
#include "gradient-model.h"
#include <vector>
#include <memory>

template<typename T = double>
class Transformer : public GradientModel<T> {
    private:
        int vocab_size;
        int embed_dim;
        int max_seq_len;

        EmbeddingLayer<T> embedding;
        SinPositionalEncoding<T> pos_encoding;
        std::vector<std::shared_ptr<TransformerBlock<T>>> blocks;
        NeuralNetworkLayer<T> output_projection;

        std::vector<int> last_token_input;
        Matrix<T> last_logits;

    public:
        Transformer(int vocab_size, int embed_dim, int num_heads,
                    int num_layers, int ff_dim, int max_seq_len,
                    std::unique_ptr<LossFunction<T>> loss,
                    std::unique_ptr<Optimizer<T>> opt,
                    std::unique_ptr<Regularizer<T>> reg);

        Matrix<T> forward(const Matrix<T>& X) override;
        Matrix<T> forward(const std::vector<int>& tokens);
        void backward(const Matrix<T>& y_true) override;
        void update() override;

        std::vector<int> generate(const std::vector<int>& prompt, int max_tokens);
        void clear_cache();
};
