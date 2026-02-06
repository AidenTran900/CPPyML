#pragma once
#include "../math/matrix.h"
#include "../core/embedding-layer.h"
#include "../core/sin-pos-encode.h"
#include "../core/transformer-block.h"
#include "../core/neural-network-layer.h"
#include "gradient-model.h"
#include <vector>
#include <memory>

class Transformer : public GradientModel {
    private:
        int vocab_size;
        int embed_dim;
        int max_seq_len;

        EmbeddingLayer embedding;
        SinPositionalEncoding pos_encoding;
        std::vector<std::shared_ptr<TransformerBlock>> blocks;
        NeuralNetworkLayer output_projection;

        std::vector<int> last_token_input;
        Matrix last_logits;

    public:
        Transformer(int vocab_size, int embed_dim, int num_heads,
                    int num_layers, int ff_dim, int max_seq_len,
                    std::unique_ptr<LossFunction> loss,
                    std::unique_ptr<Optimizer> opt,
                    std::unique_ptr<Regularizer> reg);

        Matrix forward(const Matrix& X) override;
        Matrix forward(const std::vector<int>& tokens);
        void backward(const Matrix& y_true) override;
        void update() override;
};
