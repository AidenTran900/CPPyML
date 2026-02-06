#pragma once
#include "../math/matrix.h"
#include "attention-layer.h"
#include "layer-norm.h"
#include "neural-network-layer.h"
#include "optimizer.h"

class TransformerBlock {
    private:
        AttentionLayer attention;
        LayerNorm norm1;
        LayerNorm norm2;
        NeuralNetworkLayer ff1;
        NeuralNetworkLayer ff2;

        Matrix attention_input_cache;
        Matrix ff_input_cache;

    public:
        TransformerBlock(int embed_dim, int num_heads, int ff_dim);

        Matrix forward(const Matrix& input);
        Matrix backward(const Matrix& grad_output);
        void update(Optimizer* opt);
};
