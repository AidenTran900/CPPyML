#pragma once
#include "../math/matrix.h"
#include "attention-layer.h"
#include "layer-norm.h"
#include "rms-norm.h"
#include "neural-network-layer.h"
#include "transformer-config.h"
#include "optimizer.h"
#include <memory>

template<typename T = double>
class TransformerBlock {
    private:
        NormType norm_type;
        FFNType ffn_type;
        NormPosition norm_position;

        AttentionLayer<T> attention;

        // Norms
        std::unique_ptr<LayerNorm<T>> ln1, ln2;
        std::unique_ptr<RMSNorm<T>> rms1, rms2;

        // FFN layers
        NeuralNetworkLayer<T> ff1;   // embed_dim → ff_dim (up/standard)
        NeuralNetworkLayer<T> ff2;   // ff_dim → embed_dim (down)
        std::unique_ptr<NeuralNetworkLayer<T>> ff_gate; // embed_dim → ff_dim (gated only)

        // Caches
        Matrix<T> attention_input_cache;
        Matrix<T> ff_input_cache;
        Matrix<T> gate_cache;
        Matrix<T> up_cache;

        // Internal dispatch
        Matrix<T> normForward1(const Matrix<T>& x);
        Matrix<T> normForward2(const Matrix<T>& x);
        Matrix<T> normBackward1(const Matrix<T>& g);
        Matrix<T> normBackward2(const Matrix<T>& g);
        Matrix<T> ffnForward(const Matrix<T>& x);
        Matrix<T> ffnBackward(const Matrix<T>& g);

    public:
        // Legacy constructor (POST_NORM, LAYER_NORM, STANDARD FFN)
        TransformerBlock(int embed_dim, int num_heads, int ff_dim);

        // Configurable constructor
        TransformerBlock(int embed_dim, int num_heads, int ff_dim,
                         NormType norm_type, FFNType ffn_type,
                         NormPosition norm_position);

        Matrix<T> forward(const Matrix<T>& input);
        Matrix<T> forward_cached(const Matrix<T>& input);
        void clear_cache();
        Matrix<T> backward(const Matrix<T>& grad_output);
        void update(Optimizer<T>* opt);

        // Accessors
        AttentionLayer<T>& getAttention() { return attention; }
        NeuralNetworkLayer<T>& getFF1() { return ff1; }
        NeuralNetworkLayer<T>& getFF2() { return ff2; }

        // Config-aware accessors
        NormType getNormType() const { return norm_type; }
        FFNType getFFNType() const { return ffn_type; }
        LayerNorm<T>* getLayerNorm1() { return ln1.get(); }
        LayerNorm<T>* getLayerNorm2() { return ln2.get(); }
        RMSNorm<T>* getRMSNorm1() { return rms1.get(); }
        RMSNorm<T>* getRMSNorm2() { return rms2.get(); }
        NeuralNetworkLayer<T>* getFFGate() { return ff_gate.get(); }
};
