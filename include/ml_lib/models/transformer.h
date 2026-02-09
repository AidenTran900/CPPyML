#pragma once
#include "../math/matrix.h"
#include "../core/embedding-layer.h"
#include "../core/sin-pos-encode.h"
#include "../core/transformer-block.h"
#include "../core/transformer-config.h"
#include "../core/neural-network-layer.h"
#include "../core/layer-norm.h"
#include "../core/rms-norm.h"
#include "../core/token-sampler.h"
#include "gradient-model.h"
#include <vector>
#include <memory>

#ifdef ML_USE_CUDA
#include "../cuda/gpu-transformer.h"
#endif

template<typename T = double>
class Transformer : public GradientModel<T> {
    private:
        int vocab_size;
        int embed_dim;
        int max_seq_len;
        PosEncType pos_enc_type;

        EmbeddingLayer<T> embedding;
        std::unique_ptr<SinPositionalEncoding<T>> pos_encoding;
        std::vector<std::shared_ptr<TransformerBlock<T>>> blocks;
        NeuralNetworkLayer<T> output_projection;

        // Optional output norm (before output_projection)
        std::unique_ptr<LayerNorm<T>> output_ln;
        std::unique_ptr<RMSNorm<T>> output_rms;

        std::vector<int> last_token_input;
        Matrix<T> last_logits;
        std::vector<int> stop_tokens;

    public:
        // Legacy constructor (SINUSOIDAL, POST_NORM, LAYER_NORM, STANDARD FFN)
        Transformer(int vocab_size, int embed_dim, int num_heads,
                    int num_layers, int ff_dim, int max_seq_len,
                    std::unique_ptr<LossFunction<T>> loss,
                    std::unique_ptr<Optimizer<T>> opt,
                    std::unique_ptr<Regularizer<T>> reg);

        // Config-based constructor
        Transformer(const TransformerConfig& config,
                    std::unique_ptr<LossFunction<T>> loss,
                    std::unique_ptr<Optimizer<T>> opt,
                    std::unique_ptr<Regularizer<T>> reg);

        Matrix<T> forward(const Matrix<T>& X) override;
        Matrix<T> forward(const std::vector<int>& tokens);
        void backward(const Matrix<T>& y_true) override;
        void update() override;

        std::vector<int> generate(const std::vector<int>& prompt, int max_tokens);
        std::vector<int> generate(const std::vector<int>& prompt, int max_tokens,
                                  const TokenSampler<T>& sampler);
        void clear_cache();
        void setEosToken(int id) { stop_tokens = {id}; }
        void addStopToken(int id) { stop_tokens.push_back(id); }

        EmbeddingLayer<T>& getEmbedding() { return embedding; }
        NeuralNetworkLayer<T>& getOutputProjection() { return output_projection; }
        std::vector<std::shared_ptr<TransformerBlock<T>>>& getBlocks() { return blocks; }

        // Output norm accessors (for weight loading)
        LayerNorm<T>* getOutputLayerNorm() { return output_ln.get(); }
        RMSNorm<T>* getOutputRMSNorm() { return output_rms.get(); }

#ifdef ML_USE_CUDA
    private:
        std::unique_ptr<GpuTransformerWeights<__half>> gpu_weights_;
        std::unique_ptr<GpuKVCache<__half>> gpu_kv_cache_;
        std::unique_ptr<GpuScratch<__half>> gpu_scratch_;

    public:
        std::vector<int> generate_gpu(const std::vector<int>& prompt, int max_tokens);
        std::vector<int> generate_gpu(const std::vector<int>& prompt, int max_tokens,
                                       const TokenSampler<T>& sampler);
#endif
};
