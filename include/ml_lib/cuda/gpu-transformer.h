#pragma once
#ifdef ML_USE_CUDA

#include "cuda-matrix.h"
#include "../core/transformer-config.h"
#include <vector>
#include <memory>

template<typename T> class Transformer;
template<typename T> class TokenSampler;

template<typename T>
struct GpuLayerWeights {
    CudaMatrix<T> W_q, W_k, W_v, W_o;
    CudaMatrix<T> rms1_gamma, rms2_gamma;
    CudaMatrix<T> ff_gate_w, ff_up_w, ff_down_w;
    CudaMatrix<T> rope_cos, rope_sin;
};

template<typename T>
struct GpuTransformerWeights {
    CudaMatrix<T> token_embedding;
    std::vector<GpuLayerWeights<T>> layers;
    CudaMatrix<T> output_rms_gamma;
    CudaMatrix<T> output_proj_w;
};

template<typename T>
struct GpuKVCache {
    struct LayerCache {
        CudaMatrix<T> K;  // pre-allocated (max_seq_len, kv_dim)
        CudaMatrix<T> V;
    };
    std::vector<LayerCache> layers;
    int len = 0;
    void clear() { len = 0; }
};

template<typename T>
struct GpuScratch {
    CudaMatrix<T> x;           // (max_seq_len, embed_dim)
    CudaMatrix<T> norm_out;    // (max_seq_len, embed_dim)
    CudaMatrix<T> Q;           // (max_seq_len, embed_dim)
    CudaMatrix<T> K;           // (max_seq_len, kv_dim)
    CudaMatrix<T> V;           // (max_seq_len, kv_dim)
    CudaMatrix<T> attn_out;    // (max_seq_len, embed_dim)
    CudaMatrix<T> scores;      // (max_seq_len, max_seq_len) per head (reused)
    CudaMatrix<T> gate_out;    // (max_seq_len, ff_dim)
    CudaMatrix<T> up_out;      // (max_seq_len, ff_dim)
    CudaMatrix<T> silu_out;    // (max_seq_len, ff_dim)
    CudaMatrix<T> ffn_out;     // (max_seq_len, embed_dim)
    CudaMatrix<T> logits;      // (1, vocab_size)
    CudaMatrix<T> last_row;    // (1, embed_dim)

    void allocate(int max_seq_len, int embed_dim, int ff_dim, int kv_dim, int vocab_size);
};

// Transfer CPU model weights to GPU (with optional type conversion, e.g. float→__half)
template<typename GpuT, typename CpuT>
GpuTransformerWeights<GpuT> transfer_weights_to_gpu(Transformer<CpuT>& model);

// GPU forward pass — batched prefill
template<typename T>
void gpu_forward_prefill(const std::vector<int>& tokens,
                         const GpuTransformerWeights<T>& weights,
                         GpuKVCache<T>& kv_cache,
                         GpuScratch<T>& scratch,
                         int num_heads, int num_kv_heads, int head_dim,
                         int embed_dim, int ff_dim, float rms_eps);

// GPU forward pass — single token decode
template<typename T>
void gpu_forward_cached(int token,
                        const GpuTransformerWeights<T>& weights,
                        GpuKVCache<T>& kv_cache,
                        GpuScratch<T>& scratch,
                        int num_heads, int num_kv_heads, int head_dim,
                        int embed_dim, int ff_dim, float rms_eps);

#endif // ML_USE_CUDA
