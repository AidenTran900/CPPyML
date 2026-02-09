#pragma once
#ifdef ML_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cuda_kernels {

template<typename T>
void rms_norm(const T* input, const T* gamma, T* output,
              int rows, int cols, float epsilon, cudaStream_t stream = 0);

template<typename T>
void rope(T* Q, T* K, const T* cos_table, const T* sin_table,
          int seq_len, int num_heads, int num_kv_heads,
          int head_dim, int start_pos, cudaStream_t stream = 0);

template<typename T>
void softmax(const T* input, T* output, int rows, int cols, cudaStream_t stream = 0);

template<typename T>
void silu_hadamard(const T* gate, const T* up, T* output, int n, cudaStream_t stream = 0);

template<typename T>
void add_inplace(T* a, const T* b, int n, cudaStream_t stream = 0);

template<typename T>
void scale_causal_mask(T* scores, int rows, int cols, T scale, int offset, cudaStream_t stream = 0);

template<typename T>
void embedding_lookup(const int* token_ids, const T* table, T* output,
                      int num_tokens, int embed_dim, cudaStream_t stream = 0);

} // namespace cuda_kernels

#endif // ML_USE_CUDA
