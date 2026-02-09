#pragma once
#ifdef ML_USE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <cstddef>

template<typename T> class Matrix;  // forward declare CPU Matrix

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e)); \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) \
        throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(s))); \
} while(0)

template<typename T>
class CudaMatrix {
private:
    T* d_data = nullptr;
    int m_rows = 0;
    int m_cols = 0;

    static cublasHandle_t& handle() {
        static cublasHandle_t h = nullptr;
        if (!h) {
            CUBLAS_CHECK(cublasCreate(&h));
        }
        return h;
    }

public:
    CudaMatrix() = default;

    // Allocate zeroed GPU memory
    CudaMatrix(int rows, int cols) : m_rows(rows), m_cols(cols) {
        if (rows > 0 && cols > 0) {
            CUDA_CHECK(cudaMalloc(&d_data, size_bytes()));
            CUDA_CHECK(cudaMemset(d_data, 0, size_bytes()));
        }
    }

    // Upload from host pointer
    CudaMatrix(int rows, int cols, const T* host_ptr) : m_rows(rows), m_cols(cols) {
        if (rows > 0 && cols > 0) {
            CUDA_CHECK(cudaMalloc(&d_data, size_bytes()));
            CUDA_CHECK(cudaMemcpy(d_data, host_ptr, size_bytes(), cudaMemcpyHostToDevice));
        }
    }

    // Upload from CPU Matrix (same type)
    CudaMatrix(const Matrix<T>& cpu);

    // Upload from float CPU Matrix (converts float→T on host, e.g. float→__half)
    static CudaMatrix from_cpu_float(const Matrix<float>& cpu);

    // Download GPU data as float CPU Matrix (converts T→float, e.g. __half→float)
    Matrix<float> to_cpu_float() const;

    ~CudaMatrix() {
        if (d_data) cudaFree(d_data);
    }

    // Move semantics
    CudaMatrix(CudaMatrix&& other) noexcept
        : d_data(other.d_data), m_rows(other.m_rows), m_cols(other.m_cols) {
        other.d_data = nullptr;
        other.m_rows = 0;
        other.m_cols = 0;
    }

    CudaMatrix& operator=(CudaMatrix&& other) noexcept {
        if (this != &other) {
            if (d_data) cudaFree(d_data);
            d_data = other.d_data;
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            other.d_data = nullptr;
            other.m_rows = 0;
            other.m_cols = 0;
        }
        return *this;
    }

    // No copy
    CudaMatrix(const CudaMatrix&) = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;

    // Accessors
    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    T* data() { return d_data; }
    const T* data() const { return d_data; }
    bool empty() const { return d_data == nullptr; }
    size_t size_bytes() const { return static_cast<size_t>(m_rows) * m_cols * sizeof(T); }

    // Download to CPU Matrix
    Matrix<T> to_cpu() const;

    // Copy a single row to host buffer
    void copy_row_to_host(int row, T* host_dst) const {
        CUDA_CHECK(cudaMemcpy(host_dst, d_data + static_cast<size_t>(row) * m_cols,
                              m_cols * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Row-major matmul: C = A * B
    // cuBLAS is column-major, so we compute C^T = B^T * A^T
    // by calling sgemm with B, A (swapped) and leading dims = cols
    static CudaMatrix matmul(const CudaMatrix& A, const CudaMatrix& B);

    // Device-to-device copy of a row into this matrix at a given row offset
    void write_row(int dst_row, const T* src, int ncols) {
        CUDA_CHECK(cudaMemcpy(d_data + static_cast<size_t>(dst_row) * m_cols,
                              src, ncols * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    // Device-to-device copy of multiple rows
    void write_rows(int dst_row, const T* src, int nrows, int ncols) {
        CUDA_CHECK(cudaMemcpy(d_data + static_cast<size_t>(dst_row) * m_cols,
                              src, static_cast<size_t>(nrows) * ncols * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    }
};

using CudaMatrixF32 = CudaMatrix<float>;
using CudaMatrixF16 = CudaMatrix<__half>;

#endif // ML_USE_CUDA
