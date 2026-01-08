#include "ml_lib/math/matrix.h"
#include "config.h"
#include <cmath>
#if ML_HAS_AVX2 && ML_USE_SIMD
    #include <immintrin.h>
#endif

Matrix::Matrix()
    : m_rows(0), 
      m_cols(0), 
      m_data(0,0) {}

Matrix::Matrix(int rows, int cols, double init_val)
    : m_rows(rows), 
      m_cols(cols), 
      m_data(rows * cols, init_val) {}

Matrix::Matrix(const std::vector<std::vector<double>>& vec) {
    if (vec.empty()) {
        m_rows = 0;
        m_cols = 0;
        return;
    }
    m_rows = vec.size();
    m_cols = vec[0].size();
    m_data.resize(m_rows * m_cols);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            m_data[i * m_cols + j] = vec[i][j];
        }
    }
}

double& Matrix::operator()(int i, int j) { // Setter
    return m_data[i * m_cols + j];
}

double Matrix::operator()(int i, int j) const {
    return m_data[i * m_cols + j];
}

const double* Matrix::getRow(int row) const {
    return &m_data[row * m_cols];
}

double* Matrix::getRow(int row) {
    return &m_data[row * m_cols];
}

std::vector<double> Matrix::getRowVector(int row) const {
    const double* row_ptr = getRow(row);
    return std::vector<double>(row_ptr, row_ptr + m_cols);
}

Matrix Matrix::row(int r) const {
    Matrix result(1, m_cols);
    for (int j = 0; j < m_cols; j++) {
        result(0, j) = (*this)(r, j);
    }
    return result;
}

void Matrix::swapRows(int row1, int row2) {
    if (row1 == row2) return;
    double* r1 = getRow(row1);
    double* r2 = getRow(row2);
    for (int j = 0; j < m_cols; j++) {
        std::swap(r1[j], r2[j]);
    }
}

void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            std::cout << std::setw(8) << (*this)(i, j) << " ";
        }
        std::cout << "\n";
    }
}


Matrix Matrix::operator+(const Matrix& other) const {
    if (m_cols != other.m_cols || m_rows != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible.");
    }
    Matrix result(m_rows, m_cols);
    size_t i = 0;

    #if ML_HAS_AVX2 && ML_USE_SIMD
    for (; i + 4 <= m_rows * m_cols; i += 4) {
        __m256d vec1 = _mm256_loadu_pd(&m_data[i]);
        __m256d vec2 = _mm256_loadu_pd(&other.m_data[i]);
        __m256d sum = _mm256_add_pd(vec1, vec2);
        _mm256_storeu_pd(&result.m_data[i], sum);
    }
    #endif
    
    int size = m_rows * m_cols;
    for (; i < size; i++) {
        result.m_data[i] = m_data[i] + other.m_data[i];
    }
    return result;
}


Matrix Matrix::operator-(const Matrix& other) const {
    if (m_cols != other.m_cols || m_rows != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible.");
    }
    Matrix result(m_rows, m_cols);
    size_t i = 0;

    #if ML_HAS_AVX2 && ML_USE_SIMD
    for (; i + 4 <= m_rows * m_cols; i += 4) {
        __m256d vec1 = _mm256_loadu_pd(&m_data[i]);
        __m256d vec2 = _mm256_loadu_pd(&other.m_data[i]);
        __m256d sum = _mm256_sub_pd(vec1, vec2);
        _mm256_storeu_pd(&result.m_data[i], sum);
    }
    #endif
    
    int size = m_rows * m_cols;
    for (; i < size; i++) {
        result.m_data[i] = m_data[i] - other.m_data[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    // Check dimensions
    if (m_cols != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible. "
            "Matrix 1 cols (" + std::to_string(m_cols) +
            ") != Matrix 2 rows (" + std::to_string(other.m_rows) + ").");
    }

    Matrix result(m_rows, other.m_cols);

    #if ML_HAS_AVX2 && ML_USE_SIMD
    for (size_t i = 0; i < m_rows; i++) {
        size_t j = 0;
        for (; j + 4 <= other.m_cols; j += 4) {
            __m256d sum = _mm256_setzero_pd();
            for (int h = 0; h < m_cols; h++) {
                __m256d vec1 = _mm256_set1_pd((*this)(i, h));
                __m256d vec2 = _mm256_loadu_pd(&other.m_data[h * other.m_cols + j]);
                sum = _mm256_fmadd_pd(vec1, vec2, sum);
            }
            _mm256_storeu_pd(&result.m_data[i * other.m_cols + j], sum);
        }
        for (; j < other.m_cols; j++) {
            for (int h = 0; h < m_cols; h++) {
                result(i, j) += (*this)(i, h) * other(h, j);
            }
        }
    }
    #else
    for (size_t i = 0; i < m_rows; i++) {
        for (size_t j = 0; j < other.m_cols; j++) {
            for (int h = 0; h < m_cols; h++) {
                result(i, j) += (*this)(i, h) * other(h, j);
            }
        }
    }
    #endif

    return result;
}


Matrix Matrix::hadamard(const Matrix& other) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }

    size_t i = 0;
    Matrix result(m_rows, m_cols);
    int size = m_rows * m_cols;

    #if ML_HAS_AVX2 && ML_USE_SIMD
    for (; i + 4 <= size; i += 4) {
        __m256d vec1 = _mm256_loadu_pd(&m_data[i]);
        __m256d vec2 = _mm256_loadu_pd(&other.m_data[i]);
        __m256d product = _mm256_mul_pd(vec1, vec2);
        _mm256_storeu_pd(&result.m_data[i], product);
    }
    #endif

    for (; i < size; i++) {
        result.m_data[i] = m_data[i] * other.m_data[i];
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(m_rows, m_cols);
    int size = m_rows * m_cols;
    size_t i = 0;
    
    #if ML_HAS_AVX2 && ML_USE_SIMD
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    for (; i + 4 <= size; i += 4) {
        __m256d vec = _mm256_loadu_pd(&m_data[i]);
        __m256d scaled = _mm256_mul_pd(vec, scalar_vec);
        _mm256_storeu_pd(&result.m_data[i], scaled);
    }
    #endif
    
    for (; i < size; i++) {
        result.m_data[i] = m_data[i] * scalar;
    }
    return result;
}

EliminationResult Matrix::forwardElimination(const Matrix& m, const Matrix& aug) {
    if (m.empty()) {
        return {m, aug, 0};
    }

    bool is_augmented = !aug.empty();
    if (is_augmented && m.m_rows != aug.m_rows) {
        throw std::invalid_argument("Augmented matrix row count must match input matrix row count.");
    }

    Matrix m_c = m;
    Matrix aug_c = aug;

    int pivot_row = 0;
    int swaps = 0;

    for (int j = 0; j < m_c.m_cols && pivot_row < m_c.m_rows; j++) { // Condition means loop thru cols until we run out of rows to eliminate
        int max_row_ind = pivot_row;
        double max_val = std::abs(m_c(pivot_row, j));

        // Track > val in column for swap
        for (int i = pivot_row + 1; i < m_c.m_rows; i++) { // Start at pivot_row so prevent unecessary checks
            if (std::abs(m_c(i, j)) > max_val) {
                max_val = std::abs(m_c(i, j));
                max_row_ind = i;
            }
        }

        // Rows w/ greatest vals at col are on top
        if (max_row_ind != pivot_row) {
            m_c.swapRows(pivot_row, max_row_ind);
            if (is_augmented) {
                aug_c.swapRows(pivot_row, max_row_ind);
            }
            swaps++;
        }

        // Prevent super large #s (1/0.000000001 super big), also floating pt errors :/
        if (std::abs(m_c(pivot_row, j)) < 1e-9) {
            m_c(pivot_row, j) = 0.0;
            continue;
        }

        double pivot = m_c(pivot_row, j);

        for (int i = pivot_row + 1; i < m_c.m_rows; i++) {
            double target = m_c(i, j);
            double c = target / pivot;

            double* current_row = m_c.getRow(i);
            const double* pivot_row_data = m_c.getRow(pivot_row);
            for (int z = j; z < m_c.m_cols; z++) {
                current_row[z] -= pivot_row_data[z] * c;
            }

            if (is_augmented) {
                double* aug_current = aug_c.getRow(i);
                const double* aug_pivot = aug_c.getRow(pivot_row);
                for (int z = 0; z < aug_c.m_cols; z++) {
                    aug_current[z] -= aug_pivot[z] * c;
                }
            }

            current_row[j] = 0.0;
        }

        pivot_row++;
    }

    return {m_c, aug_c, swaps};
}

EliminationResult Matrix::backwardElimination(const Matrix& m, const Matrix& aug) {
    if (m.empty()) {
        return {m, aug, 0};
    }

    bool is_augmented = !aug.empty();
    if (is_augmented && m.m_rows != aug.m_rows) {
        throw std::invalid_argument("Augmented matrix row count must match input matrix row count.");
    }

    Matrix m_c = m;
    Matrix aug_c = aug;

    for (int i = m_c.m_rows - 1; i >= 0; i--) {

        // Get pivot
        int pivot_col = -1;
        for (int j = 0; j < m_c.m_cols; j++) {
            if (std::abs(m_c(i, j)) > 1e-9) {
                pivot_col = j;
                break;
            }
        }

        if (pivot_col == -1) {
            continue;
        }

        double pivot_val = m_c(i, pivot_col);

        double* pivot_row_data = m_c.getRow(i);
        for (int j = pivot_col; j < m_c.m_cols; j++) {
            pivot_row_data[j] /= pivot_val;
        }
        if (is_augmented) {
            double* aug_pivot_row = aug_c.getRow(i);
            for (int j = 0; j < aug_c.m_cols; j++) {
                aug_pivot_row[j] /= pivot_val;
            }
        }
        pivot_row_data[pivot_col] = 1.0;

        for (int k = i - 1; k >= 0; k--) {
            double target_val = m_c(k, pivot_col);

            double* target_row = m_c.getRow(k);
            for (int j = pivot_col; j < m_c.m_cols; j++) {
                target_row[j] -= target_val * pivot_row_data[j];
            }

            if (is_augmented) {
                double* aug_target = aug_c.getRow(k);
                const double* aug_pivot = aug_c.getRow(i);
                for (int j = 0; j < aug_c.m_cols; j++) {
                    aug_target[j] -= target_val * aug_pivot[j];
                }
            }
            target_row[pivot_col] = 0.0;
        }
    }

    return {m_c, aug_c, 0};
}

Matrix Matrix::inverse() const {
    if (empty()) {
        throw std::invalid_argument("Cannot invert an empty matrix.");
    }

    if (m_rows != m_cols) {
        throw std::invalid_argument("Cannot invert a non-square matrix.");
    }

    Matrix identity(m_rows, m_rows);
    for (int i = 0; i < m_rows; i++) {
        identity(i, i) = 1.0;
    }

    EliminationResult forward_result = forwardElimination(*this, identity);

    for (int i = 0; i < m_rows; i++) { // Check if diagonal entry is 0
        if (std::abs(forward_result.matrix(i, i)) < 1e-9) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
    }

    EliminationResult backward_result = backwardElimination(forward_result.matrix, forward_result.augmented);

    return backward_result.augmented;
}

Matrix Matrix::transpose() const {
    Matrix result(m_cols, m_rows);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

double Matrix::determinant() const {
    if (empty()) {
        return 0.0;
    }

    // Check dimensions
    if (m_rows != m_cols) {
        throw std::invalid_argument("Matrix dimensions are not square.");
    }
    if (m_rows == 1 && m_cols == 1) {
        return (*this)(0, 0);
    }

    if (m_rows == 2 && m_cols == 2) {
        return (*this)(0, 0) * (*this)(1, 1) - (*this)(1, 0) * (*this)(0, 1);
    }

    EliminationResult elim_result = forwardElimination(*this);

    double det = 1.0;
    for (int i = 0; i < m_rows; i++) {
        if (std::abs(elim_result.matrix(i, i)) < 1e-9) { // Stop early if "0"
            return 0.0;
        }
        det *= elim_result.matrix(i, i);
    }

    if (elim_result.swaps % 2 != 0) {
        det *= -1.0;
    }

    return det;
}

double Matrix::dot(const Matrix& m) const {
    if (rows() != m.rows() || cols() != m.cols()) {
        throw std::invalid_argument("Dimension mismatch for dot product");
    }

    double result = 0.0;
    int size = m_rows * m_cols;
    size_t i = 0;
    
    #if ML_HAS_AVX2 && ML_USE_SIMD
    __m256d sum_vec = _mm256_setzero_pd();
    
    for (; i + 4 <= size; i += 4) {
        __m256d vec1 = _mm256_loadu_pd(&m_data[i]);
        __m256d vec2 = _mm256_loadu_pd(&m.m_data[i]);
        sum_vec = _mm256_fmadd_pd(vec1, vec2, sum_vec); 
    }
    
    alignas(32) double temp[4];
    _mm256_store_pd(temp, sum_vec);
    result = temp[0] + temp[1] + temp[2] + temp[3];
    #endif
    
    for (; i < size; i++) {
        result += m_data[i] * m.m_data[i];
    }
    return result;
}

Matrix Matrix::sign() const {
    Matrix result(m_rows, m_cols);
    int size = m_rows * m_cols;
    for (int i = 0; i < size; i++) {
        double val = m_data[i];
        result.m_data[i] = (val > 0.0) ? 1.0 : ((val < 0.0) ? -1.0 : 0.0);
    }
    return result;
}

bool Matrix::operator==(const Matrix& other) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        return false;
    }

    int size = m_rows * m_cols;
    for (int i = 0; i < size; i++) {
        if (m_data[i] != other.m_data[i]) {
            return false;
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}

bool Matrix::approxEqual(const Matrix& other, double epsilon) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        return false;
    }

    int size = m_rows * m_cols;
    for (int i = 0; i < size; i++) {
        if (std::abs(m_data[i] - other.m_data[i]) > epsilon) {
            return false;
        }
    }
    return true;
}