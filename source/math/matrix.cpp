#include "ml_lib/math/matrix.h"
#include <cmath>

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

double Matrix::operator()(int i, int j) const { // Getter
    return m_data[i * m_cols + j];
}

void Matrix::swapRows(int row1, int row2) {
    for (int j = 0; j < m_cols; j++) {
        std::swap((*this)(row1, j), (*this)(row2, j));
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

Matrix Matrix::add(const Matrix& other) const {
    // Check dimensions
    if (m_cols != other.m_cols || m_rows != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible.");
    }

    Matrix result(m_rows, m_cols);

    // Do add stuff
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return result;
}

Matrix Matrix::sub(const Matrix& other) const {
    // Check dimensions
    if (m_cols != other.m_cols || m_rows != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible.");
    }

    Matrix result(m_rows, m_cols);

    // Do sub stuff
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    // Check dimensions
    if (m_cols != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible. "
            "Matrix 1 cols (" + std::to_string(m_cols) +
            ") != Matrix 2 rows (" + std::to_string(other.m_rows) + ").");
    }

    Matrix result(m_rows, other.m_cols);

    // Do multiplication stuff
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < other.m_cols; j++) {
            for (int h = 0; h < m_cols; h++) {
                result(i, j) += (*this)(i, h) * other(h, j);
            }
        }
    }
    return result;
}

Matrix Matrix::scale(double scalar) const {
    Matrix result(m_rows, m_cols);

    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(i, j) = (*this)(i, j) * scalar;
        }
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

        // Do subtraction
        for (int i = pivot_row + 1; i < m_c.m_rows; i++) {
            double target = m_c(i, j); // Element we want to be 0
            double c = target / pivot; // Provides coefficient for subtraction

            for (int z = j; z < m_c.m_cols; z++) {
                m_c(i, z) -= m_c(pivot_row, z) * c;
            }

            if (is_augmented) {
                for (int z = 0; z < aug_c.m_cols; z++) {
                    aug_c(i, z) -= aug_c(pivot_row, z) * c;
                }
            }

            m_c(i, j) = 0.0;
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

        // Normalize pivot row
        double pivot_val = m_c(i, pivot_col);

        for (int j = pivot_col; j < m_c.m_cols; j++) {
            m_c(i, j) /= pivot_val;
        }
        if (is_augmented) {
            for (int j = 0; j < aug_c.m_cols; j++) {
                aug_c(i, j) /= pivot_val;
            }
        }
        m_c(i, pivot_col) = 1.0;

        // Eliminate all elements above pivot
        for (int k = i - 1; k >= 0; k--) {
            double target_val = m_c(k, pivot_col);

            for (int j = pivot_col; j < m_c.m_cols; j++) {
                m_c(k, j) -= target_val * m_c(i, j);
            }

            if (is_augmented) {
                for (int j = 0; j < aug_c.m_cols; j++) {
                    aug_c(k, j) -= target_val * aug_c(i, j);
                }
            }
            m_c(k, pivot_col) = 0.0;
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
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) {
            result += (*this)(i,j) * m(i,j);
        }
    }
    return result;
}

Matrix Matrix::sign() const {
    Matrix result(m_rows, m_cols);
    
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            double val = (*this)(i, j);
            if (val > 0) {
                result(i, j) = 1.0;
            } else if (val < 0) {
                result(i, j) = -1.0;
            } else {
                result(i, j) = 0.0;
            }
        }
    }
    return result;
}