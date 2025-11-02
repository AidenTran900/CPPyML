#include <vector>
#include <stdexcept> 
#include <iostream>
#include <iomanip>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

struct EliminationResult {
    Matrix matrix;
    int swaps;
};

void printMatrix(Matrix m) {
    for (std::vector<double> row : m) {
        for (double val : row) {
            fmt::print("{} ", val);
        }
        fmt::print("\n");
    }
}

Matrix addMatrix(Matrix m1, Matrix m2) {
    int m1_rows = m1.size();
    int m1_cols = m1[0].size();
    int m2_rows = m2.size();
    int m2_cols = m2[0].size();

    // Check dimensions
    if (m1_cols != m2_cols || m1_rows != m2_rows) { 
        throw std::invalid_argument("Matrix dimensions are incompatible."); 
    }

    Matrix result(m1_rows, std::vector<double>(m1_cols));

    // Do add stuff
    for (int i = 0; i < m1_rows; i++) {
        for (int j = 0; j < m1_cols; j++) {
            result[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return result;
}

Matrix multiplyMatrix(Matrix m1, Matrix m2) {
    int m1_rows = m1.size();
    int m1_cols = m1[0].size();
    int m2_rows = m2.size();
    int m2_cols = m2[0].size();

    // Check dimensions
    if (m1_cols != m2_rows) { 
        throw std::invalid_argument("Matrix dimensions are incompatible. ""Matrix 1 cols (" + std::to_string(m1_cols) + ") != Matrix 2 rows (" + std::to_string(m2_rows) + ")."); 
    }

    Matrix result(m1_rows, Vector(m2_cols));

    // Do multiplication stuff
    for (int i = 0; i < m1_rows; i++) {
        for (int j = 0; j < m2_cols; j++) {
            for (int h = 0; h < m1_cols; h++) {
                result[i][j] += m1[i][h] * m2[h][j];
            }
        }
    }
    return result;
}

EliminationResult gaussianElimination(Matrix m) {
    if (m.empty() || m[0].empty()) { 
        return {m, 0};
    }

    int m_rows = m.size();
    int m_cols = m[0].size();

    Matrix u = m;
    Matrix identity(m_rows, Vector(m_cols));

    int pivot_row = 0;
    int swaps = 0;

    for (int j = 0; j < m_cols && pivot_row < m_rows; j++) { // Condition means loop thru cols until we run out of rows to eliminate
        int max_row_ind = pivot_row;
        double max_val = std::abs(u[pivot_row][j]); 

        // Track > val in column for swap
        for (int i = pivot_row + 1; i < m_rows; i++) { // Start at pivot_row so prevent unecessary checks
            if (std::abs(u[i][j]) > max_val) {
                max_val = std::abs(u[i][j]); 
                max_row_ind = i;
            }
        }

        // Rows w/ greatest vals at col are on top
        if (max_row_ind != pivot_row) {
            std::swap(u[pivot_row], u[max_row_ind]);
            swaps++;
        }

        // Prevent super large #s (1/0.000000001 super big), also floating pt errors :/ 
        if (std::abs(u[pivot_row][j]) < 1e-9) {
            u[pivot_row][j] = 0.0;
            continue;
        }

        double pivot = u[pivot_row][j];

        // Do subtraction 
        for (int i = pivot_row + 1; i < m_rows; i++) {
            double target = u[i][j]; // Element we want to be 0
            double c = target / pivot; // Provides coefficient for subtraction

            for (int z = j; z < m_cols; z++) {
                u[i][z] -= u[pivot_row][z] * c;
            }

            u[i][j] = 0.0;
        }

        pivot_row++;
    }

    return {u, swaps};
}

double determinant(Matrix m) {
    if (m.empty()) {
        return 0.0;
    }

    int m_rows = m.size();
    int m_cols = m[0].size();
    
    // Check dimensions
    if (m_rows != m_cols) { 
        throw std::invalid_argument("Matrix dimensions are not square."); 
    }
    if (m_rows == 1 && m_cols == 1) {
        return m[0][0];
    }

    if (m_rows == 2 && m_cols == 2) {
        return m[0][0] * m[1][1] - m[1][0] * m[0][1];
    } 

    EliminationResult elim_result = gaussianElimination(m);

    double det = 1.0;
    for (int i = 0; i < m_rows; i++) {
        if (std::abs(elim_result.matrix[i][i]) < 1e-9) { // Stop early if "0"
            return 0.0;
        }
        det *= elim_result.matrix[i][i];
    }

    if (elim_result.swaps % 2 != 0) {
        det *= -1.0;
    }

    return det;
}