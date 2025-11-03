#include <vector>
#include <stdexcept> 
#include <iostream>
#include <iomanip>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

struct EliminationResult {
    Matrix matrix;
    Matrix augmented;
    int swaps;
};

void printMatrix(Matrix m) {
    std::cout << std::fixed << std::setprecision(2); 

    for (const auto& row : m) {
        for (double val : row) {
            std::cout << std::setw(8) << val << " "; 
        }
        std::cout << "\n";
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

EliminationResult forwardElimination(Matrix m, Matrix aug = Matrix()) {
    if (m.empty() || m[0].empty()) { 
        return {m, aug, 0};
    }

    int m_rows = m.size();
    int m_cols = m[0].size();

    bool is_augmented = !(aug.empty() || aug[0].empty());
    if (is_augmented) {
        if (m_rows != static_cast<int>(aug.size())) {
             throw std::invalid_argument("Augmented matrix row count must match input matrix row count.");
        }
    }

    Matrix m_c = m;
    Matrix aug_c = aug;

    int pivot_row = 0;
    int swaps = 0;

    for (int j = 0; j < m_cols && pivot_row < m_rows; j++) { // Condition means loop thru cols until we run out of rows to eliminate
        int max_row_ind = pivot_row;
        double max_val = std::abs(m_c[pivot_row][j]); 

        // Track > val in column for swap
        for (int i = pivot_row + 1; i < m_rows; i++) { // Start at pivot_row so prevent unecessary checks
            if (std::abs(m_c[i][j]) > max_val) {
                max_val = std::abs(m_c[i][j]); 
                max_row_ind = i;
            }
        }

        // Rows w/ greatest vals at col are on top
        if (max_row_ind != pivot_row) {
            std::swap(m_c[pivot_row], m_c[max_row_ind]);
            if (is_augmented) {
                std::swap(aug_c[pivot_row], aug_c[max_row_ind]);
            }
            swaps++;
        }

        // Prevent super large #s (1/0.000000001 super big), also floating pt errors :/ 
        if (std::abs(m_c[pivot_row][j]) < 1e-9) {
            m_c[pivot_row][j] = 0.0;
            continue;
        }

        double pivot = m_c[pivot_row][j];

        // Do subtraction 
        for (int i = pivot_row + 1; i < m_rows; i++) {
            double target = m_c[i][j]; // Element we want to be 0
            double c = target / pivot; // Provides coefficient for subtraction

            for (int z = j; z < m_cols; z++) {
                m_c[i][z] -= m_c[pivot_row][z] * c;
            }

            if (is_augmented) {
                for (int z = 0; z < static_cast<int>(aug_c[0].size()); z++) {
                    aug_c[i][z] -= aug_c[pivot_row][z] * c;
                }
            }

            m_c[i][j] = 0.0;
        }

        pivot_row++;
    }

    return {m_c, aug_c, swaps};
}

EliminationResult backwardElimination(Matrix m, Matrix aug = Matrix()) {
    if (m.empty() || m[0].empty()) { 
        return {m, aug, 0};
    }

    int m_rows = m.size();
    int m_cols = m[0].size();

    bool is_augmented = !aug.empty();
    int aug_cols = 0;
    if (is_augmented) {
        if (m_rows != static_cast<int>(aug.size())) {
             throw std::invalid_argument("Augmented matrix row count must match input matrix row count.");
        }
        if (!aug.empty() && !aug[0].empty()) {
            aug_cols = aug[0].size();
        }
    }

    Matrix m_c = m;
    Matrix aug_c = aug;

    for (int i = m_rows - 1; i >= 0; i--) {
        
        // Get pivot
        int pivot_col = -1;
        for (int j = 0; j < m_cols; j++) {
            if (std::abs(m_c[i][j]) > 1e-9) {
                pivot_col = j;
                break;
            }
        }

        if (pivot_col == -1) {
            continue;
        }

        // Normalize pivot row 
        double pivot_val = m_c[i][pivot_col];
        
        for (int j = pivot_col; j < m_cols; j++) {
            m_c[i][j] /= pivot_val;
        }
        if (is_augmented) {
            for (int j = 0; j < aug_cols; j++) {
                aug_c[i][j] /= pivot_val;
            }
        }
        m_c[i][pivot_col] = 1.0; 

        // Eliminate all elements above pivot
        for (int k = i - 1; k >= 0; k--) {
            double target_val = m_c[k][pivot_col];
            
            for (int j = pivot_col; j < m_cols; j++) {
                m_c[k][j] -= target_val * m_c[i][j];
            }

            if (is_augmented) {
                for (int j = 0; j < aug_cols; j++) {
                    aug_c[k][j] -= target_val * aug_c[i][j];
                }
            }
            m_c[k][pivot_col] = 0.0;
        }
    }

    return {m_c, aug_c, 0};
}

Matrix inverse(Matrix m) {
    if (m.empty()) {
        throw std::invalid_argument("Cannot invert an empty matrix.");
    }

    int m_rows = m.size();
    int m_cols = m[0].size();

    if (m_rows != m_cols) {
        throw std::invalid_argument("Cannot invert a non-square matrix.");
    }

    Matrix identity(m_rows, Vector(m_rows, 0.0));
    for (int i = 0; i < m_rows; i++) {
        identity[i][i] = 1.0;
    }

    EliminationResult forward_result = forwardElimination(m, identity);

    for (int i = 0; i < m_rows; i++) { // Check if diagonal entry is 0
        if (std::abs(forward_result.matrix[i][i]) < 1e-9) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
    }

    EliminationResult backward_result = backwardElimination(forward_result.matrix, forward_result.augmented);

    return backward_result.augmented;
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

    EliminationResult elim_result = forwardElimination(m);

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