#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

template<typename T = double>
struct EliminationResult;

template<typename T = double>
class Matrix {
    private:
        int m_rows, m_cols;
        std::vector<T> m_data;

    public:
        Matrix();
        Matrix(int rows, int cols, double init_val = 0.0);
        Matrix(const std::vector<std::vector<T>>& vec);

        T& operator()(int i, int j); // Setter
        T operator()(int i, int j) const; // Getter

        int rows() const { return m_rows; }
        int cols() const { return m_cols; }
        bool empty() const { return m_rows == 0 || m_cols == 0; }

        const T* getRow(int row) const;
        T* getRow(int row);
        std::vector<T> getRowVector(int row) const;
        Matrix row(int r) const;

        void swapRows(int row1, int row2);
        void print() const;

        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator*(double scalar) const;
        Matrix hadamard(const Matrix& other) const;

        Matrix inverse() const;
        Matrix transpose() const;
        Matrix sign() const;
        T determinant() const;
        T dot(const Matrix& other) const;

        bool operator==(const Matrix& other) const;
        bool operator!=(const Matrix& other) const;
        bool approxEqual(const Matrix& other, double epsilon = 1e-9) const;

        Matrix verticalConcat(const Matrix& other) const;
        void appendRow(const Matrix& row);

        static EliminationResult<T> forwardElimination(const Matrix& m, const Matrix& aug = Matrix(0, 0));
        static EliminationResult<T> backwardElimination(const Matrix& m, const Matrix& aug = Matrix(0, 0));

};

template<typename T>
struct EliminationResult {
    Matrix<T> matrix;
    Matrix<T> augmented;
    int swaps;
};

using MatrixF32 = Matrix<float>;
using MatrixF64 = Matrix<double>;
