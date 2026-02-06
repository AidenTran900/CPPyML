#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

struct EliminationResult;

class Matrix {
    private:
        int m_rows, m_cols;
        std::vector<double> m_data;

    public:
        Matrix();
        Matrix(int rows, int cols, double init_val = 0.0);
        Matrix(const std::vector<std::vector<double>>& vec);

        double& operator()(int i, int j); // Setter
        double operator()(int i, int j) const; // Getter

        int rows() const { return m_rows; }
        int cols() const { return m_cols; }
        bool empty() const { return m_rows == 0 || m_cols == 0; }

        const double* getRow(int row) const;
        double* getRow(int row);
        std::vector<double> getRowVector(int row) const;
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
        double determinant() const;
        double dot(const Matrix& other) const;

        bool operator==(const Matrix& other) const;
        bool operator!=(const Matrix& other) const;
        bool approxEqual(const Matrix& other, double epsilon = 1e-9) const;

        Matrix verticalConcat(const Matrix& other) const;

        static EliminationResult forwardElimination(const Matrix& m, const Matrix& aug = Matrix(0, 0));
        static EliminationResult backwardElimination(const Matrix& m, const Matrix& aug = Matrix(0, 0));
        
};

struct EliminationResult {
    Matrix matrix;
    Matrix augmented;
    int swaps;
};
