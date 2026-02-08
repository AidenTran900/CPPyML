#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>

template<typename T = double>
class LayerNorm {
    private:
        int features; // Size of the feature dimension
        double epsilon; // Small constant for numerical stability
        Matrix<T> gamma; // Scale parameter
        Matrix<T> beta; // Shift parameter
        Matrix<T> grad_gamma; // Gradient for gamma
        Matrix<T> grad_beta; // Gradient for beta
        Matrix<T> normalized_cache; // Cache normalized input for backward pass
        std::vector<T> std_cache; // Cache std for each row (backward pass)

    public:
        LayerNorm(int features, double epsilon = 1e-5);

        Matrix<T> forward(const Matrix<T> &input);
        Matrix<T> backward(const Matrix<T> &grad_output);
        void update(Optimizer<T> *opt);

        void loadWeights(const Matrix<T>& g, const Matrix<T>& b) {
            gamma = g; beta = b;
        }
};
