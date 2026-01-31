#pragma once
#include "../math/matrix.h"
#include "optimizer.h"
#include <vector>

class LayerNorm {
    private:
        int features; // Size of the feature dimension
        double epsilon; // Small constant for numerical stability
        Matrix gamma; // Scale parameter
        Matrix beta; // Shift parameter
        Matrix grad_gamma; // Gradient for gamma
        Matrix grad_beta; // Gradient for beta
        Matrix normalized_cache; // Cache normalized input for backward pass
        std::vector<double> std_cache; // Cache std for each row (backward pass)

    public:
        LayerNorm(int features, double epsilon = 1e-5);

        Matrix forward(const Matrix &input);
        Matrix backward(const Matrix &grad_output);
        void update(Optimizer *opt);
};