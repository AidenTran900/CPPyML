#pragma once
#include "../math/matrix.h"
#include "model-interface.h"
#include <vector>

enum KERNEL {
    LINEAR,
    POLYNOMIAL,
    RBF,
    SIGMOID,
};

class SupportVectorMachine : public FitPredictModel {
    private:
        double C;
        double gamma;
        KERNEL kernel_type;

        int degree;
        double tolerance;
        int max_iter;
        double coef0;
        double bias;

        Matrix support_vectors;
        Matrix support_labels;
        std::vector<double> support_alphas;

        double kernel(const Matrix& X1, const Matrix& X2);
        double decision(const Matrix& X);
        double decisionCached(int idx, const Matrix& K_cache, const std::vector<double>& alphas, const Matrix& Y_train);
        int examineExample(int I2, const Matrix& X_train, const Matrix& Y_train, Matrix& K_cache, std::vector<double>& alphas, std::vector<double>& errors);
        int takeStep(int I1, int I2, const Matrix& X_train, const Matrix& Y_train, const Matrix& K_cache, std::vector<double>& alphas, std::vector<double>& errors);

    public:
        SupportVectorMachine(
            double C = 1.0,
            double gamma = 0.1,
            KERNEL kernel = KERNEL::LINEAR,
            int degree = 3,
            double tolerance = 1e-3,
            int max_iter = 1000,
            double coef0 = 0.0
        );

        void fit(const Matrix& X, const Matrix& y) override;
        Matrix predict(const Matrix& X) override;
};
