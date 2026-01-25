#pragma once
#include "../math/matrix.h"
#include <memory>

enum class LossType {
    MEAN_ABSOLUTE_ERROR,
        // 1/N * | actual - predicted |
        // Errors treated equally
        // | actual - predicted |^2
    MEAN_SQUARED_ERROR,
        // 1/N * | actual - predicted |^2
        // Large errors penalized more heavily
        // Moves closer to outliers than MEAN_ABSOLUTE_ERROR
    ROOT_MEAN_SQUARED_ERROR,
        // sqrt(1/N * | actual - predicted |^2 )
        // MEAN_SQUARED_ERROR / MEAN_ABSOLUTE_ERROR combined
    BINARY_CROSS_ENTROPY,
        // -1/N * Σ [ actual * log(predicted) + (1 - actual) * log(1 - predicted) ]
        // For binary classification tasks
    CATEGORICAL_CROSS_ENTROPY
        // Multi-class classification tasks

};



class LossFunction {
    public:
        virtual double compute(const Matrix& y_pred, const Matrix& y_true) const = 0;
        virtual Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const = 0;
        virtual ~LossFunction();
};

class MeanAbsoluteErrorLoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class MeanSquaredErrorLoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class RootMeanSquaredErrorLoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class BinaryCrossEntropyLoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class CategoricalCrossEntropyLoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};


std::unique_ptr<LossFunction> createLoss(LossType type);
