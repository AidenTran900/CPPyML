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



template<typename T = double>
class LossFunction {
    public:
        virtual double compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const = 0;
        virtual Matrix<T> gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const = 0;
        virtual ~LossFunction();
};

template<typename T = double>
class MeanAbsoluteErrorLoss : public LossFunction<T> {
    public:
        double compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
        Matrix<T> gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
};

template<typename T = double>
class MeanSquaredErrorLoss : public LossFunction<T> {
    public:
        double compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
        Matrix<T> gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
};

template<typename T = double>
class RootMeanSquaredErrorLoss : public LossFunction<T> {
    public:
        double compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
        Matrix<T> gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
};

template<typename T = double>
class BinaryCrossEntropyLoss : public LossFunction<T> {
    public:
        double compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
        Matrix<T> gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
};

template<typename T = double>
class CategoricalCrossEntropyLoss : public LossFunction<T> {
    public:
        double compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
        Matrix<T> gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const override;
};


template<typename T = double>
std::unique_ptr<LossFunction<T>> createLoss(LossType type);
