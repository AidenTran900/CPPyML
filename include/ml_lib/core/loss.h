#pragma once
#include "../math/matrix.h"

enum LossType {
    L1,
        // | actual - predicted |
    MAE,
        // 1/N * | actual - predicted |
        // Errors treated equally
    L2,
        // | actual - predicted |^2
    MSE,
        // 1/N * | actual - predicted |^2
        // Large errors penalized more heavily
        // Moves closer to outliers than MAE
    RMSE
        // sqrt(1/N * | actual - predicted |^2 )
        // MSE / MAE combined
};

class LossFunction {
    public:
        virtual double compute(const Matrix& y_pred, const Matrix& y_true) const = 0;
        virtual Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const = 0;
        virtual ~LossFunction() {}
};

class L1Loss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class MAELoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class L2Loss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class MSELoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class RMSELoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

LossFunction* createLoss(LossType type);
