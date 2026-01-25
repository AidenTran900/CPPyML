#include "ml_lib/core/loss.h"
#include <cmath>
#include <memory>

LossFunction::~LossFunction() {}

// MeanAbsoluteError
double MeanAbsoluteErrorLoss::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }
    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            result += abs(y_pred(i, j) - y_true(i, j));
        }
    }
    return result / n;
}

Matrix MeanAbsoluteErrorLoss::gradient(const Matrix& y_pred, const Matrix& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix(0, 0);

    return (y_pred - y_true).sign() * (1.0 / m);
}

// MeanSquaredError
double MeanSquaredErrorLoss::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double diff = y_pred(i, j) - y_true(i, j);
            result += diff * diff;
        }
    }
    return result / n;
}

Matrix MeanSquaredErrorLoss::gradient(const Matrix& y_pred, const Matrix& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix(0, 0);

    return (y_pred - y_true) * (2.0 / m);
}


// RootMeanSquaredError
double RootMeanSquaredErrorLoss::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double diff = y_pred(i, j) - y_true(i, j);
            result += diff * diff;
        }
    }

    return sqrt(result / n);
}

Matrix RootMeanSquaredErrorLoss::gradient(const Matrix& y_pred, const Matrix& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix(0, 0);

    Matrix error = y_pred - y_true;

    double mse_sum = 0.0;
    for (int i = 0; i < error.rows(); i++) {
        double err = error(i, 0);
        mse_sum += err * err;
    }
    double mse = mse_sum / m;
    double rmse_val = (mse > 1e-9) ? sqrt(mse) : 1e-9;

    double final_rmse_scale = 1.0 / (m * 2.0 * rmse_val);
    return error * final_rmse_scale;
}


// BinaryCrossEntropy
double BinaryCrossEntropyLoss::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double pred_val = y_pred(i, j);
            double true_val = y_true(i, j);
            double epsilon = 1e-9;
            pred_val = std::max(epsilon, std::min(1.0 - epsilon, pred_val));

            result += ((true_val * log(pred_val)) + ((1 - true_val) * log(1 - pred_val)));
        }
    }

    return (result / n);
}

Matrix BinaryCrossEntropyLoss::gradient(const Matrix& y_pred, const Matrix& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix(0, 0);

    return (y_pred - y_true) * (1.0 / m);
}

// CategoricalCrossEntropy
double CategoricalCrossEntropyLoss::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int m = y_pred.rows();
    if (m == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double pred_val = y_pred(i, j);
            double true_val = y_true(i, j);
            double epsilon = 1e-9;
            pred_val = std::max(epsilon, std::min(1.0 - epsilon, pred_val));

            result -= true_val * log(pred_val);
        }
    }

    return result / m;
}

Matrix CategoricalCrossEntropyLoss::gradient(const Matrix& y_pred, const Matrix& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix(0, 0);

    return (y_pred - y_true) * (1.0 / m);
}

std::unique_ptr<LossFunction> createLoss(LossType type)
{
    switch (type) {
        case LossType::MEAN_ABSOLUTE_ERROR:      return std::make_unique<MeanAbsoluteErrorLoss>();
        case LossType::MEAN_SQUARED_ERROR:       return std::make_unique<MeanSquaredErrorLoss>();
        case LossType::ROOT_MEAN_SQUARED_ERROR:  return std::make_unique<RootMeanSquaredErrorLoss>();
        case LossType::BINARY_CROSS_ENTROPY:     return std::make_unique<BinaryCrossEntropyLoss>();
        case LossType::CATEGORICAL_CROSS_ENTROPY:  return std::make_unique<CategoricalCrossEntropyLoss>();
    }

    throw std::invalid_argument("Unsupported loss type.");
}
