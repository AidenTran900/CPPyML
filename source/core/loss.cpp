#include "ml_lib/core/loss.h"
#include <cmath>
#include <memory>

template<typename T>
LossFunction<T>::~LossFunction() {}

// MeanAbsoluteError
template<typename T>
double MeanAbsoluteErrorLoss<T>::compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }
    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            result += std::abs(static_cast<double>(y_pred(i, j)) - static_cast<double>(y_true(i, j)));
        }
    }
    return result / n;
}

template<typename T>
Matrix<T> MeanAbsoluteErrorLoss<T>::gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix<T>(0, 0);

    return (y_pred - y_true).sign() * (1.0 / m);
}

// MeanSquaredError
template<typename T>
double MeanSquaredErrorLoss<T>::compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double diff = static_cast<double>(y_pred(i, j)) - static_cast<double>(y_true(i, j));
            result += diff * diff;
        }
    }
    return result / n;
}

template<typename T>
Matrix<T> MeanSquaredErrorLoss<T>::gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix<T>(0, 0);

    return (y_pred - y_true) * (2.0 / m);
}


// RootMeanSquaredError
template<typename T>
double RootMeanSquaredErrorLoss<T>::compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double diff = static_cast<double>(y_pred(i, j)) - static_cast<double>(y_true(i, j));
            result += diff * diff;
        }
    }

    return sqrt(result / n);
}

template<typename T>
Matrix<T> RootMeanSquaredErrorLoss<T>::gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix<T>(0, 0);

    Matrix<T> error = y_pred - y_true;

    double mse_sum = 0.0;
    for (int i = 0; i < error.rows(); i++) {
        double err = static_cast<double>(error(i, 0));
        mse_sum += err * err;
    }
    double mse = mse_sum / m;
    double rmse_val = (mse > 1e-9) ? sqrt(mse) : 1e-9;

    double final_rmse_scale = 1.0 / (m * 2.0 * rmse_val);
    return error * final_rmse_scale;
}


// BinaryCrossEntropy
template<typename T>
double BinaryCrossEntropyLoss<T>::compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double pred_val = static_cast<double>(y_pred(i, j));
            double true_val = static_cast<double>(y_true(i, j));
            double epsilon = 1e-9;
            pred_val = std::max(epsilon, std::min(1.0 - epsilon, pred_val));

            result += ((true_val * log(pred_val)) + ((1 - true_val) * log(1 - pred_val)));
        }
    }

    return (result / n);
}

template<typename T>
Matrix<T> BinaryCrossEntropyLoss<T>::gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix<T>(0, 0);

    return (y_pred - y_true) * (1.0 / m);
}

// CategoricalCrossEntropy
template<typename T>
double CategoricalCrossEntropyLoss<T>::compute(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    double result = 0.0;
    int m = y_pred.rows();
    if (m == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double pred_val = static_cast<double>(y_pred(i, j));
            double true_val = static_cast<double>(y_true(i, j));
            double epsilon = 1e-9;
            pred_val = std::max(epsilon, std::min(1.0 - epsilon, pred_val));

            result -= true_val * log(pred_val);
        }
    }

    return result / m;
}

template<typename T>
Matrix<T> CategoricalCrossEntropyLoss<T>::gradient(const Matrix<T>& y_pred, const Matrix<T>& y_true) const
{
    int m = y_pred.rows();
    if (m == 0) return Matrix<T>(0, 0);

    return (y_pred - y_true) * (1.0 / m);
}

template<typename T>
std::unique_ptr<LossFunction<T>> createLoss(LossType type)
{
    switch (type) {
        case LossType::MEAN_ABSOLUTE_ERROR:      return std::make_unique<MeanAbsoluteErrorLoss<T>>();
        case LossType::MEAN_SQUARED_ERROR:       return std::make_unique<MeanSquaredErrorLoss<T>>();
        case LossType::ROOT_MEAN_SQUARED_ERROR:  return std::make_unique<RootMeanSquaredErrorLoss<T>>();
        case LossType::BINARY_CROSS_ENTROPY:     return std::make_unique<BinaryCrossEntropyLoss<T>>();
        case LossType::CATEGORICAL_CROSS_ENTROPY:  return std::make_unique<CategoricalCrossEntropyLoss<T>>();
    }

    throw std::invalid_argument("Unsupported loss type.");
}

// Explicit instantiations
template class LossFunction<float>;
template class LossFunction<double>;

template class MeanAbsoluteErrorLoss<float>;
template class MeanAbsoluteErrorLoss<double>;

template class MeanSquaredErrorLoss<float>;
template class MeanSquaredErrorLoss<double>;

template class RootMeanSquaredErrorLoss<float>;
template class RootMeanSquaredErrorLoss<double>;

template class BinaryCrossEntropyLoss<float>;
template class BinaryCrossEntropyLoss<double>;

template class CategoricalCrossEntropyLoss<float>;
template class CategoricalCrossEntropyLoss<double>;

template std::unique_ptr<LossFunction<float>> createLoss<float>(LossType type);
template std::unique_ptr<LossFunction<double>> createLoss<double>(LossType type);
