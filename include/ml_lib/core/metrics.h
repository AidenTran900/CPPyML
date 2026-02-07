#pragma once
#include "../math/matrix.h"
#include <vector>

namespace metrics {

    struct ROCResult {
        std::vector<double> TPR;
        std::vector<double> FPR;
        double AUC;
    };

    double r2(const Matrix<>& y_true, const Matrix<>& y_pred);
    double adjustedR2(const Matrix<>& y_true, const Matrix<>& y_pred, int num_predictors);
    double mse(const Matrix<>& y_true, const Matrix<>& y_pred);
    double rmse(const Matrix<>& y_true, const Matrix<>& y_pred);
    double mae(const Matrix<>& y_true, const Matrix<>& y_pred);

    Matrix<> confusionMatrix(const Matrix<>& y_true, const Matrix<>& y_pred);

    double accuracy(const Matrix<>& confusion);
    double precision(const Matrix<>& confusion);
    double recall(const Matrix<>& confusion);
    double fpr(const Matrix<>& confusion);
    double f1Score(const Matrix<>& confusion);

    ROCResult rocCurve(const Matrix<>& y_true, const Matrix<>& y_pred, double resolution = 0.01);
    double auc(const ROCResult& roc_result);

}