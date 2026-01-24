#include "ml_lib/core/metrics.h"
#include <cmath>

namespace metrics {

double r2(const Matrix& y_true, const Matrix& y_pred)
{
    int rows = y_pred.rows();
    int cols = y_pred.cols();

    double SSres = 0.0; // Residual
    double SStot = 0.0; // Total

    double mean = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mean += y_true(i, j);
        }
    }
    mean /= (rows * cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff1 = y_pred(i, j) - y_true(i, j);
            double diff2 = y_true(i, j) - mean;
            SSres += diff1 * diff1;
            SStot += diff2 * diff2;
        }
    }

    if (SStot < 1e-9) { return 0; }

    return 1 - (SSres/SStot);
}

double adjustedR2(const Matrix& y_true, const Matrix& y_pred, int num_predictors)
{
    double r2_val = r2(y_true, y_pred);
    int n = y_true.rows();

    return 1 - (1 - r2_val) * ((n - 1.0) / (n - num_predictors - 1));
}

double mse(const Matrix& y_true, const Matrix& y_pred)
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

double rmse(const Matrix& y_true, const Matrix& y_pred)
{
    return sqrt(mse(y_true, y_pred));
}

double mae(const Matrix& y_true, const Matrix& y_pred)
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

Matrix confusionMatrix(const Matrix& y_true, const Matrix& y_pred)
{
    Matrix confusion = Matrix(2, 2, 0);
    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double pred_val = y_pred(i, j);
            double true_val = y_true(i, j);

            if (pred_val == true_val) {
                if (true_val == 1) {
                    confusion(0, 0) += 1; // TP
                } else {
                    confusion(1, 1) += 1; // TN
                }
            } else {
                if (true_val == 1) {
                    confusion(0, 1) += 1; // FN
                } else {
                    confusion(1, 0) += 1; // FP
                }
            }
        }
    }
    return confusion;
}

double accuracy(const Matrix& confusion)
{
    int TP = static_cast<int>(confusion(0, 0));
    int TN = static_cast<int>(confusion(1, 1));
    int FP = static_cast<int>(confusion(1, 0));
    int FN = static_cast<int>(confusion(0, 1));

    int total = TP + TN + FP + FN;
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(TP + TN) / total;
}

double precision(const Matrix& confusion)
{
    int TP = static_cast<int>(confusion(0, 0));
    int FP = static_cast<int>(confusion(1, 0));

    if (TP + FP == 0) {
        return 0.0;
    }
    return static_cast<double>(TP) / (TP + FP);
}

double recall(const Matrix& confusion)
{
    int TP = static_cast<int>(confusion(0, 0));
    int FN = static_cast<int>(confusion(0, 1));

    if (TP + FN == 0) {
        return 0.0;
    }
    return static_cast<double>(TP) / (TP + FN);
}

double fpr(const Matrix& confusion)
{
    int TN = static_cast<int>(confusion(1, 1));
    int FP = static_cast<int>(confusion(1, 0));

    if (FP + TN == 0) {
        return 0.0;
    }
    return static_cast<double>(FP) / (FP + TN);
}

double f1Score(const Matrix& confusion)
{
    double recall_val = recall(confusion);
    double precision_val = precision(confusion);

    if (precision_val + recall_val == 0.0) {
        return 0.0;
    }
    return 2.0 * (precision_val * recall_val) / (precision_val + recall_val);
}

ROCResult rocCurve(const Matrix& y_true, const Matrix& y_pred, double resolution)
{
    ROCResult result;

    for (double threshold = 1.0; threshold >= 0; threshold -= resolution) {
        Matrix y_predThresholded = Matrix(y_pred.rows(), y_pred.cols(), 0.0);
        for (int i = 0; i < y_pred.rows(); i++) {
            for (int j = 0; j < y_pred.cols(); j++) {
                y_predThresholded(i, j) = (y_pred(i, j) >= threshold) ? 1.0 : 0.0;
            }
        }

        Matrix confusion = confusionMatrix(y_true, y_predThresholded);

        double tpr = recall(confusion);
        double fpr_val = fpr(confusion);

        result.TPR.push_back(tpr);
        result.FPR.push_back(fpr_val);
    }

    result.AUC = auc(result);

    return result;
}

double auc(const ROCResult& roc_result)
{
    double area = 0;
    for (size_t i = 1; i < roc_result.FPR.size(); i++) {
        area += (roc_result.TPR[i] + roc_result.TPR[i-1]) / 2.0 * (roc_result.FPR[i] - roc_result.FPR[i - 1]);
    }
    return area;
}

}
