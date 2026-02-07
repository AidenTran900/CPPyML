#include "ml_lib/models/support-vector-machine.h"
#include <cmath>
#include <algorithm>
#include <limits>

// The math is making me go insane

SupportVectorMachine::SupportVectorMachine(
    double C,
    double gamma,
    KERNEL kernel,
    int degree,
    double tolerance,
    int max_iter,
    double coef0
)
    : C(C), gamma(gamma), kernel_type(kernel), degree(degree),
      tolerance(tolerance), max_iter(max_iter), coef0(coef0), bias(0.0)
{}

double SupportVectorMachine::kernel(const Matrix<> &X1, const Matrix<> &X2) {
    switch(this->kernel_type) {
        case KERNEL::LINEAR:
            return X1.dot(X2);
        case KERNEL::POLYNOMIAL:
            return pow(this->gamma * X1.dot(X2) + this->coef0, this->degree);
        case KERNEL::RBF: {
            Matrix<> diff = X1 - X2;
            return exp(-this->gamma * diff.dot(diff));
        }
        case KERNEL::SIGMOID:
            return tanh(this->gamma * X1.dot(X2) + this->coef0);
        default:
            return X1.dot(X2);
    }
}

double SupportVectorMachine::decision(const Matrix<>& X) {
    double result = 0.0;
    int m = this->support_vectors.rows();

    for (int i = 0; i < m; i++) {
        result += this->support_alphas[i] * this->support_labels(i, 0) * kernel(this->support_vectors.row(i), X);
    }
    return result + bias;
}

double SupportVectorMachine::decisionCached(int idx, const Matrix<>& K_cache, const std::vector<double>& alphas, const Matrix<>& Y_train) {
    double result = 0.0;
    int m = Y_train.rows();

    for (int i = 0; i < m; i++) {
        if (alphas[i] > 0) {
            result += alphas[i] * Y_train(i, 0) * K_cache(i, idx);
        }
    }
    return result + bias;
}

int SupportVectorMachine::takeStep(int I1, int I2, const Matrix<>& X_train, const Matrix<>& Y_train, const Matrix<>& K_cache, std::vector<double>& alphas, std::vector<double>& errors) {
    if (I1 == I2) return 0;

    double alpha1 = alphas[I1];
    double alpha2 = alphas[I2];
    double Y1 = Y_train(I1, 0);
    double Y2 = Y_train(I2, 0);
    double E1 = errors[I1];
    double E2 = errors[I2];

    // Bounds
    double L, H;
    if (Y1 != Y2) {
        L = std::max(0.0, alpha2 - alpha1);
        H = std::min(C, C + alpha2 - alpha1);
    } else {
        L = std::max(0.0, alpha2 + alpha1 - C);
        H = std::min(C, alpha2 + alpha1);
    }
    if (L == H) return 0; // No room to move

    double k11 = K_cache(I1, I1); // similarity of 1 to itself
    double k12 = K_cache(I1, I2); // similarity of 1 to 2
    double k22 = K_cache(I2, I2); // similarity of 2 to itself
    double eta = 2 * k12 - k11 - k22;

    if (eta >= 0) return 0; // flat/convex, unable to find max

    double alpha2_new = alpha2 - Y2 * (E1 - E2) / eta;
    alpha2_new = std::clamp(alpha2_new, L, H);

    if (std::abs(alpha2_new - alpha2) < tolerance) return 0; // Delta too small

    double alpha1_new = alpha1 + Y1 * Y2 * (alpha2 - alpha2_new);

    double bias1 = bias - E1 - Y1 * (alpha1_new - alpha1) * k11 - Y2 * (alpha2_new - alpha2) * k12;
    double bias2 = bias - E2 - Y1 * (alpha1_new - alpha1) * k12 - Y2 * (alpha2_new - alpha2) * k22;

    if (0 < alpha1_new && alpha1_new < C)
        bias = bias1;
    else if (0 < alpha2_new && alpha2_new < C)
        bias = bias2;
    else
        bias = (bias1 + bias2) / 2;

    alphas[I1] = alpha1_new;
    alphas[I2] = alpha2_new;

    int m = X_train.rows();
    for (int i = 0; i < m; i++) {
        errors[i] = decisionCached(i, K_cache, alphas, Y_train) - Y_train(i, 0);
    }

    return 1;
}

int SupportVectorMachine::examineExample(int I2, const Matrix<>& X_train, const Matrix<>& Y_train, Matrix<>& K_cache, std::vector<double>& alphas, std::vector<double>& errors) {
    double alpha2 = alphas[I2];
    double r2 = errors[I2] * Y_train(I2, 0);
    double E2 = errors[I2];

    // KKT Violations
    if ((r2 < -tolerance && alpha2 < C) || (r2 > tolerance && alpha2 > 0)) {
        int I = -1;
        double max_diff = 0;

        int m = X_train.rows();

        for (int i = 0; i < m; i++) { // Argmax
            if (alphas[i] > 0 && alphas[i] < C) {
                double diff = abs(errors[i] - E2);
                if (diff > max_diff) {
                    max_diff = diff;
                    I = i;
                }
            }
        }

        if (I >= 0 && takeStep(I, I2, X_train, Y_train, K_cache, alphas, errors)) return 1;

        for (int i = 0; i < m; i++) {
            if (takeStep(i, I2, X_train, Y_train, K_cache, alphas, errors)) return 1;
        }

    }

    return 0;
}

void SupportVectorMachine::fit(const Matrix<> &X, const Matrix<> &Y)
{
    int m = X.rows();
    int n = X.cols();

    std::vector<double> alphas(m, 0.0);
    std::vector<double> errors(m);
    bias = 0.0;

    Matrix<> K_cache(m, m);
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double k_val = kernel(X.row(i), X.row(j));
            K_cache(i, j) = k_val;
            K_cache(j, i) = k_val;
        }
    }

    for (int i = 0; i < m; i++) {
        errors[i] = -Y(i, 0);
    }

    int iterations = 0;
    int num_changed = 0;
    bool examine_all = true;

    while ((num_changed > 0 || examine_all) && iterations < max_iter) {
        num_changed = 0;

        for (int i = 0; i < m; i++) {
            if (examine_all || (alphas[i] > 0 && alphas[i] < C))
            num_changed += examineExample(i, X, Y, K_cache, alphas, errors);
        }

        if (examine_all)
            examine_all = false;
        else if (num_changed == 0)
            examine_all = true;

        iterations++;
    }

    int sv_count = 0;
    for (int i = 0; i < m; i++) {
        if (alphas[i] > 0) sv_count++;
    }

    support_vectors = Matrix<>(sv_count, n);
    support_labels = Matrix<>(sv_count, 1);
    support_alphas = std::vector<double>(sv_count);

    int sv_idx = 0;
    for (int i = 0; i < m; i++) {
        if (alphas[i] > 0) {
            for (int j = 0; j < n; j++) {
                support_vectors(sv_idx, j) = X(i, j);
            }
            support_labels(sv_idx, 0) = Y(i, 0);
            support_alphas[sv_idx] = alphas[i];
            sv_idx++;
        }
    }
}

Matrix<> SupportVectorMachine::predict(const Matrix<> &X)
{
    int m = X.rows();
    Matrix<> predictions(m, 1);

    for (int i = 0; i < m; i++) {
        predictions(i, 0) = decision(X.row(i));
    }

    return predictions.sign();
}
