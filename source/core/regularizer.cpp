#include "ml_lib/core/regularizer.h"
#include <cmath>

double L1Regularizer::compute(const Matrix& weights) const {
    double result = 0.0;
    for (int i = 0; i < weights.rows(); i++) {
        for (int j = 0; j < weights.cols(); j++) {
            result += lambda * std::abs(weights(i, j));
        }
    }
    return result;
}

Matrix L1Regularizer::gradient(const Matrix& weights) const {
    return weights.sign().scale(lambda);
}

double L2Regularizer::compute(const Matrix& weights) const {
    double result = 0.0;
    double half_lambda = (lambda/2);
    for (int i = 0; i < weights.rows(); i++) {
        for (int j = 0; j < weights.cols(); j++) {
            double weight = weights(i, j);
            result += half_lambda * weight * weight;
        }
    }
    return result;
}

Matrix L2Regularizer::gradient(const Matrix& weights) const {
    return weights.scale(lambda);
}

double NoRegularizer::compute(const Matrix&) const {
    return 0.0;
}

Matrix NoRegularizer::gradient(const Matrix& weights) const {
    return Matrix(weights.rows(), weights.cols(), 0.0);
}

Regularizer* createRegularizer(RegularizerType type, double lambda) {
    switch (type) {
        case RegularizerType::None:
            return new NoRegularizer();
        case RegularizerType::L1:
            return new L1Regularizer(lambda);
        case RegularizerType::L2:
            return new L2Regularizer(lambda);
        default:
            return new NoRegularizer();
    }
}
