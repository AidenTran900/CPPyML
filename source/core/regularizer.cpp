#include "ml_lib/core/regularizer.h"
#include <cmath>

template<typename T>
double L1Regularizer<T>::compute(const Matrix<T>& weights) const {
    double result = 0.0;
    for (int i = 0; i < weights.rows(); i++) {
        for (int j = 0; j < weights.cols(); j++) {
            result += this->lambda * std::abs(static_cast<double>(weights(i, j)));
        }
    }
    return result;
}

template<typename T>
Matrix<T> L1Regularizer<T>::gradient(const Matrix<T>& weights) const {
    return weights.sign() * this->lambda;
}

template<typename T>
double L2Regularizer<T>::compute(const Matrix<T>& weights) const {
    double result = 0.0;
    double half_lambda = (this->lambda/2);
    for (int i = 0; i < weights.rows(); i++) {
        for (int j = 0; j < weights.cols(); j++) {
            double weight = static_cast<double>(weights(i, j));
            result += half_lambda * weight * weight;
        }
    }
    return result;
}

template<typename T>
Matrix<T> L2Regularizer<T>::gradient(const Matrix<T>& weights) const {
    return weights * this->lambda;
}

template<typename T>
double NoRegularizer<T>::compute(const Matrix<T>&) const {
    return 0.0;
}

template<typename T>
Matrix<T> NoRegularizer<T>::gradient(const Matrix<T>& weights) const {
    return Matrix<T>(weights.rows(), weights.cols(), 0.0);
}

template<typename T>
std::unique_ptr<Regularizer<T>> createRegularizer(RegularizerType type, double lambda) {
    switch (type) {
        case RegularizerType::None:
            return std::make_unique<NoRegularizer<T>>();
        case RegularizerType::L1:
            return std::make_unique<L1Regularizer<T>>(lambda);
        case RegularizerType::L2:
            return std::make_unique<L2Regularizer<T>>(lambda);
        default:
            return std::make_unique<NoRegularizer<T>>();
    }
}

// Explicit instantiations
template class Regularizer<float>;
template class Regularizer<double>;

template class L1Regularizer<float>;
template class L1Regularizer<double>;

template class L2Regularizer<float>;
template class L2Regularizer<double>;

template class NoRegularizer<float>;
template class NoRegularizer<double>;

template std::unique_ptr<Regularizer<float>> createRegularizer<float>(RegularizerType type, double lambda);
template std::unique_ptr<Regularizer<double>> createRegularizer<double>(RegularizerType type, double lambda);
