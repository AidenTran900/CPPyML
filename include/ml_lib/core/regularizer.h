#pragma once
#include "../math/matrix.h"
#include <memory>

enum class RegularizerType {
    None,
    L1,
    L2
};

template<typename T = double>
class Regularizer {
    protected:
        double lambda;
    public:
        virtual double compute(const Matrix<T>& weights) const = 0;
        virtual Matrix<T> gradient(const Matrix<T>& weights) const = 0;

        Regularizer(double l) : lambda(l) {}

        virtual ~Regularizer() {}
};

template<typename T = double>
class L1Regularizer : public Regularizer<T> {
    public:
        L1Regularizer(double l) : Regularizer<T>(l) {}
        double compute(const Matrix<T>& weights) const override;
        Matrix<T> gradient(const Matrix<T>& weights) const override;
};

template<typename T = double>
class L2Regularizer : public Regularizer<T> {
    public:
        L2Regularizer(double l) : Regularizer<T>(l) {}
        double compute(const Matrix<T>& weights) const override;
        Matrix<T> gradient(const Matrix<T>& weights) const override;
};

template<typename T = double>
class NoRegularizer : public Regularizer<T> {
    public:
        NoRegularizer() : Regularizer<T>(0.0) {}
        double compute(const Matrix<T>&) const override;
        Matrix<T> gradient(const Matrix<T>& weights) const override;
};

template<typename T = double>
std::unique_ptr<Regularizer<T>> createRegularizer(RegularizerType type, double lambda);
