#pragma once
#include "../math/matrix.h"
#include <memory>

enum class RegularizerType {
    None,
    L1,
    L2
};

class Regularizer {
    protected:
        double lambda;
    public:
        virtual double compute(const Matrix& weights) const = 0;
        virtual Matrix gradient(const Matrix& weights) const = 0;

        Regularizer(double l) : lambda(l) {}

        virtual ~Regularizer() {}
};

class L1Regularizer : public Regularizer {
    public:
        L1Regularizer(double l) : Regularizer(l) {}
        double compute(const Matrix& weights) const override;
        Matrix gradient(const Matrix& weights) const override;
};

class L2Regularizer : public Regularizer {
    public:
        L2Regularizer(double l) : Regularizer(l) {}
        double compute(const Matrix& weights) const override;
        Matrix gradient(const Matrix& weights) const override;
};

class NoRegularizer : public Regularizer {
    public:
        NoRegularizer() : Regularizer(0.0) {}
        double compute(const Matrix&) const override;
        Matrix gradient(const Matrix& weights) const override;
};

Regularizer* createRegularizer(RegularizerType type, double lambda);