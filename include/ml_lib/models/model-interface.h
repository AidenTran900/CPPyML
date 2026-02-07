#pragma once
#include "../math/matrix.h"

class Model {
    public:
        virtual ~Model() {}
};

template<typename T = double>
class GradientModelInterface : public Model {
    public:
        virtual Matrix<T> forward(const Matrix<T>& X) = 0;
        virtual void backward(const Matrix<T>& y_true) = 0;
        virtual void update() = 0;

        virtual ~GradientModelInterface() {}
};

class FitPredictModel : public Model {
    public:
        virtual void fit(const Matrix<>& X, const Matrix<>& y) = 0;
        virtual Matrix<> predict(const Matrix<>& X) = 0;

        virtual ~FitPredictModel() {}
};
