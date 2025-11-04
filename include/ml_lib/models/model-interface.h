#pragma once
#include "../math/matrix.h"

class Model {
    public:
        virtual Matrix forward(const Matrix& X) = 0;
        virtual void backward(const Matrix& y_true) = 0;
        virtual void update() = 0;

        virtual ~Model() {}
};
