#pragma once
#include "../math/matrix.h"

template<typename T = double>
class Masking {
    private:
        Matrix<T> mask;
    public:
        Masking(int dim, int tokens);
        ~Masking();
        Matrix<T> apply(const Matrix<T>& x);

};
