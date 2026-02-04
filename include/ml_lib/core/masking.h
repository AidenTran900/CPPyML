#pragma once
#include "../math/matrix.h"

class Masking {
    private:
        Matrix mask;
    public:
        Masking(int dim, int tokens);
        ~Masking();
        Matrix apply(const Matrix& x);

}