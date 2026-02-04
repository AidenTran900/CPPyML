#include "masking.h"
#include <limits>
#include <cassert>

Masking::Masking(int dim, int tokens)
{
    mask = Matrix(dim, tokens);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < tokens; j++) {
            mask(i, j) = (j <= i) ? 0.0 : -std::numeric_limits<double>::infinity();
        }
    }
}

Matrix Masking::apply(const Matrix& x)
{
    assert(x.rows() == mask.rows() && x.cols() == mask.cols());

    Matrix result = x;
    int rows = x.rows();
    int cols = x.cols();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i, j) += mask(i, j);
        }
    }
    return result;
}

Masking::~Masking(){}
