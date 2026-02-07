#include "ml_lib/core/masking.h"
#include <limits>
#include <cassert>

template<typename T>
Masking<T>::Masking(int dim, int tokens)
{
    mask = Matrix<T>(dim, tokens);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < tokens; j++) {
            mask(i, j) = (j <= i) ? static_cast<T>(0.0) : -std::numeric_limits<T>::infinity();
        }
    }
}

template<typename T>
Matrix<T> Masking<T>::apply(const Matrix<T>& x)
{
    assert(x.rows() == mask.rows() && x.cols() == mask.cols());

    Matrix<T> result = x;
    int rows = x.rows();
    int cols = x.cols();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i, j) += mask(i, j);
        }
    }
    return result;
}

template<typename T>
Masking<T>::~Masking(){}

template class Masking<float>;
template class Masking<double>;
