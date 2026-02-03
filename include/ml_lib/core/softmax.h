#pragma once
#include "../math/matrix.h"
#include <cmath>

namespace Softmax {

    inline Matrix apply(const Matrix& x)
    {
        Matrix result(x.rows(), x.cols());

        for (int i = 0; i < x.rows(); i++) {
            double max_val = x(i, 0);
            for (int j = 1; j < x.cols(); j++) {
                if (x(i, j) > max_val) {
                    max_val = x(i, j);
                }
            }

            double sum = 0.0;
            for (int j = 0; j < x.cols(); j++) {
                result(i, j) = std::exp(x(i, j) - max_val);
                sum += result(i, j);
            }

            for (int j = 0; j < x.cols(); j++) {
                result(i, j) /= sum;
            }
        }
        return result;
    }

    inline Matrix applyColumn(const Matrix& x)
    {
        Matrix result(x.rows(), x.cols());

        double max_val = x(0, 0);
        for (int i = 1; i < x.rows(); i++) {
            if (x(i, 0) > max_val) {
                max_val = x(i, 0);
            }
        }

        double sum = 0.0;
        for (int i = 0; i < x.rows(); i++) {
            result(i, 0) = std::exp(x(i, 0) - max_val);
            sum += result(i, 0);
        }

        for (int i = 0; i < x.rows(); i++) {
            result(i, 0) /= sum;
        }

        return result;
    }

    inline Matrix derivative(const Matrix& softmax_out, const Matrix& grad)
    {
        Matrix result(grad.rows(), grad.cols());

        for (int i = 0; i < grad.rows(); i++) {
            double dot_product = 0.0;
            for (int j = 0; j < grad.cols(); j++) {
                dot_product += grad(i, j) * softmax_out(i, j);
            }

            for (int j = 0; j < grad.cols(); j++) {
                result(i, j) = softmax_out(i, j) * (grad(i, j) - dot_product);
            }
        }
        return result;
    }

    inline Matrix derivativeColumn(const Matrix& softmax_out, const Matrix& grad)
    {
        int n = softmax_out.rows();
        Matrix result(n, 1);

        double dot_product = 0.0;
        for (int i = 0; i < n; i++) {
            dot_product += grad(i, 0) * softmax_out(i, 0);
        }

        for (int i = 0; i < n; i++) {
            result(i, 0) = softmax_out(i, 0) * (grad(i, 0) - dot_product);
        }

        return result;
    }

}
