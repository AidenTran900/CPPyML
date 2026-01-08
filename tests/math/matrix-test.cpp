
#include <fmt/format.h>
#include "ml_lib/math/matrix.h"

int matrixTest()
{
    // Test matrix operations
    Matrix m1 = Matrix(std::vector<std::vector<double>>{{1, 2},
                                                        {3, 4}});
    Matrix m2 = Matrix(std::vector<std::vector<double>>{{5, 6},
                                                        {7, 8}});
    Matrix m3 = Matrix(std::vector<std::vector<double>>{{-1, 4, -2},
                                                        {-4, 6, 1},
                                                        {-6, -6, -2}});

    fmt::print("Add:\n");
    (m1 + m2).print();

    fmt::print("Multiply:\n");
    (m1 * m2).print();

    fmt::print("REF of m3:\n");
    EliminationResult m3_elim = Matrix::forwardElimination(m3);
    m3_elim.matrix.print();

    fmt::print("RREF of m3:\n"); // Prints diagonal matrix with 1s and 0s duh idk what im doing
    Matrix::backwardElimination(m3_elim.matrix).matrix.print();

    fmt::print("Inverse of m3:\n");
    m3.inverse().print();

    fmt::print("Determinant m3:\n");
    fmt::print("{}\n", m3.determinant());

    fmt::print("Transpose m3:\n");
    m3.transpose().print();

    return 0;
}
