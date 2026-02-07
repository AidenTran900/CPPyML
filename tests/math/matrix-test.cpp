
#include <fmt/format.h>
#include "ml_lib/math/matrix.h"

int matrixTest()
{
    // Test matrix operations (double - default)
    Matrix<> m1 = Matrix<>(std::vector<std::vector<double>>{{1, 2},
                                                        {3, 4}});
    Matrix<> m2 = Matrix<>(std::vector<std::vector<double>>{{5, 6},
                                                        {7, 8}});
    Matrix<> m3 = Matrix<>(std::vector<std::vector<double>>{{-1, 4, -2},
                                                        {-4, 6, 1},
                                                        {-6, -6, -2}});

    fmt::print("Add:\n");
    (m1 + m2).print();

    fmt::print("Multiply:\n");
    (m1 * m2).print();

    fmt::print("REF of m3:\n");
    EliminationResult<> m3_elim = Matrix<>::forwardElimination(m3);
    m3_elim.matrix.print();

    fmt::print("RREF of m3:\n");
    Matrix<>::backwardElimination(m3_elim.matrix).matrix.print();

    fmt::print("Inverse of m3:\n");
    m3.inverse().print();

    fmt::print("Determinant m3:\n");
    fmt::print("{}\n", m3.determinant());

    fmt::print("Transpose m3:\n");
    m3.transpose().print();

    // Test float matrix (MatrixF32)
    fmt::print("\n--- Float (f32) Matrix Tests ---\n");
    MatrixF32 f1 = MatrixF32(std::vector<std::vector<float>>{{1.0f, 2.0f},
                                                              {3.0f, 4.0f}});
    MatrixF32 f2 = MatrixF32(std::vector<std::vector<float>>{{5.0f, 6.0f},
                                                              {7.0f, 8.0f}});
    fmt::print("Float Add:\n");
    (f1 + f2).print();

    fmt::print("Float Multiply:\n");
    (f1 * f2).print();

    fmt::print("Float Transpose:\n");
    f1.transpose().print();

    return 0;
}
