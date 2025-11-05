#include <fmt/format.h>
#include "ml_lib/models/linear-regression.h"
#include "ml_lib/core/loss.h"
#include "ml_lib/core/optimizer.h"
#include "ml_lib/core/regularizer.h"

int linearRegTest()
{
    LinearRegression model(
        2, 
        createLoss(LossType::MSE), 
        createOptimizer(OptimizerType::BATCH, 0.01), 
        createRegularizer(RegularizerType::L2, 0.01)
    );

    Matrix X = std::vector<std::vector<double>>{
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0}
    };

    Matrix y_true = std::vector<std::vector<double>>{
        {5.0},
        {7.0},
        {9.0}
    };

    for (int epoch = 0; epoch < 1000; epoch++) {
        Matrix y_pred = model.forward(X);
        double loss = model.computeLoss(y_pred, y_true);
        model.backward(y_true);
        model.update();

        if (epoch % 10 == 0) {
            fmt::print("Epoch {}: Loss = {}\n", epoch, loss);
        }
    }

    return 0;
}
