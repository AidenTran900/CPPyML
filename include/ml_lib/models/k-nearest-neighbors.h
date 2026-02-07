#pragma once
#include "../math/matrix.h"
#include "model-interface.h"
#include <vector>

enum DIST_METRIC {
    EUCLIDEAN,
    MANHATTAN
};

class KNearestNeighbors : public FitPredictModel {
    private:
        int k;
        Matrix<> X_train;
        Matrix<> y_train;
        DIST_METRIC distance_metric;

        double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2);
        std::vector<int> findKNearest(const std::vector<double>& test_point);

    public:
        KNearestNeighbors(int k = 3, DIST_METRIC metric = DIST_METRIC::EUCLIDEAN);

        void fit(const Matrix<>& X, const Matrix<>& y) override;
        Matrix<> predict(const Matrix<>& X) override;
};
