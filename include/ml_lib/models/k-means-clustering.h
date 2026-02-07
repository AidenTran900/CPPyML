#pragma once
#include "../math/matrix.h"
#include "model-interface.h"
#include <vector>

enum CLUSTER_INIT {
    RANDOM,
    SMART,
};

class KMeansClustering : public FitPredictModel {
    private:
        int k;

        Matrix<> centroids;
        int threshold;
        int max_iter;

        CLUSTER_INIT init_method;

    public:
        KMeansClustering(int k = 3, int threshold = 3, int max_iter = 100);

        void initCentroids(const Matrix<>& X);
        void fit(const Matrix<>& X, const Matrix<>& y) override;
        Matrix<> predict(const Matrix<>& X) override;

        Matrix<> updateCentroids(const Matrix<>& X, const std::vector<int> assignments);
        int findNearestCentroid(const Matrix<> &row);

};
