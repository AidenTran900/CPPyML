#include "ml_lib/models/k-nearest-neighbors.h"
#include <cmath>
#include <algorithm>
#include <limits>

KNearestNeighbors::KNearestNeighbors(int k, DIST_METRIC metric)
    : k(k), distance_metric(metric)
{}

double KNearestNeighbors::calculateDistance(const std::vector<double> &point1, const std::vector<double> &point2)
{
    if (point1.size() != point2.size()) {
        throw std::invalid_argument("Points must have same dimensions");
    }

    double distance = 0.0;
    int n = point1.size();

    if (distance_metric == DIST_METRIC::EUCLIDEAN) {
        for (int i = 0; i < n; i++) {
            double diff = point1[i] - point2[i];
            distance += diff * diff;
        }
        distance = std::sqrt(distance);
    }
    else if (distance_metric == DIST_METRIC::MANHATTAN) {
        for (int i = 0; i < n; i++) {
            distance += std::abs(point1[i] - point2[i]);
        }
    }

    return distance;
}

std::vector<int> KNearestNeighbors::findKNearest(const std::vector<double> &test_point)
{
    int k = this->k;
    int X_rows = X_train.rows();

    if (X_rows < k) {
        throw std::invalid_argument("Not enough points! Must have at least " + std::to_string(k) + ".");
    }

    int nearest_ind[k];
    std::vector<double> neighbor_dist = std::vector<double>(k, -1.0);
    std::vector<int> neighbor_ind = std::vector<int>(k, -1);

    for (size_t i = 0; i < X_rows; i++) {
        std::vector<double> row_vec = this->X_train.getRowVector(i);
        double dist = calculateDistance(row_vec, test_point);
        for (int n = 0; n < k; n++) {
            double current_dist = neighbor_dist[n];
            if (dist < current_dist || current_dist < 0) {
                neighbor_dist[n] = dist;
                neighbor_ind[n] = i;
            }
        }
    }
    return neighbor_ind;
}


void KNearestNeighbors::fit(const Matrix<> &X, const Matrix<> &y)
{
    X_train = X;
    y_train = y;
}

Matrix<> KNearestNeighbors::predict(const Matrix<> &X)
{
    int n_samples = X.rows();
    Matrix<> predictions(n_samples, 1);

    for (int i = 0; i < n_samples; i++) {
        std::vector<double> test_point = X.getRowVector(i);
        std::vector<int> neighbor_indices = findKNearest(test_point);

        double sum = 0.0;
        for (int idx : neighbor_indices) {
            sum += y_train(idx, 0);
        }
        predictions(i, 0) = sum / k;
    }

    return predictions;
}
