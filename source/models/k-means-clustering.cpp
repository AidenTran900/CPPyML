#include "ml_lib/models/k-means-clustering.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

KMeansClustering::KMeansClustering(int k, int threshold, int max_iter)
    : k(k), threshold(threshold), max_iter(max_iter)
{}

void KMeansClustering::initCentroids(const Matrix &X) {
    int m = X.rows();
    int n = X.cols();

    centroids = Matrix(k, n); // row vectors

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row_dis(0, m - 1);

    switch(init_method) {
        case CLUSTER_INIT::RANDOM:
            for (int i = 0; i < k; i++) {
                int rand_ind = row_dis(gen);
                Matrix rand_row = X.row(rand_ind);
                for (int j = 0; j < n; j++) {
                    centroids(i, j) = rand_row(0, j);
                }   
            }
            break;
        case CLUSTER_INIT::SMART:
            break;
    }
}

int KMeansClustering::findNearestCentroid(const Matrix &row) {
    int cols = row.cols();
    int centroid_ind = 0;
    
    Matrix first_centroid = centroids.row(0);
    double min_distance = 0;
    for (int j = 0; j < cols; j++) {
        double diff = first_centroid(0, j) - row(0, j);
        min_distance += diff * diff;
    }
    
    for (int i = 1; i < centroids.rows(); i++) {
        Matrix centroid = centroids.row(i);
        double distance = 0;
        for (int j = 0; j < cols; j++) {
            double diff = centroid(0, j) - row(0, j);
            distance += diff * diff;
        }
        
        if (distance < min_distance) {
            min_distance = distance;
            centroid_ind = i;
        }
    }
    return centroid_ind;
}

Matrix KMeansClustering::updateCentroids(const Matrix &X, const std::vector<int> assignments) {
    int m = X.rows();
    int n = X.cols();

    Matrix new_centroids(k, n);
    std::vector<int> counts(k, 0);

    for (int i = 0; i < m; i++) {
        int cluster = assignments[i];
        counts[cluster]++;

        for (int j = 0; j < n; j++) {
            new_centroids(cluster, j) += X(i, j);
        }
    }

    for (int i = 0; i < k; i++) {
        if (counts[i] > 0) {
            for (int j = 0; j < n; j++) {
                new_centroids(i, j) /= counts[i];
            }
        }
    }

    return new_centroids;
}

void KMeansClustering::fit(const Matrix &X, const Matrix &y)
{
    int m = X.rows();
    int n = X.cols();

    initCentroids(X);
    
    int iterations = 0;
    bool converged = false;
    while (!converged && iterations < max_iter){
        // assign datapoints to centroids
        std::vector<int> assignments(m);
        for (int i = 0; i < m; i++) {
            assignments[i] = findNearestCentroid(X.row(i));
        }

        // update centroids
        Matrix new_centroids = updateCentroids(X, assignments);

        centroids = new_centroids;
    }
    
}

Matrix KMeansClustering::predict(const Matrix &X)
{
    int m = X.rows();
    Matrix predictions(m, 1);

    for (int i = 0; i < m; i++) {
        predictions(i, 0) = findNearestCentroid(X.row(i));
    }

    return predictions;
}
