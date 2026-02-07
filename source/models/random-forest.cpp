#include "ml_lib/models/descision-tree.h"
#include "ml_lib/models/random-forest.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

RandomForest::RandomForest() {}

RandomForest::~RandomForest() {
    for (DescisionTree* tree : trees) {
        delete tree;
    }
    trees.clear();
}

void RandomForest::fit(const Matrix<> &X, const Matrix<> &Y)
{
    int m = X.rows();
    int n = X.cols();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row_dis(0, m - 1);

    int max_features = std::max(1, (int)std::sqrt(n));

    for (int t = 0; t < n_estimators; t++) {
        Matrix<> X_B(m, n);
        Matrix<> Y_B(m, 1);

        for (int i = 0; i < m; i++) {
            int random_idx = row_dis(gen);
            for (int j = 0; j < n; j++) {
                X_B(i, j) = X(random_idx, j);
            }
            Y_B(i, 0) = Y(random_idx, 0);
        }

        std::vector<int> all_features(n);
        for (int i = 0; i < n; i++) {
            all_features[i] = i;
        }
        std::shuffle(all_features.begin(), all_features.end(), gen); // randomly get features
        std::vector<int> selected_features(all_features.begin(), all_features.begin() + max_features);

        DescisionTree* tree = new DescisionTree();
        tree->setFeatureIndices(selected_features);
        tree->fit(X_B, Y_B);
        trees.push_back(tree);
    }
}

Matrix<> RandomForest::predict(const Matrix<> &X)
{
    int m = X.rows();
    Matrix<> predictions = Matrix<>(m, 1, 0);

    for (DescisionTree* tree : trees) {
        predictions = predictions + tree->predict(X);
    }

    predictions = predictions * (1.0/n_estimators);
    return predictions;
}
