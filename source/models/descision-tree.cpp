#include "ml_lib/models/descision-tree.h"
#include <cmath>
#include <algorithm>
#include <limits>

DescisionTree::DescisionTree()
{
    root_node = new Node();
}

void DescisionTree::setFeatureIndices(const std::vector<int>& indices)
{
    feature_indices = indices;
}

DescisionTree::~DescisionTree()
{
    deleteTree(root_node);
}

void DescisionTree::deleteTree(Node* node)
{
    if (node == nullptr) return;

    deleteTree(node->left);
    deleteTree(node->right);
    delete node;
}

std::pair<int, int> DescisionTree::countClasses(const std::vector<int>& indices) {
    int count_0 = 0;
    int count_1 = 0;
    for (int i : indices) {
        if (Y_train(i, 0) == 0) {
            count_0++;
        } else {
            count_1++;
        }
    }
    return {count_0, count_1};
}

double DescisionTree::calculateImpurity(std::vector<int> indices) {
    auto [count_0, count_1] = countClasses(indices);

    int total = indices.size();
    double p0 = (double) count_0 / total;
    double p1 = (double) count_1 / total;

    switch(impurity) {
        case (IMPURITY::GINI):
            return 1 - (p0*p0 + p1*p1);
        case (IMPURITY::ENTROPY):
            double entropy = 0;
            if (p0 > 0) entropy -= p0 * log2(p0);
            if (p1 > 0) entropy -= p1 * log2(p1);
            return entropy;
    }
    return 0;
}

int DescisionTree::getMajorityClass(std::vector<int> indices) {
    auto [count_0, count_1] = countClasses(indices);
    return (count_1 > count_0) ? 1 : 0;
}

void DescisionTree::evaluateNode(Node* node, std::vector<int> indices, int depth) {
    int n = X_train.cols();

    node->value = getMajorityClass(indices);

    if (indices.size() < min_samples_split || depth >= max_depth) {
        return;
    }

    double best_threshold = 0;
    int best_feature = 0;
    double min_impurity = -1;
    std::vector<int> best_left_indices;
    std::vector<int> best_right_indices;

    std::vector<int> features_to_try = feature_indices.empty() ? std::vector<int>() : feature_indices;
    if (features_to_try.empty()) {
        for (int i = 0; i < n; i++) {
            features_to_try.push_back(i);
        }
    }

    for (int j : features_to_try) {
        for (int idx : indices) {
            double threshold = X_train(idx, j);
            std::vector<int> left_indices;
            std::vector<int> right_indices;

            for (int x : indices) {
                if (X_train(x, j) > threshold) {
                    right_indices.push_back(x);
                } else {
                    left_indices.push_back(x);
                }
            }

            if (left_indices.empty() || right_indices.empty()) {
                continue;
            }

            double left_impurity = calculateImpurity(left_indices);
            double right_impurity = calculateImpurity(right_indices);
            double total = left_indices.size() + right_indices.size();
            double weighted_impurity = (left_indices.size() * left_impurity + right_indices.size() * right_impurity) / total;

            if ((weighted_impurity < min_impurity) || (min_impurity == -1)) {
                min_impurity = weighted_impurity;
                best_threshold = threshold;
                best_feature = j;
                best_left_indices = left_indices;
                best_right_indices = right_indices;
            }
        }
    }

    if (min_impurity == -1 || min_impurity == 0) {
        return;
    }

    node->threshold = best_threshold;
    node->feature = best_feature;

    node->left = new Node();
    node->right = new Node();

    evaluateNode(node->left, best_left_indices, depth + 1);
    evaluateNode(node->right, best_right_indices, depth + 1);
}

void DescisionTree::fit(const Matrix &X, const Matrix &Y)
{
    X_train = X;
    Y_train = Y;

    std::vector<int> indices;
    for (int i = 0; i < X.rows(); i++) {
        indices.push_back(i);
    }

    evaluateNode(root_node, indices, 0);
}

Matrix DescisionTree::predict(const Matrix &X)
{
    int m = X.rows();
    Matrix predictions = Matrix(m, 1, 0);

    for (int i = 0; i < m; i++) {
        Node* current_node = root_node;
        while (!current_node->isLeaf()) {
            if (X(i, current_node->feature) > current_node->threshold) {
                current_node = current_node->right;
            } else {
                current_node = current_node->left;
            }
        }
        predictions(i, 0) = current_node->value;
        
    }
    return predictions;
}
