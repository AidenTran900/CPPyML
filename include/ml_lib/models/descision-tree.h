#pragma once
#include "../math/matrix.h"
#include "gradient-model.h"
#include <vector>

enum IMPURITY {
    GINI,
    ENTROPY
};

class Node {
    public:
        Node* left = nullptr;
        Node* right = nullptr;

        int feature = 0;
        double threshold = 0;
        double value = 0;

        Node() {}

        Node(Node* left, Node* right, int feature, double threshold, double value)
            : left(left), right(right), feature(feature), threshold(threshold), value(value) {}

        bool isLeaf() {
            return left == nullptr && right == nullptr;
        };
};

class DescisionTree {
    private:
        Matrix X_train;
        Matrix Y_train;

        Node* root_node;

        IMPURITY impurity = GINI;
        int max_depth = 100;
        int min_samples_split = 2;
        int features = 0;

        std::pair<int, int> countClasses(const std::vector<int>& indices);
        double calculateImpurity(std::vector<int> indices);
        void evaluateNode(Node* node, std::vector<int> indices, int depth);
        int getMajorityClass(std::vector<int> indices);

    public:
        DescisionTree();

        void fit(const Matrix& X, const Matrix& y);
        Matrix predict(const Matrix& X);
};