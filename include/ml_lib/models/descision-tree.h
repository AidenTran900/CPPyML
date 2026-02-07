#pragma once
#include "../math/matrix.h"
#include "model-interface.h"
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

class DescisionTree : public FitPredictModel {
    private:
        Node* root_node;

        IMPURITY impurity = GINI;
        int max_depth = 100;
        int min_samples_split = 2;
        int features = 0;
        std::vector<int> feature_indices;

        std::pair<int, int> countClasses(const Matrix<>& Y, const std::vector<int>& indices);
        double calculateImpurity(const Matrix<>& Y, std::vector<int> indices);
        void evaluateNode(const Matrix<>& X, const Matrix<>& Y, Node* node, std::vector<int> indices, int depth);
        int getMajorityClass(const Matrix<>& Y, std::vector<int> indices);

        void deleteTree(Node* node);

    public:
        DescisionTree();
        ~DescisionTree();

        void setFeatureIndices(const std::vector<int>& indices);
        void fit(const Matrix<>& X, const Matrix<>& y) override;
        Matrix<> predict(const Matrix<>& X) override;
};
