#pragma once
#include "../math/matrix.h"
#include "model-interface.h"
#include "descision-tree.h"
#include <vector>


class RandomForest : public FitPredictModel {
    private:
        std::vector<DescisionTree*> trees;

        IMPURITY impurity = IMPURITY::GINI;
        int n_estimators = 3;

    public:
        RandomForest();
        ~RandomForest();

        void fit(const Matrix<>& X, const Matrix<>& y) override;
        Matrix<> predict(const Matrix<>& X) override;
};
