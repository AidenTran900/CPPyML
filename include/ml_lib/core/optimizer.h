#pragma once
#include "../math/matrix.h"
#include <memory>

enum OptimizerType {
    BATCH,
        // Using full dataset
    STOCHASTIC,
        // Update parameters after 1 sample
    MINI_BATCH
        // Update parameters after small batches
};

class Optimizer {
    protected:
        double learning_rate;

    public:
        Optimizer(double lr) : learning_rate(lr) {}

        virtual void step(Matrix& param, const Matrix& grad) = 0;

        void setLearningRate(double lr) { learning_rate = lr; }
        double getLearningRate() const { return learning_rate; }

        virtual ~Optimizer() {}
};

class BatchOptimizer : public Optimizer {
    public:
        BatchOptimizer(double lr) : Optimizer(lr) {}

        void step(Matrix& param, const Matrix& grad) override;
};

class StochasticOptimizer : public Optimizer {
    public:
        StochasticOptimizer(double lr) : Optimizer(lr) {}

        void step(Matrix& param, const Matrix& grad) override;
};

class MiniBatchOptimizer : public Optimizer {
    public:
        MiniBatchOptimizer(double lr) : Optimizer(lr) {}

        void step(Matrix& param, const Matrix& grad) override;
};

std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, double lr);