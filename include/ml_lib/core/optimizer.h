#pragma once
#include "../math/matrix.h"
#include <memory>
#include <unordered_map>

enum OptimizerType {
    STOCHASTIC,
        // Update parameters after 1 sample
    MINI_BATCH,
        // Update parameters after small batches
    MOMENTUM,
        // SGD with momentum
    ADAGRAD,
        // Adaptive gradient algorithm
    RMSPROP,
        // Root mean square propagation
    ADAM
        // Adaptive moment estimation
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

class StochasticOptimizer : public Optimizer {
    public:
        StochasticOptimizer(double lr) : Optimizer(lr) {}

        void step(Matrix& param, const Matrix& grad) override;
};

class MiniBatchOptimizer : public Optimizer {
    private:
        int batch_size;
        int current_step;
        std::unordered_map<Matrix*, Matrix> accumulated_grads;

    public:
        MiniBatchOptimizer(double lr, int batch_size = 32)
            : Optimizer(lr), batch_size(batch_size), current_step(0) {}

        void step(Matrix& param, const Matrix& grad) override;
        void setBatchSize(int size) { batch_size = size; }
        int getBatchSize() const { return batch_size; }
};

class AdamOptimizer : public Optimizer {
   private:
       double beta1;
       double beta2;
       double epsilon;
       std::unordered_map<const Matrix*, Matrix> first_moment;
       std::unordered_map<const Matrix*, Matrix> second_moment;
       std::unordered_map<const Matrix*, int> timesteps;

   public:
       AdamOptimizer(double lr, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
           : Optimizer(lr), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

       void step(Matrix& param, const Matrix& grad) override;
};

class MomentumOptimizer : public Optimizer {
    private:
        double momentum;
        std::unordered_map<const Matrix*, Matrix> velocity;

    public:
        MomentumOptimizer(double lr, double momentum = 0.9)
            : Optimizer(lr), momentum(momentum) {}

        void step(Matrix& param, const Matrix& grad) override;
};

class AdaGradOptimizer : public Optimizer {
    private:
        double epsilon;
        std::unordered_map<const Matrix*, Matrix> accumulated_squared_grads;

    public:
        AdaGradOptimizer(double lr, double epsilon = 1e-8)
            : Optimizer(lr), epsilon(epsilon) {}

        void step(Matrix& param, const Matrix& grad) override;
};

class RMSpropOptimizer : public Optimizer {
    private:
        double decay_rate;
        double epsilon;
        std::unordered_map<const Matrix*, Matrix> mean_squared_grads;

    public:
        RMSpropOptimizer(double lr, double decay_rate = 0.9, double epsilon = 1e-8)
            : Optimizer(lr), decay_rate(decay_rate), epsilon(epsilon) {}

        void step(Matrix& param, const Matrix& grad) override;
};

std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, double lr);