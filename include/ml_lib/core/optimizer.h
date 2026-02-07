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

template<typename T = double>
class Optimizer {
    protected:
        double learning_rate;

    public:
        Optimizer(double lr) : learning_rate(lr) {}

        virtual void step(Matrix<T>& param, const Matrix<T>& grad) = 0;

        void setLearningRate(double lr) { learning_rate = lr; }
        double getLearningRate() const { return learning_rate; }

        virtual ~Optimizer() {}
};

template<typename T = double>
class StochasticOptimizer : public Optimizer<T> {
    public:
        StochasticOptimizer(double lr) : Optimizer<T>(lr) {}

        void step(Matrix<T>& param, const Matrix<T>& grad) override;
};

template<typename T = double>
class MiniBatchOptimizer : public Optimizer<T> {
    private:
        int batch_size;
        int current_step;
        std::unordered_map<Matrix<T>*, Matrix<T>> accumulated_grads;

    public:
        MiniBatchOptimizer(double lr, int batch_size = 32)
            : Optimizer<T>(lr), batch_size(batch_size), current_step(0) {}

        void step(Matrix<T>& param, const Matrix<T>& grad) override;
        void setBatchSize(int size) { batch_size = size; }
        int getBatchSize() const { return batch_size; }
};

template<typename T = double>
class AdamOptimizer : public Optimizer<T> {
   private:
       double beta1;
       double beta2;
       double epsilon;
       std::unordered_map<const Matrix<T>*, Matrix<T>> first_moment;
       std::unordered_map<const Matrix<T>*, Matrix<T>> second_moment;
       std::unordered_map<const Matrix<T>*, int> timesteps;

   public:
       AdamOptimizer(double lr, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
           : Optimizer<T>(lr), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

       void step(Matrix<T>& param, const Matrix<T>& grad) override;
};

template<typename T = double>
class MomentumOptimizer : public Optimizer<T> {
    private:
        double momentum;
        std::unordered_map<const Matrix<T>*, Matrix<T>> velocity;

    public:
        MomentumOptimizer(double lr, double momentum = 0.9)
            : Optimizer<T>(lr), momentum(momentum) {}

        void step(Matrix<T>& param, const Matrix<T>& grad) override;
};

template<typename T = double>
class AdaGradOptimizer : public Optimizer<T> {
    private:
        double epsilon;
        std::unordered_map<const Matrix<T>*, Matrix<T>> accumulated_squared_grads;

    public:
        AdaGradOptimizer(double lr, double epsilon = 1e-8)
            : Optimizer<T>(lr), epsilon(epsilon) {}

        void step(Matrix<T>& param, const Matrix<T>& grad) override;
};

template<typename T = double>
class RMSpropOptimizer : public Optimizer<T> {
    private:
        double decay_rate;
        double epsilon;
        std::unordered_map<const Matrix<T>*, Matrix<T>> mean_squared_grads;

    public:
        RMSpropOptimizer(double lr, double decay_rate = 0.9, double epsilon = 1e-8)
            : Optimizer<T>(lr), decay_rate(decay_rate), epsilon(epsilon) {}

        void step(Matrix<T>& param, const Matrix<T>& grad) override;
};

template<typename T = double>
std::unique_ptr<Optimizer<T>> createOptimizer(OptimizerType type, double lr);
