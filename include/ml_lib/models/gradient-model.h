#pragma once
#include "../math/matrix.h"
#include "../core/loss.h"
#include "../core/optimizer.h"
#include "../core/regularizer.h"
#include "model-interface.h"
#include <memory>

template<typename T = double>
class GradientModel : public GradientModelInterface<T> {
    protected:
        std::unique_ptr<LossFunction<T>> loss_func;
        std::unique_ptr<Optimizer<T>> optimizer;
        std::unique_ptr<Regularizer<T>> regularizer;

        int batch_size;
        int epochs;

        Matrix<T> last_input;
        Matrix<T> last_output;

    public:
        GradientModel(std::unique_ptr<LossFunction<T>> loss,
                      std::unique_ptr<Optimizer<T>> opt,
                      std::unique_ptr<Regularizer<T>> reg)
            : loss_func(std::move(loss)),
              optimizer(std::move(opt)),
              regularizer(std::move(reg)),
              batch_size(32),
              epochs(100) {}

        virtual Matrix<T> forward(const Matrix<T>& X) = 0;
        virtual void backward(const Matrix<T>& y_true) = 0;
        virtual void update() = 0;

        double computeLoss(const Matrix<T>& y_pred, const Matrix<T>& y_true) {
            return loss_func->compute(y_pred, y_true);
        }

        void setLearningRate(double lr) {
            if (optimizer) {
                optimizer->setLearningRate(lr);
            }
        }

        void setEpochs(int ep) {
            epochs = ep;
        }

        void setBatchSize(int b) {
            batch_size = b;
        }

        // Move constructor and assignment for pybind11 compatibility
        GradientModel(GradientModel&&) = default;
        GradientModel& operator=(GradientModel&&) = default;

        // Disable copy (unique_ptr members are not copyable)
        GradientModel(const GradientModel&) = delete;
        GradientModel& operator=(const GradientModel&) = delete;

        virtual ~GradientModel() = default;
};
