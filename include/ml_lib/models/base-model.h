#pragma once
#include "../math/matrix.h"
#include "../core/loss.h"
#include "../core/optimizer.h"

class BaseModel {
    protected:
        LossFunction* loss_func;
        Optimizer* optimizer;

        int batch_size;
        int epochs;

        Matrix last_input;
        Matrix last_output;

    public:
        BaseModel(LossFunction* loss, Optimizer* opt)
            : loss_func(loss), optimizer(opt), batch_size(32), epochs(100) {}

        virtual Matrix forward(const Matrix& X) = 0;
        virtual void backward(const Matrix& y_true) = 0;
        virtual void update() = 0;

        double computeLoss(const Matrix& y_pred, const Matrix& y_true) {
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

        virtual ~BaseModel() {
            delete loss_func;
            delete optimizer;
        }
};
