#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ml_lib/math/matrix.h"
#include "ml_lib/core/loss.h"
#include "ml_lib/core/optimizer.h"
#include "ml_lib/core/regularizer.h"
#include "ml_lib/core/metrics.h"
#include "ml_lib/models/linear-regression.h"
#include "ml_lib/models/logistic-regression.h"
#include "ml_lib/models/support-vector-machine.h"
#include "ml_lib/models/k-means-clustering.h"
#include "ml_lib/models/k-nearest-neighbors.h"
#include "ml_lib/models/descision-tree.h"
#include "ml_lib/models/random-forest.h"

namespace py = pybind11;

// Helper function to convert numpy array to Matrix
Matrix numpy_to_matrix(py::array_t<double> input) {
    py::buffer_info buf = input.request();

    if (buf.ndim == 1) {
        Matrix mat(buf.shape[0], 1);
        double* ptr = static_cast<double*>(buf.ptr);
        for (size_t i = 0; i < buf.shape[0]; i++) {
            mat(i, 0) = ptr[i];
        }
        return mat;
    } else if (buf.ndim == 2) {
        Matrix mat(buf.shape[0], buf.shape[1]);
        double* ptr = static_cast<double*>(buf.ptr);
        for (size_t i = 0; i < buf.shape[0]; i++) {
            for (size_t j = 0; j < buf.shape[1]; j++) {
                mat(i, j) = ptr[i * buf.shape[1] + j];
            }
        }
        return mat;
    } else {
        throw std::runtime_error("Input array must be 1D or 2D");
    }
}

// Helper function to convert Matrix to numpy array
py::array_t<double> matrix_to_numpy(const Matrix& mat) {
    auto result = py::array_t<double>({mat.rows(), mat.cols()});
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            ptr[i * mat.cols() + j] = mat(i, j);
        }
    }
    return result;
}

PYBIND11_MODULE(ml_lib, m) {
    m.doc() = "Python bindings for C++ ML library";

    // Matrix class
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def(py::init([](py::array_t<double> arr) {
            return numpy_to_matrix(arr);
        }), py::arg("array"), "Create Matrix from numpy array")
        .def("rows", &Matrix::rows, "Get number of rows")
        .def("cols", &Matrix::cols, "Get number of columns")
        .def("__getitem__", [](const Matrix& mat, std::pair<int, int> idx) {
            return mat(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix& mat, std::pair<int, int> idx, double value) {
            mat(idx.first, idx.second) = value;
        })
        .def("to_numpy", [](const Matrix& mat) {
            return matrix_to_numpy(mat);
        }, "Convert Matrix to numpy array")
        .def("__repr__", [](const Matrix& mat) {
            return "<Matrix " + std::to_string(mat.rows()) + "x" + std::to_string(mat.cols()) + ">";
        });

    // Enums
    py::enum_<LossType>(m, "LossType")
        .value("MEAN_ABSOLUTE_ERROR", LossType::MEAN_ABSOLUTE_ERROR, "Mean Absolute Error")
        .value("MEAN_SQUARED_ERROR", LossType::MEAN_SQUARED_ERROR, "Mean Squared Error")
        .value("ROOT_MEAN_SQUARED_ERROR", LossType::ROOT_MEAN_SQUARED_ERROR, "Root Mean Squared Error")
        .value("BINARY_CROSS_ENTROPY", LossType::BINARY_CROSS_ENTROPY, "Binary Cross Entropy")
        .export_values();

    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("BATCH", OptimizerType::BATCH, "Batch gradient descent")
        .value("STOCHASTIC", OptimizerType::STOCHASTIC, "Stochastic gradient descent")
        .value("MINI_BATCH", OptimizerType::MINI_BATCH, "Mini-batch gradient descent")
        .export_values();

    py::enum_<RegularizerType>(m, "RegularizerType")
        .value("NONE", RegularizerType::None, "No regularization")
        .value("L1", RegularizerType::L1, "L1 (Lasso) regularization")
        .value("L2", RegularizerType::L2, "L2 (Ridge) regularization")
        .export_values();

    py::enum_<KERNEL>(m, "Kernel")
        .value("LINEAR", KERNEL::LINEAR, "Linear kernel")
        .value("POLYNOMIAL", KERNEL::POLYNOMIAL, "Polynomial kernel")
        .value("RBF", KERNEL::RBF, "Radial basis function kernel")
        .value("SIGMOID", KERNEL::SIGMOID, "Sigmoid kernel")
        .export_values();

    py::enum_<CLUSTER_INIT>(m, "ClusterInit")
        .value("RANDOM", CLUSTER_INIT::RANDOM, "Random initialization")
        .value("SMART", CLUSTER_INIT::SMART, "Smart initialization (K-means++)")
        .export_values();

    py::enum_<IMPURITY>(m, "Impurity")
        .value("GINI", IMPURITY::GINI, "Gini impurity")
        .value("ENTROPY", IMPURITY::ENTROPY, "Entropy impurity")
        .export_values();

    // Loss Functions
    py::class_<LossFunction>(m, "LossFunction");

    py::class_<MeanSquaredErrorLoss, LossFunction>(m, "MeanSquaredErrorLoss")
        .def(py::init<>(), "Mean Squared Error loss");

    py::class_<MeanAbsoluteErrorLoss, LossFunction>(m, "MeanAbsoluteErrorLoss")
        .def(py::init<>(), "Mean Absolute Error loss");

    py::class_<RootMeanSquaredErrorLoss, LossFunction>(m, "RootMeanSquaredErrorLoss")
        .def(py::init<>(), "Root Mean Squared Error loss");

    py::class_<BinaryCrossEntropyLoss, LossFunction>(m, "BinaryCrossEntropyLoss")
        .def(py::init<>(), "Binary Cross Entropy loss");

    // Optimizers
    py::class_<Optimizer>(m, "Optimizer")
        .def("set_learning_rate", &Optimizer::setLearningRate, py::arg("lr"))
        .def("get_learning_rate", &Optimizer::getLearningRate);

    py::class_<BatchOptimizer, Optimizer>(m, "BatchOptimizer")
        .def(py::init<double>(), py::arg("learning_rate") = 0.01,
             "Batch Gradient Descent optimizer");

    py::class_<StochasticOptimizer, Optimizer>(m, "StochasticOptimizer")
        .def(py::init<double>(), py::arg("learning_rate") = 0.01,
             "Stochastic Gradient Descent optimizer");

    py::class_<MiniBatchOptimizer, Optimizer>(m, "MiniBatchOptimizer")
        .def(py::init<double>(), py::arg("learning_rate") = 0.01,
             "Mini-batch Gradient Descent optimizer");

    // Regularizers
    py::class_<Regularizer>(m, "Regularizer");

    py::class_<L1Regularizer, Regularizer>(m, "L1Regularizer")
        .def(py::init<double>(), py::arg("lambda") = 0.01,
             "L1 (Lasso) regularization");

    py::class_<L2Regularizer, Regularizer>(m, "L2Regularizer")
        .def(py::init<double>(), py::arg("lambda") = 0.01,
             "L2 (Ridge) regularization");

    // Base Model Interface
    py::class_<GradientModelInterface>(m, "GradientModelInterface");

    py::class_<FitPredictModel>(m, "FitPredictModel")
        .def("fit", [](FitPredictModel& model, py::array_t<double> X, py::array_t<double> y) {
            Matrix X_mat = numpy_to_matrix(X);
            Matrix y_mat = numpy_to_matrix(y);
            model.fit(X_mat, y_mat);
        }, py::arg("X"), py::arg("y"), "Fit the model to training data")
        .def("predict", [](FitPredictModel& model, py::array_t<double> X) {
            Matrix X_mat = numpy_to_matrix(X);
            Matrix result = model.predict(X_mat);
            return matrix_to_numpy(result);
        }, py::arg("X"), "Make predictions on input data");

    // K-Means Clustering
    py::class_<KMeansClustering, FitPredictModel>(m, "KMeansClustering")
        .def(py::init<int, int, int>(),
             py::arg("k") = 3,
             py::arg("threshold") = 3,
             py::arg("max_iter") = 100,
             "K-Means clustering algorithm\n\n"
             "Parameters:\n"
             "  k: Number of clusters (default=3)\n"
             "  threshold: Convergence threshold (default=3)\n"
             "  max_iter: Maximum iterations (default=100)");

    // K-Nearest Neighbors
    py::class_<KNearestNeighbors, FitPredictModel>(m, "KNearestNeighbors")
        .def(py::init<int>(),
             py::arg("k") = 3,
             "K-Nearest Neighbors classifier\n\n"
             "Parameters:\n"
             "  k: Number of neighbors to consider (default=3)");

    // Decision Tree
    py::class_<DescisionTree, FitPredictModel>(m, "DecisionTree")
        .def(py::init<>(),
             "Decision Tree classifier");

    // Random Forest
    py::class_<RandomForest, FitPredictModel>(m, "RandomForest")
        .def(py::init<>(),
             "Random Forest classifier");

    // Linear Regression
    py::class_<LinearRegression, GradientModelInterface>(m, "LinearRegression")
        .def(py::init([](int input_dim, LossType loss_type, OptimizerType opt_type,
                         double learning_rate, RegularizerType reg_type, double lambda_) {
            return LinearRegression(
                input_dim,
                createLoss(loss_type),
                createOptimizer(opt_type, learning_rate),
                createRegularizer(reg_type, lambda_)
            );
        }),
             py::arg("input_dim"),
             py::arg("loss") = LossType::MEAN_SQUARED_ERROR,
             py::arg("optimizer") = OptimizerType::BATCH,
             py::arg("learning_rate") = 0.01,
             py::arg("regularizer") = RegularizerType::L2,
             py::arg("lambda_") = 0.01,
             "Linear Regression model\n\n"
             "Parameters:\n"
             "  input_dim: Number of input features\n"
             "  loss: Loss type (LossType.MEAN_SQUARED_ERROR, LossType.MEAN_ABSOLUTE_ERROR, etc.)\n"
             "  optimizer: Optimizer type (OptimizerType.BATCH, etc.)\n"
             "  learning_rate: Learning rate for optimizer (default=0.01)\n"
             "  regularizer: Regularizer type (RegularizerType.L2, etc.)\n"
             "  lambda_: Regularization strength (default=0.01)")
        .def("forward", [](LinearRegression& model, py::array_t<double> X) {
            Matrix X_mat = numpy_to_matrix(X);
            Matrix result = model.forward(X_mat);
            return matrix_to_numpy(result);
        }, py::arg("X"), "Forward pass");

    // Logistic Regression
    py::class_<LogisticRegression, LinearRegression>(m, "LogisticRegression")
        .def(py::init([](int input_dim, LossType loss_type, OptimizerType opt_type,
                         double learning_rate, RegularizerType reg_type, double lambda_) {
            return LogisticRegression(
                input_dim,
                createLoss(loss_type),
                createOptimizer(opt_type, learning_rate),
                createRegularizer(reg_type, lambda_)
            );
        }),
             py::arg("input_dim"),
             py::arg("loss") = LossType::BINARY_CROSS_ENTROPY,
             py::arg("optimizer") = OptimizerType::BATCH,
             py::arg("learning_rate") = 0.01,
             py::arg("regularizer") = RegularizerType::L2,
             py::arg("lambda_") = 0.01,
             "Logistic Regression model for binary classification\n\n"
             "Parameters:\n"
             "  input_dim: Number of input features\n"
             "  loss: Loss type (LossType.BINARY_CROSS_ENTROPY recommended)\n"
             "  optimizer: Optimizer type (OptimizerType.BATCH, etc.)\n"
             "  learning_rate: Learning rate for optimizer (default=0.01)\n"
             "  regularizer: Regularizer type (RegularizerType.L2, etc.)\n"
             "  lambda_: Regularization strength (default=0.01)")
        .def("predict", [](LogisticRegression& model, py::array_t<double> X) {
            Matrix X_mat = numpy_to_matrix(X);
            Matrix result = model.predict(X_mat);
            return matrix_to_numpy(result);
        }, py::arg("X"), "Predict class labels (0 or 1)")
        .def("predict_proba", [](LogisticRegression& model, py::array_t<double> X) {
            Matrix X_mat = numpy_to_matrix(X);
            Matrix result = model.forward(X_mat);
            return matrix_to_numpy(result);
        }, py::arg("X"), "Predict class probabilities");

    // Support Vector Machine
    py::class_<SupportVectorMachine, FitPredictModel>(m, "SupportVectorMachine")
        .def(py::init<double, double, KERNEL, int, double, int, double>(),
             py::arg("C") = 1.0,
             py::arg("gamma") = 0.1,
             py::arg_v("kernel", KERNEL::LINEAR, "Kernel.LINEAR"),
             py::arg("degree") = 3,
             py::arg("tolerance") = 1e-3,
             py::arg("max_iter") = 1000,
             py::arg("coef0") = 0.0,
             "Support Vector Machine classifier\n\n"
             "Parameters:\n"
             "  C: Regularization parameter (default=1.0)\n"
             "  gamma: Kernel coefficient (default=0.1)\n"
             "  kernel: Kernel type (LINEAR, POLYNOMIAL, RBF, SIGMOID)\n"
             "  degree: Degree for polynomial kernel (default=3)\n"
             "  tolerance: Tolerance for stopping criterion (default=1e-3)\n"
             "  max_iter: Maximum iterations (default=1000)\n"
             "  coef0: Independent term in kernel function (default=0.0)");

    // Metrics module
    py::module_ metrics_module = m.def_submodule("metrics", "Evaluation metrics");

    metrics_module.def("accuracy", [](py::array_t<double> y_true, py::array_t<double> y_pred) -> double {
        Matrix y_true_mat = numpy_to_matrix(y_true);
        Matrix y_pred_mat = numpy_to_matrix(y_pred);
        Matrix confusion = metrics::confusionMatrix(y_true_mat, y_pred_mat);
        return metrics::accuracy(confusion);
    }, py::arg("y_true"), py::arg("y_pred"), "Calculate accuracy score");

    metrics_module.def("precision", [](py::array_t<double> y_true, py::array_t<double> y_pred) -> double {
        Matrix y_true_mat = numpy_to_matrix(y_true);
        Matrix y_pred_mat = numpy_to_matrix(y_pred);
        Matrix confusion = metrics::confusionMatrix(y_true_mat, y_pred_mat);
        return metrics::precision(confusion);
    }, py::arg("y_true"), py::arg("y_pred"), "Calculate precision score");

    metrics_module.def("recall", [](py::array_t<double> y_true, py::array_t<double> y_pred) -> double {
        Matrix y_true_mat = numpy_to_matrix(y_true);
        Matrix y_pred_mat = numpy_to_matrix(y_pred);
        Matrix confusion = metrics::confusionMatrix(y_true_mat, y_pred_mat);
        return metrics::recall(confusion);
    }, py::arg("y_true"), py::arg("y_pred"), "Calculate recall score");

    metrics_module.def("f1_score", [](py::array_t<double> y_true, py::array_t<double> y_pred) -> double {
        Matrix y_true_mat = numpy_to_matrix(y_true);
        Matrix y_pred_mat = numpy_to_matrix(y_pred);
        Matrix confusion = metrics::confusionMatrix(y_true_mat, y_pred_mat);
        return metrics::f1Score(confusion);
    }, py::arg("y_true"), py::arg("y_pred"), "Calculate F1 score");

    metrics_module.def("mse", [](py::array_t<double> y_true, py::array_t<double> y_pred) -> double {
        Matrix y_true_mat = numpy_to_matrix(y_true);
        Matrix y_pred_mat = numpy_to_matrix(y_pred);
        return metrics::mse(y_true_mat, y_pred_mat);
    }, py::arg("y_true"), py::arg("y_pred"), "Calculate mean squared error");

    metrics_module.def("mae", [](py::array_t<double> y_true, py::array_t<double> y_pred) -> double {
        Matrix y_true_mat = numpy_to_matrix(y_true);
        Matrix y_pred_mat = numpy_to_matrix(y_pred);
        return metrics::mae(y_true_mat, y_pred_mat);
    }, py::arg("y_true"), py::arg("y_pred"), "Calculate mean absolute error");

    metrics_module.def("rmse", [](py::array_t<double> y_true, py::array_t<double> y_pred) -> double {
        Matrix y_true_mat = numpy_to_matrix(y_true);
        Matrix y_pred_mat = numpy_to_matrix(y_pred);
        return metrics::rmse(y_true_mat, y_pred_mat);
    }, py::arg("y_true"), py::arg("y_pred"), "Calculate root mean squared error");

    metrics_module.def("r2_score", [](py::array_t<double> y_true, py::array_t<double> y_pred) -> double {
        Matrix y_true_mat = numpy_to_matrix(y_true);
        Matrix y_pred_mat = numpy_to_matrix(y_pred);
        return metrics::r2(y_true_mat, y_pred_mat);
    }, py::arg("y_true"), py::arg("y_pred"), "Calculate R² score");
}
