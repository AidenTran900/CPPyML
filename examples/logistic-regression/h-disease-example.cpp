#include "h-disease-example.h"
#include <fmt/format.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "ml_lib/models/logistic-regression.h"
#include "ml_lib/core/loss.h"
#include "ml_lib/core/optimizer.h"
#include "ml_lib/core/regularizer.h"
#include "ml_lib/core/metrics.h"
#include "ml_lib/utils/csv_utils.h"

struct HeartDiseaseData {
    std::vector<std::vector<double>> features;
    std::vector<double> labels;
    std::vector<std::string> feature_names;
};

HeartDiseaseData loadFraminghamCSV(const std::string& filename) {
    using namespace ml_lib::utils;

    HeartDiseaseData data;
    data.feature_names = {
        "male", "age", "education", "currentSmoker", "cigsPerDay",
        "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
        "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    };

    try {
        bool success = CSVUtils::readFeatureTarget(
            filename,
            15,
            data.features,
            data.labels,
            true  // has header
        );

        if (success) {
            fmt::print("Loaded {} heart disease samples from {}\n",
                      data.features.size(), filename);
        } else {
            fmt::print("Error: Failed to load data from {}\n", filename);
        }
    } catch (const std::exception& e) {
        fmt::print("Error loading CSV: {}\n", e.what());
    }

    return data;
}

static void normalizeFeatures(std::vector<std::vector<double>>& features) {
    if (features.empty()) return;

    const size_t num_features = features[0].size();
    const size_t num_samples = features.size();
    const double inv_samples = 1.0 / num_samples;

    for (size_t f = 0; f < num_features; f++) {
        double sum = 0.0;
        double sum_sq = 0.0;

        for (size_t i = 0; i < num_samples; i++) {
            double val = features[i][f];
            sum += val;
            sum_sq += val * val;
        }

        double mean = sum * inv_samples;
        double variance = (sum_sq * inv_samples) - (mean * mean);
        double std_dev = std::sqrt(variance);

        if (std_dev > 1e-9) {
            double inv_std = 1.0 / std_dev;
            for (size_t i = 0; i < num_samples; i++) {
                features[i][f] = (features[i][f] - mean) * inv_std;
            }
        }
    }
}

static void splitData(const HeartDiseaseData& data,
               Matrix& X_train, Matrix& y_train,
               Matrix& X_test, Matrix& y_test,
               double test_ratio = 0.2) {

    const size_t total_samples = data.features.size();

    std::vector<size_t> indices(total_samples);
    for (size_t i = 0; i < total_samples; i++) {
        indices[i] = i;
    }

    std::mt19937 gen(42);
    std::shuffle(indices.begin(), indices.end(), gen);

    const size_t train_size = total_samples - static_cast<size_t>(total_samples * test_ratio);

    std::vector<std::vector<double>> train_features;
    std::vector<std::vector<double>> test_features;
    std::vector<std::vector<double>> train_labels;
    std::vector<std::vector<double>> test_labels;

    train_features.reserve(train_size);
    train_labels.reserve(train_size);
    test_features.reserve(total_samples - train_size);
    test_labels.reserve(total_samples - train_size);

    for (size_t i = 0; i < train_size; i++) {
        size_t idx = indices[i];
        train_features.push_back(data.features[idx]);
        train_labels.push_back({data.labels[idx]});
    }

    for (size_t i = train_size; i < total_samples; i++) {
        size_t idx = indices[i];
        test_features.push_back(data.features[idx]);
        test_labels.push_back({data.labels[idx]});
    }

    X_train = Matrix(std::move(train_features));
    y_train = Matrix(std::move(train_labels));
    X_test = Matrix(std::move(test_features));
    y_test = Matrix(std::move(test_labels));
}

int runHeartDiseaseExample() {
    fmt::print("{:-<70}\n", "");
    fmt::print("Heart Disease Prediction - Logistic Regression\n");
    fmt::print("Dataset: Framingham Heart Study - 10 Year CHD Risk\n");
    fmt::print("{:-<70}\n", "");

    HeartDiseaseData data = loadFraminghamCSV("./examples/datasets/heart-disease.csv");

    if (data.features.empty()) {
        fmt::print("Error: No data loaded!\n");
        return 1;
    }

    normalizeFeatures(data.features);
    fmt::print("Features normalized using z-score normalization\n\n");

    Matrix X_train, y_train, X_test, y_test;
    splitData(data, X_train, y_train, X_test, y_test, 0.2);

    fmt::print("{:-<70}\n", "");
    fmt::print("Dataset Split:\n");
    fmt::print("{:-<70}\n", "");
    fmt::print("    Training samples: {}\n", X_train.rows());
    fmt::print("    Test samples: {}\n", X_test.rows());
    fmt::print("    Features: {}\n", X_train.cols());
    fmt::print("    Feature names:\n");
    for (size_t i = 0; i < data.feature_names.size(); i++) {
        fmt::print("        {}: {}\n", i + 1, data.feature_names[i]);
    }
    fmt::print("\n");

    int train_positive = 0, test_positive = 0;
    for (int i = 0; i < y_train.rows(); i++) {
        if (y_train(i, 0) > 0.5) train_positive++;
    }
    for (int i = 0; i < y_test.rows(); i++) {
        if (y_test(i, 0) > 0.5) test_positive++;
    }

    fmt::print("Class distribution:\n");
    fmt::print("    Training: {} positive ({:.1f}%), {} negative ({:.1f}%)\n",
               train_positive, (100.0 * train_positive) / y_train.rows(),
               y_train.rows() - train_positive, (100.0 * (y_train.rows() - train_positive)) / y_train.rows());
    fmt::print("    Test: {} positive ({:.1f}%), {} negative ({:.1f}%)\n\n",
               test_positive, (100.0 * test_positive) / y_test.rows(),
               y_test.rows() - test_positive, (100.0 * (y_test.rows() - test_positive)) / y_test.rows());

    LogisticRegression model(
        X_train.cols(),
        createLoss(LossType::BCE),
        createOptimizer(OptimizerType::BATCH, 0.1),
        createRegularizer(RegularizerType::L2, 0.01)
    );

    fmt::print("{:-<70}\n", "");
    fmt::print("Training Progress:\n");
    fmt::print("{:-<70}\n", "");

    constexpr int epochs = 1000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix y_pred = model.forward(X_train);
        double loss = model.computeLoss(y_pred, y_train);
        model.backward(y_train);
        model.update();

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            // Calculate accuracy manually
            int correct = 0;
            for (int i = 0; i < y_pred.rows(); i++) {
                int predicted = y_pred(i, 0) > 0.5 ? 1 : 0;
                int actual = static_cast<int>(y_train(i, 0));
                if (predicted == actual) correct++;
            }
            double accuracy = static_cast<double>(correct) / y_pred.rows();

            fmt::print("Epoch {:4d}: Loss = {:.6f}, Accuracy = {:.4f}\n",
                      epoch, loss, accuracy);
        }
    }

    Matrix y_train_pred = model.forward(X_train);
    Matrix y_test_pred = model.forward(X_test);

    fmt::print("\n{:-<70}\n", "");
    fmt::print("Sample Predictions (Test Set - First 10):\n");
    fmt::print("{:-<70}\n", "");
    fmt::print("{:<10} {:<15} {:<15} {:<15}\n", "Actual", "Probability", "Predicted", "Correct");
    fmt::print("{:-<70}\n", "");

    const int display_count = std::min(10, y_test.rows());
    for (int i = 0; i < display_count; i++) {
        int actual = static_cast<int>(y_test(i, 0));
        double probability = y_test_pred(i, 0);
        int predicted = probability > 0.5 ? 1 : 0;
        bool correct = (actual == predicted);

        fmt::print("{:<10} {:<15.4f} {:<15} {:<15}\n",
                   actual == 1 ? "CHD" : "No CHD",
                   probability,
                   predicted == 1 ? "CHD" : "No CHD",
                   correct ? "Yes" : "No");
    }

    fmt::print("\n{:-<70}\n", "");
    fmt::print("Model Summary:\n");
    fmt::print("{:-<70}\n", "");
    fmt::print("Loss Function: Binary Cross-Entropy (BCE)\n");
    fmt::print("Optimizer: Batch Gradient Descent (lr=0.1)\n");
    fmt::print("Regularization: L2 (lambda=0.01)\n");
    fmt::print("Training Epochs: {}\n", epochs);

    fmt::print("{:-<70}\n", "");
    return 0;
}
