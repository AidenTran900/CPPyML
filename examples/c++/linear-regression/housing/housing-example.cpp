#include "housing-example.h"
#include <fmt/format.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "ml_lib/models/linear-regression.h"
#include "ml_lib/core/loss.h"
#include "ml_lib/core/optimizer.h"
#include "ml_lib/core/regularizer.h"
#include "ml_lib/core/metrics.h"
#include "ml_lib/utils/csv_utils.h"

struct HousingData {
    std::vector<std::vector<double>> features;
    std::vector<double> prices;
    std::vector<std::string> feature_names;
};

inline double parseFurnishing(const std::string& status) {
    char first = status[0];
    if (first == 'f' || first == 'F') return 2.0;
    if (first == 's' || first == 'S') return 1.0;
    return 0.0;
}

HousingData loadHousingCSV(const std::string& filename) {
    using namespace ml_lib::utils;

    HousingData data;
    data.feature_names = {"area", "bedrooms", "bathrooms", "stories",
                          "mainroad", "guestroom", "basement", "hotwaterheating",
                          "airconditioning", "parking", "prefarea", "furnishingstatus"};

    std::vector<std::function<double(const std::string&)>> parsers;

    // Price (millions)
    parsers.push_back([](const std::string& s) {
        return std::stod(s) / 1000000.0;
    });

    // Number data: area, bedrooms, bathrooms, stories
    for (int i = 0; i < 4; i++) {
        parsers.push_back([](const std::string& s) {
            return std::stod(s);
        });
    }

    // Yes/No
    for (int i = 0; i < 7; i++) {
        parsers.push_back(CSVUtils::parseYesNo);
    }

    // Furnishing status
    parsers.push_back(parseFurnishing);

    try {
        auto all_data = CSVUtils::readWithParsers(filename, parsers, true);

        // Split into features and prices
        for (const auto& row : all_data) {
            if (!row.empty()) {
                data.prices.push_back(row[0]);
                data.features.push_back(std::vector<double>(row.begin() + 1, row.end()));
            }
        }

        fmt::print("Loaded {} housing samples from {}\n", data.features.size(), filename);
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

static void splitData(const HousingData& data,
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
    std::vector<std::vector<double>> train_prices;
    std::vector<std::vector<double>> test_prices;

    train_features.reserve(train_size);
    train_prices.reserve(train_size);
    test_features.reserve(total_samples - train_size);
    test_prices.reserve(total_samples - train_size);

    for (size_t i = 0; i < train_size; i++) {
        size_t idx = indices[i];
        train_features.push_back(data.features[idx]);
        train_prices.push_back({data.prices[idx]});
    }

    for (size_t i = train_size; i < total_samples; i++) {
        size_t idx = indices[i];
        test_features.push_back(data.features[idx]);
        test_prices.push_back({data.prices[idx]});
    }

    X_train = Matrix(std::move(train_features));
    y_train = Matrix(std::move(train_prices));
    X_test = Matrix(std::move(test_features));
    y_test = Matrix(std::move(test_prices));
}

int runHousingExample() {
    fmt::print("{:-<60}\n", "");
    fmt::print("Housing Price Prediction - Linear Regression\n");
    fmt::print("{:-<60}\n", "");

    HousingData data = loadHousingCSV("./examples/datasets/housing.csv");

    if (data.features.empty()) {
        fmt::print("Error: No data loaded!\n");
        return 1;
    }

    normalizeFeatures(data.features);
    fmt::print("Features normalized using z-score normalization\n\n");

    Matrix X_train, y_train, X_test, y_test;
    splitData(data, X_train, y_train, X_test, y_test, 0.2);

    fmt::print("{:-<60}\n", "");
    fmt::print("Dataset Split:\n");
    fmt::print("{:-<60}\n", "");
    fmt::print("    Training samples: {}\n", X_train.rows());
    fmt::print("    Test samples: {}\n", X_test.rows());
    fmt::print("    Features: {}\n", X_train.cols());
    fmt::print("    Feature names: area, bedrooms, bathrooms, stories,\n");
    fmt::print("        mainroad, guestroom, basement, hotwaterheating,\n");
    fmt::print("        airconditioning, parking, prefarea, furnishing\n\n");

    LinearRegression model(
        X_train.cols(),
        createLoss(LossType::ROOT_MEAN_SQUARED_ERROR),
        createOptimizer(OptimizerType::BATCH, 0.01),
        createRegularizer(RegularizerType::L2, 0.01)
    );

    fmt::print("{:-<60}\n", "");
    fmt::print("Training Progress:\n");
    fmt::print("{:-<60}\n", "");

    constexpr int epochs = 2000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix y_pred = model.forward(X_train);
        double loss = model.computeLoss(y_pred, y_train);
        model.backward(y_train);
        model.update();

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            fmt::print("Epoch {:4d}: Loss = {:.6f}, R² = {:.4f}\n",
                      epoch, loss, metrics::r2(y_train, y_pred));
        }
    }

    fmt::print("\n{:-<60}\n", "");
    fmt::print("Training Set Metrics:\n");
    fmt::print("{:-<60}\n", "");
    Matrix y_train_pred = model.forward(X_train);

    fmt::print("MSE:  {:.4f} (million²)\n", metrics::mse(y_train, y_train_pred));
    fmt::print("RMSE: {:.4f} million\n", metrics::rmse(y_train, y_train_pred));
    fmt::print("MAE:  {:.4f} million\n", metrics::mae(y_train, y_train_pred));
    fmt::print("R^2:   {:.4f}\n", metrics::r2(y_train, y_train_pred));

    fmt::print("\n{:-<60}\n", "");
    fmt::print("Test Set Metrics:\n");
    fmt::print("{:-<60}\n", "");
    Matrix y_test_pred = model.forward(X_test);

    fmt::print("MSE:  {:.4f} (million²)\n", metrics::mse(y_test, y_test_pred));
    fmt::print("RMSE: {:.4f} million\n", metrics::rmse(y_test, y_test_pred));
    fmt::print("MAE:  {:.4f} million\n", metrics::mae(y_test, y_test_pred));
    fmt::print("R^2:   {:.4f}\n", metrics::r2(y_test, y_test_pred));

    fmt::print("\n{:-<60}\n", "");
    fmt::print("Sample Predictions (Test Set - First 10):\n");
    fmt::print("{:-<60}\n", "");
    fmt::print("{:<15} {:<15} {:<15} {:<15}\n", "Actual (M)", "Predicted (M)", "Error (M)", "Error %");
    fmt::print("{:-<60}\n", "");

    const int display_count = std::min(10, y_test.rows());
    for (int i = 0; i < display_count; i++) {
        double actual = y_test(i, 0);
        double predicted = y_test_pred(i, 0);
        double error = actual - predicted;
        fmt::print("{:<15.3f} {:<15.3f} {:<15.3f} {:<15.2f}\n",
                   actual, predicted, error, (std::abs(error) / actual) * 100.0);
    }

    fmt::print("\n{:-<60}\n", "");
    fmt::print("Model Summary:\n");
    fmt::print("{:-<60}\n", "");
    fmt::print("Loss Function: Mean Squared Error (MSE)\n");
    fmt::print("Optimizer: Batch Gradient Descent (lr=0.01)\n");
    fmt::print("Regularization: L2 (lambda=0.01)\n");
    fmt::print("Training Epochs: {}\n", epochs);

    fmt::print("{:-<60}\n", "");
    return 0;
}
