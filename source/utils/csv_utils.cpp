#include "ml_lib/utils/csv_utils.h"
#include <stdexcept>
#include <cctype>
#include <algorithm>

namespace ml_lib {
namespace utils {

std::vector<std::vector<double>> CSVUtils::readNumeric(
    const std::string& filename,
    bool has_header) {

    std::vector<std::vector<double>> result;

    csv::CSVFormat format;
    format.header_row(has_header ? 0 : -1);

    csv::CSVReader reader(filename, format);

    for (csv::CSVRow& row : reader) {
        std::vector<double> row_data;
        row_data.reserve(row.size());

        for (csv::CSVField& field : row) {
            row_data.push_back(field.get<double>());
        }

        result.push_back(std::move(row_data));
    }

    return result;
}

std::vector<std::vector<double>> CSVUtils::readWithParsers(
    const std::string& filename,
    const std::vector<std::function<double(const std::string&)>>& column_parsers,
    bool has_header) {

    std::vector<std::vector<double>> result;

    csv::CSVFormat format;
    format.header_row(has_header ? 0 : -1);

    csv::CSVReader reader(filename, format);

    for (csv::CSVRow& row : reader) {
        std::vector<double> row_data;
        row_data.reserve(row.size());

        size_t col_idx = 0;
        for (csv::CSVField& field : row) {
            if (col_idx < column_parsers.size() && column_parsers[col_idx]) {
                std::string field_str = field.get<std::string>();
                row_data.push_back(column_parsers[col_idx](field_str));
            } else {
                row_data.push_back(field.get<double>());
            }
            col_idx++;
        }

        result.push_back(std::move(row_data));
    }

    return result;
}

std::vector<std::vector<double>> CSVUtils::readColumns(
    const std::string& filename,
    const std::vector<size_t>& column_indices,
    bool has_header) {

    std::vector<std::vector<double>> result;

    csv::CSVFormat format;
    format.header_row(has_header ? 0 : -1);

    csv::CSVReader reader(filename, format);

    for (csv::CSVRow& row : reader) {
        std::vector<double> row_data;
        row_data.reserve(column_indices.size());

        for (size_t idx : column_indices) {
            if (idx < row.size()) {
                row_data.push_back(row[idx].get<double>());
            }
        }

        if (!row_data.empty()) {
            result.push_back(std::move(row_data));
        }
    }

    return result;
}

bool CSVUtils::readFeatureTarget(
    const std::string& filename,
    size_t target_column,
    std::vector<std::vector<double>>& features,
    std::vector<double>& target,
    bool has_header) {

    features.clear();
    target.clear();

    csv::CSVFormat format;
    format.header_row(has_header ? 0 : -1);

    try {
        csv::CSVReader reader(filename, format);

        for (csv::CSVRow& row : reader) {
            std::vector<double> feature_row;
            double target_value = 0.0;
            bool has_missing = false;

            size_t col_idx = 0;
            for (csv::CSVField& field : row) {
                std::string field_str = field.get<std::string>();

                // Check for missing values (NA, empty, etc.)
                if (field_str.empty() || field_str == "NA" || field_str == "N/A") {
                    has_missing = true;
                    break;
                }

                double value;
                try {
                    value = field.get<double>();
                } catch (...) {
                    has_missing = true;
                    break;
                }

                if (col_idx == target_column) {
                    target_value = value;
                } else {
                    feature_row.push_back(value);
                }
                col_idx++;
            }

            // Only add row if it has no missing values
            if (!has_missing && !feature_row.empty()) {
                features.push_back(std::move(feature_row));
                target.push_back(target_value);
            }
        }

        return !features.empty();

    } catch (const std::exception&) {
        return false;
    }
}

// Common parsing utilities
double CSVUtils::parseBinary(const std::string& value) {
    if (value.empty()) return 0.0;
    char first = static_cast<char>(std::tolower(static_cast<unsigned char>(value[0])));
    return (first == 'y' || first == '1' || first == 't') ? 1.0 : 0.0;
}

double CSVUtils::parseYesNo(const std::string& value) {
    if (value.empty()) return 0.0;
    char first = static_cast<char>(std::tolower(static_cast<unsigned char>(value[0])));
    return (first == 'y') ? 1.0 : 0.0;
}

double CSVUtils::parseTrueFalse(const std::string& value) {
    if (value.empty()) return 0.0;
    char first = static_cast<char>(std::tolower(static_cast<unsigned char>(value[0])));
    return (first == 't') ? 1.0 : 0.0;
}

} // namespace utils
} // namespace ml_lib
