#include "ml_lib/core/token-sampler.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

// TemperatureProcessor

template<typename T>
TemperatureProcessor<T>::TemperatureProcessor(double temperature)
    : temperature(temperature)
{
    if (temperature <= 0.0)
        throw std::invalid_argument("Temperature must be > 0");
}

template<typename T>
void TemperatureProcessor<T>::process(Matrix<T>& logits) const
{
    int cols = logits.cols();

    T max_val = logits(0, 0);
    for (int j = 1; j < cols; j++) {
        if (logits(0, j) > max_val) max_val = logits(0, j);
    }

    double sum = 0.0;
    for (int j = 0; j < cols; j++) {
        double scaled = (static_cast<double>(logits(0, j)) - static_cast<double>(max_val)) / temperature;
        logits(0, j) = static_cast<T>(std::exp(scaled));
        sum += static_cast<double>(logits(0, j));
    }

    for (int j = 0; j < cols; j++) {
        logits(0, j) = static_cast<T>(static_cast<double>(logits(0, j)) / sum);
    }
}

// Top K Processor

template<typename T>
TopKProcessor<T>::TopKProcessor(int k) : k(k)
{
    if (k < 1)
        throw std::invalid_argument("k must be >= 1");
}

template<typename T>
void TopKProcessor<T>::process(Matrix<T>& logits) const
{
    int cols = logits.cols();
    if (k >= cols) return;

    // find largest value as threshold
    std::vector<T> vals(cols);
    for (int j = 0; j < cols; j++) vals[j] = logits(0, j);
    std::nth_element(vals.begin(), vals.begin() + k - 1, vals.end(), std::greater<T>());
    T threshold = vals[k - 1];

    // renormalize and zero out rest
    double sum = 0.0;
    for (int j = 0; j < cols; j++) {
        if (logits(0, j) < threshold) {
            logits(0, j) = static_cast<T>(0);
        } else {
            sum += static_cast<double>(logits(0, j));
        }
    }

    if (sum > 0.0) {
        for (int j = 0; j < cols; j++) {
            logits(0, j) = static_cast<T>(static_cast<double>(logits(0, j)) / sum);
        }
    }
}

// Top P Processor

template<typename T>
TopPProcessor<T>::TopPProcessor(double p) : p(p)
{
    if (p <= 0.0 || p > 1.0)
        throw std::invalid_argument("p must be in (0, 1]");
}

template<typename T>
void TopPProcessor<T>::process(Matrix<T>& logits) const
{
    int cols = logits.cols();

    // index to probability pairs descending
    std::vector<std::pair<T, int>> sorted(cols);
    for (int j = 0; j < cols; j++) {
        sorted[j] = {logits(0, j), j};
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // accumulate until we reach p, zero out the rest
    double cumulative = 0.0;
    std::vector<bool> keep(cols, false);
    for (auto& [prob, idx] : sorted) {
        keep[idx] = true;
        cumulative += static_cast<double>(prob);
        if (cumulative >= p) break;
    }

    double sum = 0.0;
    for (int j = 0; j < cols; j++) {
        if (!keep[j]) {
            logits(0, j) = static_cast<T>(0);
        } else {
            sum += static_cast<double>(logits(0, j));
        }
    }

    if (sum > 0.0) {
        for (int j = 0; j < cols; j++) {
            logits(0, j) = static_cast<T>(static_cast<double>(logits(0, j)) / sum);
        }
    }
}

// Greedy Selector

template<typename T>
int GreedySelector<T>::select(const Matrix<T>& probs) const
{
    int best = 0;
    T best_val = probs(0, 0);
    for (int j = 1; j < probs.cols(); j++) {
        if (probs(0, j) > best_val) {
            best_val = probs(0, j);
            best = j;
        }
    }
    return best;
}

// Categorical Selector

template<typename T>
CategoricalSelector<T>::CategoricalSelector()
    : rng(std::random_device{}())
{}

template<typename T>
CategoricalSelector<T>::CategoricalSelector(unsigned int seed)
    : rng(seed)
{}

template<typename T>
int CategoricalSelector<T>::select(const Matrix<T>& probs) const
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cumulative = 0.0;

    for (int j = 0; j < probs.cols(); j++) {
        cumulative += static_cast<double>(probs(0, j));
        if (r <= cumulative) return j;
    }

    return probs.cols() - 1;
}


// Token Sampler
template<typename T>
TokenSampler<T>::TokenSampler()
    : selector(std::make_unique<GreedySelector<T>>())
{}

template<typename T>
void TokenSampler<T>::addProcessor(std::unique_ptr<LogitProcessor<T>> proc)
{
    processors.push_back(std::move(proc));
}

template<typename T>
void TokenSampler<T>::setSelector(std::unique_ptr<TokenSelector<T>> sel)
{
    selector = std::move(sel);
}

template<typename T>
int TokenSampler<T>::sample(Matrix<T> logits) const
{
    for (auto& proc : processors) {
        proc->process(logits);
    }
    return selector->select(logits);
}


template class LogitProcessor<float>;
template class LogitProcessor<double>;

template class TemperatureProcessor<float>;
template class TemperatureProcessor<double>;

template class TopKProcessor<float>;
template class TopKProcessor<double>;

template class TopPProcessor<float>;
template class TopPProcessor<double>;

template class TokenSelector<float>;
template class TokenSelector<double>;

template class GreedySelector<float>;
template class GreedySelector<double>;

template class CategoricalSelector<float>;
template class CategoricalSelector<double>;

template class TokenSampler<float>;
template class TokenSampler<double>;
