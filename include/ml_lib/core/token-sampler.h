#pragma once
#include "../math/matrix.h"
#include <memory>
#include <vector>
#include <random>

template<typename T = double>
class LogitProcessor {
    public:
        virtual ~LogitProcessor() = default;
        virtual void process(Matrix<T>& logits) const = 0;
};

template<typename T = double>
class TemperatureProcessor : public LogitProcessor<T> {
    double temperature;
    public:
        TemperatureProcessor(double temperature);
        void process(Matrix<T>& logits) const override;
};

template<typename T = double>
class TopKProcessor : public LogitProcessor<T> {
    int k;
    public:
        TopKProcessor(int k);
        void process(Matrix<T>& logits) const override;
};

template<typename T = double>
class TopPProcessor : public LogitProcessor<T> {
    double p;
    public:
        TopPProcessor(double p);
        void process(Matrix<T>& logits) const override;
};

template<typename T = double>
class TokenSelector {
    public:
        virtual ~TokenSelector() = default;
        virtual int select(const Matrix<T>& probs) const = 0;
};

template<typename T = double>
class GreedySelector : public TokenSelector<T> {
    public:
        int select(const Matrix<T>& probs) const override;
};

template<typename T = double>
class CategoricalSelector : public TokenSelector<T> {
    mutable std::mt19937 rng;
    public:
        CategoricalSelector();
        CategoricalSelector(unsigned int seed);
        int select(const Matrix<T>& probs) const override;
};

template<typename T = double>
class TokenSampler {
    std::vector<std::unique_ptr<LogitProcessor<T>>> processors;
    std::unique_ptr<TokenSelector<T>> selector;

    public:
        TokenSampler();
        void addProcessor(std::unique_ptr<LogitProcessor<T>> proc);
        void setSelector(std::unique_ptr<TokenSelector<T>> sel);
        int sample(Matrix<T> logits) const;
};
