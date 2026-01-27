#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

enum TOKENIZER_TYPE {
    WORD,
    CHARACTER,
    BPE,
    SENTENCE
};

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& pair) const {
        return std::hash<std::string>()(pair.first) ^ (std::hash<std::string>()(pair.second) << 1);
    }
};

class Tokenizer {
    public:
        Tokenizer();
        std::vector<std::string> tokenize(const std::string& text);

        void trainBPE(const std::vector<std::string>& corpus, size_t num_merges);
        bool isBPETrained() const { return !bpe_merges.empty(); }

    private:
        TOKENIZER_TYPE tokenizer_type;

        std::vector<std::pair<std::string, std::string>> bpe_merges;

        std::unordered_map<std::string, int> vocab;

        std::vector<std::string> wordTokenize(const std::string& text);
        std::vector<std::string> characterTokenize(const std::string& text);
        std::vector<std::string> bpeTokenize(const std::string& text);
        std::vector<std::string> sentenceTokenize(const std::string& text);

        std::vector<std::string> applyBPEMerges(const std::string& word);
};