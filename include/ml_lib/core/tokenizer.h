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

        void setType(TOKENIZER_TYPE type);
        TOKENIZER_TYPE getType() const { return tokenizer_type; }

        std::vector<std::string> tokenize(const std::string& text);

        void buildVocab(const std::vector<std::string>& corpus);
        void trainBPE(const std::vector<std::string>& corpus, size_t num_merges);
        bool isBPETrained() const { return !bpe_merges.empty(); }

        std::vector<int> encode(const std::string& text);
        std::string decode(const std::vector<int>& ids);

        int vocabSize() const { return static_cast<int>(vocab.size()); }

        void setVocab(const std::unordered_map<std::string, int>& v) {
            vocab = v;
            buildReverseVocab();
        }
        void setMerges(const std::vector<std::pair<std::string, std::string>>& m) {
            bpe_merges = m;
        }

    private:
        TOKENIZER_TYPE tokenizer_type;

        std::vector<std::pair<std::string, std::string>> bpe_merges;

        std::unordered_map<std::string, int> vocab;
        std::unordered_map<int, std::string> reverse_vocab;

        void buildReverseVocab();

        std::vector<std::string> wordTokenize(const std::string& text);
        std::vector<std::string> characterTokenize(const std::string& text);
        std::vector<std::string> bpeTokenize(const std::string& text);
        std::vector<std::string> sentenceTokenize(const std::string& text);

        std::vector<std::string> applyBPEMerges(const std::string& word);
};