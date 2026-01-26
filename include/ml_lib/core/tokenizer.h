#pragma once 
#include <string>
#include <vector>

enum TOKENIZER_TYPE {
    WORD,
    CHARACTER,
    BPE,
    SENTENCE
};


class Tokenizer {
    public:
        Tokenizer();
        std::vector<std::string> tokenize(const std::string& text);

    private:
        TOKENIZER_TYPE tokenizer_type;

        std::vector<std::string> wordTokenize(const std::string& text);
        std::vector<std::string> characterTokenize(const std::string& text);
        std::vector<std::string> bpeTokenize(const std::string& text);
        std::vector<std::string> sentenceTokenize(const std::string& text);
};