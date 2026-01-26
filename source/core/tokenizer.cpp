#include "tokenizer.h"

Tokenizer::Tokenizer()
    : tokenizer_type(TOKENIZER_TYPE::WORD) {}  

std::vector<std::string> wordTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string new_word = "";
    for (size_t i = 0; i < text.size(); ++i) {
        if (std::isspace(text[i])) {
            tokens.push_back(new_word);
            new_word = "";
            continue;
        }
        new_word += text[i];
    }

    return tokens;
}
std::vector<std::string> characterTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    for (size_t i = 0; i < text.size(); ++i) {
        tokens.push_back(std::string(1, text[i]));
    }
    return tokens;
}
std::vector<std::string> bpeTokenize(const std::string& text) {

}
std::vector<std::string> sentenceTokenize(const std::string& text) {

}


std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    switch (tokenizer_type) {
        case TOKENIZER_TYPE::WORD:
            return wordTokenize(text);
        case TOKENIZER_TYPE::CHARACTER:
            return characterTokenize(text);
        case TOKENIZER_TYPE::BPE:
            return bpeTokenize(text);
        case TOKENIZER_TYPE::SENTENCE:
            return sentenceTokenize(text);
        default:
            return {};
    }
}
