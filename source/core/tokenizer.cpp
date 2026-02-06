#include "ml_lib/core/tokenizer.h"
#include <algorithm>
#include <cctype>

Tokenizer::Tokenizer()
    : tokenizer_type(TOKENIZER_TYPE::WORD) {}

std::vector<std::string> Tokenizer::wordTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string new_word = "";
    for (size_t i = 0; i < text.size(); ++i) {
        if (std::isspace(text[i])) {
            if (!new_word.empty()) {
                tokens.push_back(new_word);
                new_word = "";
            }
            continue;
        }
        new_word += text[i];
    }
    if (!new_word.empty()) {
        tokens.push_back(new_word);
    }
    return tokens;
}

std::vector<std::string> Tokenizer::characterTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    for (size_t i = 0; i < text.size(); ++i) {
        tokens.push_back(std::string(1, text[i]));
    }
    return tokens;
}

std::vector<std::string> Tokenizer::sentenceTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string sentence = "";
    for (size_t i = 0; i < text.size(); ++i) {
        sentence += text[i];
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            tokens.push_back(sentence);
            sentence = "";
        }
    }
    if (!sentence.empty()) {
        tokens.push_back(sentence);
    }
    return tokens;
}

void Tokenizer::trainBPE(const std::vector<std::string>& corpus, size_t num_merges) {

    // word frequencies
    std::unordered_map<std::string, int> word_freqs;

    for (const std::string& text : corpus) {
        std::string word = "";
        for (const char c : text) {
            if (std::isspace(c)) {
                if (!word.empty()) {
                    word += "</w>";
                    word_freqs[word]++;
                    word = "";
                }
            } else {
                if (!word.empty()) word += " ";
                word += c;
            }
        }
        if (!word.empty()) {
            word += " </w>";
            word_freqs[word]++;
        }
    }

    // BPE merges
    for (size_t merge_idx = 0; merge_idx < num_merges; ++merge_idx) {
        std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_counts;

        // count pairs
        for (const auto &[word, freq] : word_freqs) {
            std::vector<std::string> symbols;
            std::string current = "";
            for (const char c : word) {
                if (c == ' ') {
                    if (!current.empty()) {
                        symbols.push_back(current);
                        current = "";
                    }
                } else {
                    current += c;
                }
            }
            if (!current.empty()) symbols.push_back(current);

            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                pair_counts[{symbols[i], symbols[i + 1]}] += freq;
            }
        }

        if (pair_counts.empty()) break;

        // find best pair
        auto best_pair = std::max_element(
            pair_counts.begin(), pair_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );

        if (best_pair->second < 2) break;

        // merge best pair
        auto [first, second] = best_pair->first;
        std::string merged = first + second;

        bpe_merges.push_back({first, second});

        std::unordered_map<std::string, int> new_word_freqs;
        std::string pattern = first + " " + second;

        for (const auto& [word, freq] : word_freqs) {
            std::string new_word = word;
            size_t pos;
            while ((pos = new_word.find(pattern)) != std::string::npos) {
                new_word.replace(pos, pattern.length(), merged);
            }
            new_word_freqs[new_word] = freq;
        }
        word_freqs = std::move(new_word_freqs);
    }

    // build vocab
    int id = 0;
    for (const auto &[word, _] : word_freqs) {
        std::string current = "";
        for (const char c : word) {
            if (c == ' ') {
                if (!current.empty() && vocab.find(current) == vocab.end()) {
                    vocab[current] = id++;
                }
                current = "";
            } else {
                current += c;
            }
        }
        if (!current.empty() && vocab.find(current) == vocab.end()) {
            vocab[current] = id++;
        }
    }
}

std::vector<std::string> Tokenizer::applyBPEMerges(const std::string& word) {
    if (word.empty()) return {};

    std::vector<std::string> tokens;
    for (char c : word) {
        tokens.push_back(std::string(1, c));
    }
    tokens.back() += "</w>"; 

    for (const auto &[first, second] : bpe_merges) {
        std::vector<std::string> new_tokens;
        size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size() && tokens[i] == first && tokens[i + 1] == second) {
                new_tokens.push_back(first + second);
                i += 2;
            } else {
                new_tokens.push_back(tokens[i]);
                i++;
            }
        }
        tokens = std::move(new_tokens);
        if (tokens.size() == 1) break;
    }

    return tokens;
}

std::vector<std::string> Tokenizer::bpeTokenize(const std::string& text) {
    std::vector<std::string> tokens;

    if (bpe_merges.empty()) {
        return characterTokenize(text);
    }

    // split text to words and merge
    std::string word = "";
    for (size_t i = 0; i < text.size(); ++i) {
        if (std::isspace(text[i])) {
            if (!word.empty()) {
                auto word_tokens = applyBPEMerges(word);
                tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
                word = "";
            }
        } else {
            word += text[i];
        }
    }
    if (!word.empty()) {
        auto word_tokens = applyBPEMerges(word);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }

    return tokens;
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
