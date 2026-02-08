#pragma once
#include "gguf-loader.h"
#include "../models/transformer.h"
#include "../core/tokenizer.h"
#include <memory>
#include <string>

template<typename T = float>
class LlamaLoader {
    public:
        static std::unique_ptr<Transformer<T>> load(const std::string& gguf_path,
                                                     Tokenizer& tokenizer,
                                                     int max_seq_len = 2048);
};
