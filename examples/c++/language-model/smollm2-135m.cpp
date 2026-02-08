#include "ml_lib/utils/llama-loader.h"
#include "ml_lib/core/token-sampler.h"

int main() {
    Tokenizer tokenizer;
    auto model = LlamaLoader<float>::load("examples/datasets/language-model/SmolLM2-135M-Instruct-f16.gguf", tokenizer, 128);

    TokenSampler<float> sampler;
    sampler.addProcessor(std::make_unique<TemperatureProcessor<float>>(0.7));
    sampler.addProcessor(std::make_unique<TopPProcessor<float>>(0.9));
    sampler.setSelector(std::make_unique<CategoricalSelector<float>>());

    std::vector<int> prompt = tokenizer.encode(
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n");

    std::cout << "Token IDs (" << prompt.size() << "): ";
    for (int id : prompt) std::cout << id << " ";
    std::cout << std::endl;

    std::vector<int> output = model->generate(prompt, 100, sampler);
    std::string text = tokenizer.decode(output);

    std::cout << "Generated text:\n" << text << std::endl;

    return 0;
}