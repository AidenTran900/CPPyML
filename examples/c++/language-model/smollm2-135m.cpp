#include "ml_lib/utils/llama-loader.h"
#include "ml_lib/core/token-sampler.h"

int main() {
    Tokenizer tokenizer;
    auto model = LlamaLoader<float>::load("examples/datasets/language-model/Llama-3.2-1B-Instruct-Q8_0.gguf", tokenizer, 2048);

    TokenSampler<float> sampler;
    sampler.addProcessor(std::make_unique<TemperatureProcessor<float>>(0.7));
    sampler.addProcessor(std::make_unique<TopPProcessor<float>>(0.9));
    sampler.setSelector(std::make_unique<CategoricalSelector<float>>());

    std::string input;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, input) || input == "exit") break;
        if (input.empty()) continue;

       std::string formatted =
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant.<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            + input +
            "\n<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>";
        std::vector<int> prompt = tokenizer.encode(formatted);
#ifdef ML_USE_CUDA
        std::vector<int> output = model->generate_gpu(prompt, 2048, sampler);
#else
        std::vector<int> output = model->generate(prompt, 2048, sampler);
#endif
        std::vector<int> generated(output.begin() + prompt.size(), output.end());
        std::cout << tokenizer.decode(generated) << std::endl;
    }

    return 0;
}