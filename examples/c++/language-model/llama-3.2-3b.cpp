#include "ml_lib/utils/llama-loader.h"
#include "ml_lib/core/token-sampler.h"

Tokenizer tokenizer;
auto model = LlamaLoader<float>::load("llama-3.2-3b.f16.gguf", tokenizer, 2048);

TokenSampler<float> sampler;
sampler.addProcessor(std::make_unique<TemperatureProcessor<float>>(0.7));
sampler.addProcessor(std::make_unique<TopPProcessor<float>>(0.9));
sampler.setSelector(std::make_unique<CategoricalSelector<float>>());

std::vector<int> prompt = tokenizer.encode("Once upon a time");
std::vector<int> output = model->generate(prompt, 100, sampler);
std::string text = tokenizer.decode(output);