# Language Model Examples

## SmolLM2-135M

A small instruction-tuned language model loaded from GGUF format.

### Download Model

**F16 (270 MB):**
```bash
huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-f16.gguf --local-dir examples/datasets/language-model/
```

**Q8_0 quantized (143 MB):**
```bash
huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf --local-dir examples/datasets/language-model/
```

> Requires `huggingface-cli`: `pip install huggingface-hub`

### Supported GGUF Quantizations

F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1

### Run

```bash
cmake --build build/
./build/Build
```

To use the quantized model, update the path in `smollm2-135m.cpp`:
```cpp
auto model = LlamaLoader<float>::load(
    "examples/datasets/language-model/SmolLM2-135M-Instruct-Q8_0.gguf",
    tokenizer, 128);
```
