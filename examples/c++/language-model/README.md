# Language Model Examples

## Llama 3.2-1B Instruct

An instruction-tuned language model loaded from GGUF format with streaming token generation.

### Download Model

**Q8_0 quantized (~1.1 GB):**
```bash
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q8_0.gguf --local-dir examples/datasets/language-model/
```

> Requires `huggingface-cli`: `pip install huggingface-hub`

### Supported GGUF Quantizations

F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1

### Build & Run

```bash
cmake --build build/
./build/LlamaExample
```

To use a different quantization, update the path in `llama-3.2-1b.cpp`:
```cpp
auto model = LlamaLoader<float>::load(
    "examples/datasets/language-model/Llama-3.2-1B-Instruct-Q8_0.gguf",
    tokenizer, 2048);
```
