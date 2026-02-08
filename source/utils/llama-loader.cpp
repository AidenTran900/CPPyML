#include "ml_lib/utils/llama-loader.h"
#include <stdexcept>
#include <sstream>

template<typename T>
std::unique_ptr<Transformer<T>> LlamaLoader<T>::load(const std::string& gguf_path,
                                                      Tokenizer& tokenizer,
                                                      int max_seq_len)
{
    GGUFFile gguf;
    if (!gguf.load(gguf_path)) {
        throw std::runtime_error("Failed to load GGUF file: " + gguf_path);
    }

    // Read architecture metadata
    std::string arch = gguf.getArchitecture();

    TransformerConfig config;
    config.vocab_size   = gguf.getVocabSize();
    config.embed_dim    = gguf.getEmbedDim();
    config.num_heads    = gguf.getNumHeads();
    config.num_kv_heads = gguf.getNumKVHeads();
    config.num_layers   = gguf.getNumLayers();
    config.ff_dim       = gguf.getFFDim();
    config.max_seq_len  = max_seq_len;

    // LLaMA architecture defaults
    config.norm_type     = NormType::RMS_NORM;
    config.ffn_type      = FFNType::GATED_SILU;
    config.norm_position = NormPosition::PRE_NORM;
    config.pos_enc_type  = PosEncType::ROTARY;
    config.output_norm   = true;

    // RoPE theta (LLaMA 3.x uses 500000)
    std::string theta_key = arch + ".rope.freq_base";
    if (gguf.hasKey(theta_key)) {
        config.rope_theta = gguf.getFloat(theta_key);
    }

    // Create transformer (nullptr loss/opt/reg for inference)
    auto model = std::make_unique<Transformer<T>>(
        config, nullptr, nullptr, nullptr);

    // Load tokenizer from GGUF metadata
    gguf.loadTokenizer(tokenizer);

    // Load embedding weights (not transposed — rows are vocab entries)
    model->getEmbedding().loadWeights(
        gguf.loadTensor<T>("token_embd.weight"));

    // Load per-layer weights
    // GGUF stores linear weights as (out_features, in_features),
    // but layers expect (in_features, out_features) for X * W.
    auto& blocks = model->getBlocks();
    for (int i = 0; i < config.num_layers; i++) {
        std::string prefix = "blk." + std::to_string(i) + ".";

        blocks[i]->getAttention().loadWeights(
            gguf.loadTensor<T>(prefix + "attn_q.weight").transpose(),
            gguf.loadTensor<T>(prefix + "attn_k.weight").transpose(),
            gguf.loadTensor<T>(prefix + "attn_v.weight").transpose(),
            gguf.loadTensor<T>(prefix + "attn_output.weight").transpose());

        blocks[i]->getRMSNorm1()->loadWeights(
            gguf.loadTensor<T>(prefix + "attn_norm.weight").transpose());

        blocks[i]->getRMSNorm2()->loadWeights(
            gguf.loadTensor<T>(prefix + "ffn_norm.weight").transpose());

        Matrix<T> gate_w = gguf.loadTensor<T>(prefix + "ffn_gate.weight").transpose();
        Matrix<T> up_w   = gguf.loadTensor<T>(prefix + "ffn_up.weight").transpose();
        Matrix<T> down_w = gguf.loadTensor<T>(prefix + "ffn_down.weight").transpose();

        Matrix<T> gate_bias(gate_w.cols(), 1);
        Matrix<T> up_bias(up_w.cols(), 1);
        Matrix<T> down_bias(down_w.cols(), 1);

        blocks[i]->getFFGate()->loadWeights(gate_w, gate_bias);
        blocks[i]->getFF1().loadWeights(up_w, up_bias);
        blocks[i]->getFF2().loadWeights(down_w, down_bias);
    }

    model->getOutputRMSNorm()->loadWeights(
        gguf.loadTensor<T>("output_norm.weight").transpose());

    const GGUFTensorInfo* output_tensor = gguf.findTensor("output.weight");
    Matrix<T> output_w = output_tensor
        ? gguf.loadTensor<T>("output.weight").transpose()
        : gguf.loadTensor<T>("token_embd.weight").transpose();

    Matrix<T> output_bias(output_w.cols(), 1);
    model->getOutputProjection().loadWeights(output_w, output_bias);

    if (gguf.hasKey("tokenizer.ggml.eos_token_id")) {
        model->addStopToken(static_cast<int>(gguf.getUint("tokenizer.ggml.eos_token_id")));
    }
    if (gguf.hasKey("tokenizer.ggml.eot_token_id")) {
        model->addStopToken(static_cast<int>(gguf.getUint("tokenizer.ggml.eot_token_id")));
    }

    return model;
}

template class LlamaLoader<float>;
template class LlamaLoader<double>;
