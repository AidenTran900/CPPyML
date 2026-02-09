#include "ml_lib/models/transformer.h"
#include <algorithm>

// Legacy constructor
template<typename T>
Transformer<T>::Transformer(int vocab_size, int embed_dim, int num_heads,
                         int num_layers, int ff_dim, int max_seq_len,
                         std::unique_ptr<LossFunction<T>> loss,
                         std::unique_ptr<Optimizer<T>> opt,
                         std::unique_ptr<Regularizer<T>> reg)
    : GradientModel<T>(std::move(loss), std::move(opt), std::move(reg)),
      vocab_size(vocab_size),
      embed_dim(embed_dim),
      max_seq_len(max_seq_len),
      pos_enc_type(PosEncType::SINUSOIDAL),
      embedding(vocab_size, embed_dim),
      pos_encoding(std::make_unique<SinPositionalEncoding<T>>(embed_dim, max_seq_len)),
      output_projection(embed_dim, vocab_size, ACTIVATION_FUNC::LINEAR)
{
    for (int i = 0; i < num_layers; i++) {
        blocks.push_back(std::make_shared<TransformerBlock<T>>(embed_dim, num_heads, ff_dim));
    }
}

// Config-based constructor
template<typename T>
Transformer<T>::Transformer(const TransformerConfig& config,
                         std::unique_ptr<LossFunction<T>> loss,
                         std::unique_ptr<Optimizer<T>> opt,
                         std::unique_ptr<Regularizer<T>> reg)
    : GradientModel<T>(std::move(loss), std::move(opt), std::move(reg)),
      vocab_size(config.vocab_size),
      embed_dim(config.embed_dim),
      max_seq_len(config.max_seq_len),
      pos_enc_type(config.pos_enc_type),
      embedding(config.vocab_size, config.embed_dim),
      output_projection(config.embed_dim, config.vocab_size, ACTIVATION_FUNC::LINEAR)
{
    // Positional encoding: only create for SINUSOIDAL (RoPE is inside attention)
    if (config.pos_enc_type == PosEncType::SINUSOIDAL) {
        pos_encoding = std::make_unique<SinPositionalEncoding<T>>(
            config.embed_dim, config.max_seq_len);
    }

    // Transformer blocks with configurable norm/FFN/position
    for (int i = 0; i < config.num_layers; i++) {
        auto block = std::make_shared<TransformerBlock<T>>(
            config.embed_dim, config.num_heads, config.ff_dim,
            config.norm_type, config.ffn_type, config.norm_position,
            config.num_kv_heads);

        // Enable RoPE inside each attention layer
        if (config.pos_enc_type == PosEncType::ROTARY) {
            block->getAttention().enableRoPE(config.max_seq_len, config.rope_theta);
        }

        blocks.push_back(std::move(block));
    }

    // Optional output norm before output_projection
    if (config.output_norm) {
        if (config.norm_type == NormType::LAYER_NORM) {
            output_ln = std::make_unique<LayerNorm<T>>(config.embed_dim);
        } else {
            output_rms = std::make_unique<RMSNorm<T>>(config.embed_dim);
        }
    }
}

template<typename T>
Matrix<T> Transformer<T>::forward(const std::vector<int>& tokens)
{
    last_token_input = tokens;

    Matrix<T> embedded = embedding.forward(tokens);
    Matrix<T> x = pos_encoding ? pos_encoding->forward(embedded) : embedded;

    for (auto& block : blocks) {
        x = block->forward(x);
    }

    if (output_rms) x = output_rms->forward(x);
    else if (output_ln) x = output_ln->forward(x);

    last_logits = output_projection.forward(x);
    this->last_output = last_logits;

    return last_logits;
}

template<typename T>
Matrix<T> Transformer<T>::forward(const Matrix<T>& X)
{
    std::vector<int> tokens;
    for (int i = 0; i < X.rows(); i++) {
        tokens.push_back(static_cast<int>(X(i, 0)));
    }
    return forward(tokens);
}

template<typename T>
void Transformer<T>::backward(const Matrix<T>& y_true)
{
    Matrix<T> grad = this->loss_func->gradient(last_logits, y_true);

    grad = output_projection.backward(grad);

    if (output_rms) grad = output_rms->backward(grad);
    else if (output_ln) grad = output_ln->backward(grad);

    for (int i = blocks.size() - 1; i >= 0; i--) {
        grad = blocks[i]->backward(grad);
    }

    embedding.backward(grad);
}

template<typename T>
void Transformer<T>::update()
{
    embedding.update(this->optimizer.get());
    for (auto& block : blocks) {
        block->update(this->optimizer.get());
    }
    if (output_rms) output_rms->update(this->optimizer.get());
    else if (output_ln) output_ln->update(this->optimizer.get());
    output_projection.update(this->optimizer.get());
}

template<typename T>
std::vector<int> Transformer<T>::generate(const std::vector<int>& prompt, int max_tokens)
{
    TokenSampler<T> greedy;
    return generate(prompt, max_tokens, greedy);
}

template<typename T>
std::vector<int> Transformer<T>::generate(const std::vector<int>& prompt, int max_tokens,
                                          const TokenSampler<T>& sampler)
{
    return generate(prompt, max_tokens, sampler, nullptr);
}

template<typename T>
std::vector<int> Transformer<T>::generate(const std::vector<int>& prompt, int max_tokens,
                                          const TokenSampler<T>& sampler,
                                          std::function<void(int)> on_token)
{
    clear_cache();

    std::vector<int> output = prompt;

    // Batched prefill — process all prompt tokens at once
    Matrix<T> embedded = embedding.forward(prompt);  // (seq_len, embed_dim)
    Matrix<T> x = pos_encoding ? pos_encoding->forward(embedded) : embedded;

    for (auto& block : blocks) {
        x = block->forward_prefill(x);
    }

    // Extract last row for output projection
    int seq_len = x.rows();
    Matrix<T> last_row(1, embed_dim);
    for (int j = 0; j < embed_dim; j++) {
        last_row(0, j) = x(seq_len - 1, j);
    }
    x = last_row;

    if (output_rms) x = output_rms->forward(x);
    else if (output_ln) x = output_ln->forward(x);

    // get first predicted token from last prefill
    Matrix<T> logits = output_projection.forward(x);
    int pos = prompt.size();

    for (int t = 0; t < max_tokens; t++) {
        int token = sampler.sample(logits);

        output.push_back(token);
        if (on_token) on_token(token);
        pos++;

        if (std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end()) break;
        if (pos >= max_seq_len) break;
        if (t == max_tokens - 1) break;

        // process new token through cache
        Matrix<T> embedded = embedding.forward({token});
        x = pos_encoding ? pos_encoding->forward(embedded, pos - 1) : embedded;
        for (auto& block : blocks) {
            x = block->forward_cached(x);
        }

        if (output_rms) x = output_rms->forward(x);
        else if (output_ln) x = output_ln->forward(x);

        logits = output_projection.forward(x);
    }

    return output;
}

template<typename T>
void Transformer<T>::clear_cache()
{
    for (auto& block : blocks) {
        block->clear_cache();
    }
}

// Explicit template instantiation
template class Transformer<float>;
template class Transformer<double>;
