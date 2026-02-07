#include "ml_lib/models/transformer.h"

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
      embedding(vocab_size, embed_dim),
      pos_encoding(embed_dim, max_seq_len),
      output_projection(embed_dim, vocab_size, ACTIVATION_FUNC::SOFTMAX)
{
    for (int i = 0; i < num_layers; i++) {
        blocks.push_back(std::make_shared<TransformerBlock<T>>(embed_dim, num_heads, ff_dim));
    }
}

template<typename T>
Matrix<T> Transformer<T>::forward(const std::vector<int>& tokens)
{
    last_token_input = tokens;

    Matrix<T> embedded = embedding.forward(tokens);
    Matrix<T> x = pos_encoding.forward(embedded);

    for (auto& block : blocks) {
        x = block->forward(x);
    }

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
    output_projection.update(this->optimizer.get());
}

template<typename T>
std::vector<int> Transformer<T>::generate(const std::vector<int>& prompt, int max_tokens)
{
    clear_cache();

    std::vector<int> output = prompt;
    Matrix<T> x;

    // prefill tokens to build up the KV cache
    for (int i = 0; i < (int)prompt.size(); i++) {
        Matrix<T> embedded = embedding.forward({prompt[i]});
        x = pos_encoding.forward(embedded, i);

        for (auto& block : blocks) {
            x = block->forward_cached(x);
        }
    }

    // get first predicted token from last prefill
    Matrix<T> logits = output_projection.forward(x);
    int pos = prompt.size();

    for (int t = 0; t < max_tokens; t++) {
        // pick token with highest logit
        int best_token = 0;
        T best_val = logits(0, 0);
        for (int v = 1; v < vocab_size; v++) {
            if (logits(0, v) > best_val) {
                best_val = logits(0, v);
                best_token = v;
            }
        }

        output.push_back(best_token);
        pos++;

        if (pos >= max_seq_len) break;
        if (t == max_tokens - 1) break;

        // process new token through cache
        Matrix<T> embedded = embedding.forward({best_token});
        x = pos_encoding.forward(embedded, pos - 1);
        for (auto& block : blocks) {
            x = block->forward_cached(x);
        }
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
