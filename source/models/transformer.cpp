#include "ml_lib/models/transformer.h"

Transformer::Transformer(int vocab_size, int embed_dim, int num_heads,
                         int num_layers, int ff_dim, int max_seq_len,
                         std::unique_ptr<LossFunction> loss,
                         std::unique_ptr<Optimizer> opt,
                         std::unique_ptr<Regularizer> reg)
    : GradientModel(std::move(loss), std::move(opt), std::move(reg)),
      vocab_size(vocab_size),
      embed_dim(embed_dim),
      max_seq_len(max_seq_len),
      embedding(vocab_size, embed_dim),
      pos_encoding(embed_dim, max_seq_len),
      output_projection(embed_dim, vocab_size, ACTIVATION_FUNC::SOFTMAX)
{
    for (int i = 0; i < num_layers; i++) {
        blocks.push_back(std::make_shared<TransformerBlock>(embed_dim, num_heads, ff_dim));
    }
}

Matrix Transformer::forward(const std::vector<int>& tokens)
{
    last_token_input = tokens;

    Matrix embedded = embedding.forward(tokens);
    Matrix x = pos_encoding.forward(embedded);

    for (auto& block : blocks) {
        x = block->forward(x);
    }

    last_logits = output_projection.forward(x);
    last_output = last_logits;

    return last_logits;
}

Matrix Transformer::forward(const Matrix& X)
{
    std::vector<int> tokens;
    for (int i = 0; i < X.rows(); i++) {
        tokens.push_back(static_cast<int>(X(i, 0)));
    }
    return forward(tokens);
}

void Transformer::backward(const Matrix& y_true)
{
    Matrix grad = loss_func->gradient(last_logits, y_true);

    grad = output_projection.backward(grad);

    for (int i = blocks.size() - 1; i >= 0; i--) {
        grad = blocks[i]->backward(grad);
    }

    embedding.backward(grad);
}

void Transformer::update()
{
    embedding.update(optimizer.get());
    for (auto& block : blocks) {
        block->update(optimizer.get());
    }
    output_projection.update(optimizer.get());
}
