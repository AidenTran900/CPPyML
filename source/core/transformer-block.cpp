#include "ml_lib/core/transformer-block.h"

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, int ff_dim)
    : attention(embed_dim, num_heads),
      norm1(embed_dim),
      norm2(embed_dim),
      ff1(embed_dim, ff_dim, ACTIVATION_FUNC::RELU),
      ff2(ff_dim, embed_dim, ACTIVATION_FUNC::LINEAR)
{
}

Matrix TransformerBlock::forward(const Matrix& input)
{
    attention_input_cache = input;

    // Self Attention
    Matrix attn_out = attention.forward(input);

    // Add & Norm
    Matrix residual1 = input + attn_out;
    Matrix normed1 = norm1.forward(residual1);

    // Feed Forward
    ff_input_cache = normed1;

    // Add & Norm
    Matrix ff_out = ff2.forward(ff1.forward(normed1));
    Matrix residual2 = normed1 + ff_out;
    Matrix output = norm2.forward(residual2);

    return output;
}

Matrix TransformerBlock::forward_cached(const Matrix& input)
{
    Matrix attn_out = attention.forward_cached(input);

    Matrix residual1 = input + attn_out;
    Matrix normed1 = norm1.forward(residual1);

    Matrix ff_out = ff2.forward(ff1.forward(normed1));
    Matrix residual2 = normed1 + ff_out;
    Matrix output = norm2.forward(residual2);

    return output;
}

void TransformerBlock::clear_cache()
{
    attention.clear_cache();
}

Matrix TransformerBlock::backward(const Matrix& grad_output)
{
    Matrix grad_norm2 = norm2.backward(grad_output);

    Matrix grad_ff2 = ff2.backward(grad_norm2);
    Matrix grad_ff1 = ff1.backward(grad_ff2);

    Matrix grad_residual1 = grad_norm2 + grad_ff1;

    Matrix grad_norm1 = norm1.backward(grad_residual1);

    Matrix grad_attn = attention.backward(grad_norm1);

    Matrix grad_input = grad_norm1 + grad_attn;

    return grad_input;
}

void TransformerBlock::update(Optimizer* opt)
{
    attention.update(opt);
    norm1.update(opt);
    ff1.update(opt);
    ff2.update(opt);
    norm2.update(opt);
}
