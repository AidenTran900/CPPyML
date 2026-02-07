#include "ml_lib/core/transformer-block.h"

template<typename T>
TransformerBlock<T>::TransformerBlock(int embed_dim, int num_heads, int ff_dim)
    : attention(embed_dim, num_heads),
      norm1(embed_dim),
      norm2(embed_dim),
      ff1(embed_dim, ff_dim, ACTIVATION_FUNC::RELU),
      ff2(ff_dim, embed_dim, ACTIVATION_FUNC::LINEAR)
{
}

template<typename T>
Matrix<T> TransformerBlock<T>::forward(const Matrix<T>& input)
{
    attention_input_cache = input;

    // Self Attention
    Matrix<T> attn_out = attention.forward(input);

    // Add & Norm
    Matrix<T> residual1 = input + attn_out;
    Matrix<T> normed1 = norm1.forward(residual1);

    // Feed Forward
    ff_input_cache = normed1;

    // Add & Norm
    Matrix<T> ff_out = ff2.forward(ff1.forward(normed1));
    Matrix<T> residual2 = normed1 + ff_out;
    Matrix<T> output = norm2.forward(residual2);

    return output;
}

template<typename T>
Matrix<T> TransformerBlock<T>::forward_cached(const Matrix<T>& input)
{
    Matrix<T> attn_out = attention.forward_cached(input);

    Matrix<T> residual1 = input + attn_out;
    Matrix<T> normed1 = norm1.forward(residual1);

    Matrix<T> ff_out = ff2.forward(ff1.forward(normed1));
    Matrix<T> residual2 = normed1 + ff_out;
    Matrix<T> output = norm2.forward(residual2);

    return output;
}

template<typename T>
void TransformerBlock<T>::clear_cache()
{
    attention.clear_cache();
}

template<typename T>
Matrix<T> TransformerBlock<T>::backward(const Matrix<T>& grad_output)
{
    Matrix<T> grad_norm2 = norm2.backward(grad_output);

    Matrix<T> grad_ff2 = ff2.backward(grad_norm2);
    Matrix<T> grad_ff1 = ff1.backward(grad_ff2);

    Matrix<T> grad_residual1 = grad_norm2 + grad_ff1;

    Matrix<T> grad_norm1 = norm1.backward(grad_residual1);

    Matrix<T> grad_attn = attention.backward(grad_norm1);

    Matrix<T> grad_input = grad_norm1 + grad_attn;

    return grad_input;
}

template<typename T>
void TransformerBlock<T>::update(Optimizer<T>* opt)
{
    attention.update(opt);
    norm1.update(opt);
    ff1.update(opt);
    ff2.update(opt);
    norm2.update(opt);
}

template class TransformerBlock<float>;
template class TransformerBlock<double>;
