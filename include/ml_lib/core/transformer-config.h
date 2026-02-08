#pragma once

enum class NormType { LAYER_NORM, RMS_NORM };
enum class FFNType { STANDARD, GATED_SILU };
enum class NormPosition { PRE_NORM, POST_NORM };
enum class PosEncType { SINUSOIDAL, ROTARY };

struct TransformerConfig {
    int vocab_size;
    int embed_dim;
    int num_heads;
    int num_kv_heads = -1; // -1 means same as num_heads (standard MHA)
    int num_layers;
    int ff_dim;
    int max_seq_len;
    NormType norm_type = NormType::LAYER_NORM;
    FFNType ffn_type = FFNType::STANDARD;
    NormPosition norm_position = NormPosition::POST_NORM;
    PosEncType pos_enc_type = PosEncType::SINUSOIDAL;
    double rope_theta = 10000.0;
    bool output_norm = false;
};
