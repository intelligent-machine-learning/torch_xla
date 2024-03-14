#ifndef XLA_TORCH_XLA_CSRC_ATTENTION_H
#define XLA_TORCH_XLA_CSRC_ATTENTION_H

#include <string>

#include "xla/xla_data.pb.h"

namespace torch_xla {

std::string GetfMHACustomCallName(bool is_backward, bool has_mask,
                                  bool has_bias, bool has_dropout);

std::string GetfMHABackendConfig(int64_t batch, int64_t num_heads,
                                 int64_t q_seq_len, int64_t kv_seq_len,
                                 xla::PrimitiveType dtype, double fmha_scale,
                                 double dropout_rate, int64_t seed,
                                 bool is_flash_attention, bool is_causal_mask,
                                 bool is_backward);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_ATTENTION_H