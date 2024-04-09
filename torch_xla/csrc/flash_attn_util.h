#ifndef XLA_TORCH_XLA_CSRC_FLASH_ATTN_UTIL_H_
#define XLA_TORCH_XLA_CSRC_FLASH_ATTN_UTIL_H_

#include <optional>
#include <string>

#include "torch_xla/csrc/tensor.h"
#include "xla/shape.h"

namespace torch_xla {

std::string GetFlashAttnBackendConfig(
    float dropout_rate, float scale, bool is_causal, bool deterministic,
    bool has_alibi_slopes,
    const std::optional<int>& max_seqlen_q = absl::nullopt,
    const std::optional<int>& max_seqlen_k = absl::nullopt);

void CheckFlashAttnFwdOperands(const XLATensorPtr& query,
                               const XLATensorPtr& key,
                               const XLATensorPtr& value, bool is_varlen,
                               const XLATensorPtr& alibi_slopes);

void CheckFlashAttnBwdOperands(const XLATensorPtr& grad_output,
                               const XLATensorPtr& query,
                               const XLATensorPtr& key,
                               const XLATensorPtr& value,
                               const XLATensorPtr& output,
                               const XLATensorPtr& softmax_lse,
                               const XLATensorPtr& rng_state, bool is_varlen,
                               const XLATensorPtr& alibi_slopes);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_FLASH_ATTN_UTIL_H_
