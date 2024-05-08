#ifndef XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTN_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTN_BACKWARD_H_

#include "absl/types/optional.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class FlashAttnBackward : public XlaNode {
 public:
  FlashAttnBackward(
      const torch::lazy::Value& grad_output, const torch::lazy::Value& query,
      const torch::lazy::Value& key, const torch::lazy::Value& value,
      const torch::lazy::Value& output, const torch::lazy::Value& softmax_lse,
      const torch::lazy::Value& rng_state,
      const absl::optional<torch::lazy::Value>& cu_seqlens_query,
      const absl::optional<torch::lazy::Value>& cu_seqlens_key,
      const absl::optional<int>& max_seqlen_q,
      const absl::optional<int>& max_seqlen_k, float dropout_rate, float scale,
      bool is_causal, const absl::optional<torch::lazy::Value>& alibi_slopes,
      bool deterministic);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

 private:
  bool is_varlen_;
  absl::optional<int> max_seqlen_q_;
  absl::optional<int> max_seqlen_k_;
  float dropout_rate_;
  double scale_;
  bool is_causal_;
  bool has_alibi_slopes_;
  bool deterministic_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTN_BACKWARD_H_