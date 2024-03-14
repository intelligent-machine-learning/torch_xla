#ifndef XLA_TORCH_XLA_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_H_

#include "absl/types/optional.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ScaledDotProductAttentionBackward : public XlaNode {
 public:
  ScaledDotProductAttentionBackward(
      const torch::lazy::Value& query, const torch::lazy::Value& key,
      const torch::lazy::Value& value, const torch::lazy::Value& activation,
      const torch::lazy::Value& grad_output,
      const torch::lazy::Value& fwd_output,
      const absl::optional<torch::lazy::Value>& mask,
      const absl::optional<torch::lazy::Value>& bias, double scale,
      double dropout_rate, int64_t seed, bool is_flash_attention,
      bool is_causal_mask);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

 private:
  double scale_;
  double dropout_rate_;
  int64_t seed_;
  bool is_flash_attention_;
  bool is_causal_mask_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_H_