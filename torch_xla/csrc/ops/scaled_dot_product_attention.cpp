#include "torch_xla/csrc/ops/scaled_dot_product_attention.h"

#include "torch_xla/csrc/attention.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace torch_xla {

static xla::Shape GetSDPAForwardOutputShape(const xla::Shape& q_shape,
                                            const xla::Shape& k_shape,
                                            bool is_flash_attention) {
  const int64_t batch = q_shape.dimensions(0);
  const int64_t num_heads = q_shape.dimensions(1);
  const int64_t q_seq_len = q_shape.dimensions(2);
  const int64_t kv_seq_len = k_shape.dimensions(2);

  // output_shape is same as the q_shape
  const xla::Shape& output_shape = q_shape;
  if (is_flash_attention) {
    const xla::Shape& softmax_stat_shape = xla::ShapeUtil::MakeShape(
        xla::PrimitiveType::F32, {batch, num_heads, q_seq_len});
    // {output, softmax_stat}
    return xla::ShapeUtil::MakeTupleShape({output_shape, softmax_stat_shape});
  } else {
    const xla::Shape& activation_shape = xla::ShapeUtil::MakeShape(
        q_shape.element_type(), {batch, num_heads, q_seq_len, kv_seq_len});
    // {output, activation}
    return xla::ShapeUtil::MakeTupleShape({output_shape, activation_shape});
  }
}

ScaledDotProductAttention::ScaledDotProductAttention(
    const torch::lazy::Value& query, const torch::lazy::Value& key,
    const torch::lazy::Value& value,
    const absl::optional<torch::lazy::Value>& mask,
    const absl::optional<torch::lazy::Value>& bias, double scale,
    double dropout_rate, int64_t seed, bool is_flash_attention,
    bool is_causal_mask)
    : XlaNode(
          torch::lazy::OpKind(xla_scaled_dot_product_attention),
          torch_xla::runtime::util::GetValuesVector<torch::lazy::Value>(
              {query, key, value}, {&mask, &bias}),
          [&]() {
            return GetSDPAForwardOutputShape(
                GetXlaShape(query), GetXlaShape(key), is_flash_attention);
          },
          /*num_outputs=*/2,
          torch::lazy::MHash(scale, dropout_rate, seed, is_flash_attention,
                             is_causal_mask)),
      scale_(scale),
      dropout_rate_(dropout_rate),
      seed_(seed),
      is_flash_attention_(is_flash_attention),
      is_causal_mask_(is_causal_mask) {}

torch::lazy::NodePtr ScaledDotProductAttention::Clone(
    torch::lazy::OpList operands) const {
  absl::optional<torch::lazy::Value> mask;
  if (operands.size() > 3) {
    mask = operands.at(3);
  }

  absl::optional<torch::lazy::Value> bias;
  if (operands.size() > 4) {
    bias = operands.at(4);
  }

  return torch::lazy::MakeNode<ScaledDotProductAttention>(
      operands.at(0), operands.at(1), operands.at(2), mask, bias, scale_,
      dropout_rate_, seed_, is_flash_attention_, is_causal_mask_);
}

XlaOpVector ScaledDotProductAttention::Lower(LoweringContext* loctx) const {
  xla::XlaOp q = loctx->GetOutputOp(operand(0));
  xla::XlaOp k = loctx->GetOutputOp(operand(1));
  xla::XlaOp v = loctx->GetOutputOp(operand(2));

  bool has_mask = false;
  bool has_bias = false;
  bool has_dropout = dropout_rate_ > 0;

  // {Q, K, V, mask*, bias*}
  std::vector<xla::XlaOp> inputs = {q, k, v};
  if (operands().size() > 3) {
    inputs.push_back(loctx->GetOutputOp(operand(3)));
    has_mask = true;
  }
  if (operands().size() > 4) {
    inputs.push_back(loctx->GetOutputOp(operand(4)));
    has_bias = true;
  }

  xla::Shape q_shape = ShapeHelper::ShapeOfXlaOp(q);
  xla::Shape k_shape = ShapeHelper::ShapeOfXlaOp(k);

  // {output, scratch, softmax_stat/activation}
  xla::Shape custom_call_result_shape =
      GetSDPAForwardOutputShape(q_shape, k_shape, is_flash_attention_);
  const xla::Shape& scratch_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::U8, {16});
  custom_call_result_shape.mutable_tuple_shapes()->insert(
      custom_call_result_shape.tuple_shapes().begin() + 1, scratch_shape);

  xla::XlaOp custom_call_result = xla::CustomCall(
      q.builder(),
      GetfMHACustomCallName(/*is_backward*/ false, has_mask, has_bias,
                            has_dropout),
      inputs, custom_call_result_shape,
      GetfMHABackendConfig(
          /*batch*/ q_shape.dimensions(0),
          /*num_heads*/ q_shape.dimensions(1),
          /*q_seq_len*/ q_shape.dimensions(2),
          /*kv_seq_len*/ k_shape.dimensions(2),
          /*dtype*/ q_shape.element_type(), scale_, dropout_rate_, seed_,
          is_flash_attention_, is_causal_mask_,
          /*is_backward*/ false));

  return ReturnOps({xla::GetTupleElement(custom_call_result, 0),
                    xla::GetTupleElement(custom_call_result, 2)},
                   loctx);
}

std::string ScaledDotProductAttention::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", scale=" << scale_
     << ", dropout_rate=" << dropout_rate_ << ", seed=" << seed_
     << ", is_flash_attention=" << is_flash_attention_
     << ", is_causal_mask=" << is_causal_mask_;
  return ss.str();
}

}  // namespace torch_xla
