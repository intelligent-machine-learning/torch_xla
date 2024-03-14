#include "torch_xla/csrc/ops/scaled_dot_product_attention_backward.h"

#include "torch_xla/csrc/attention.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace torch_xla {

static xla::Shape GetSDPABackwardOutputShape(const xla::Shape& q_shape,
                                             const xla::Shape& k_shape,
                                             const xla::Shape& v_shape) {
  return xla::ShapeUtil::MakeTupleShape({q_shape, k_shape, v_shape});
}

ScaledDotProductAttentionBackward::ScaledDotProductAttentionBackward(
    const torch::lazy::Value& query, const torch::lazy::Value& key,
    const torch::lazy::Value& value, const torch::lazy::Value& activation,
    const torch::lazy::Value& grad_output, const torch::lazy::Value& fwd_output,
    const absl::optional<torch::lazy::Value>& mask,
    const absl::optional<torch::lazy::Value>& bias, double scale,
    double dropout_rate, int64_t seed, bool is_flash_attention,
    bool is_causal_mask)
    : XlaNode(
          torch::lazy::OpKind(xla_scaled_dot_product_attention_backward),
          torch_xla::runtime::util::GetValuesVector<torch::lazy::Value>(
              {query, key, value, activation, grad_output, fwd_output},
              {&mask, &bias}),
          [&]() {
            return GetSDPABackwardOutputShape(
                GetXlaShape(query), GetXlaShape(key), GetXlaShape(value));
          },
          /*num_outputs=*/3,
          torch::lazy::MHash(scale, dropout_rate, seed, is_flash_attention,
                             is_causal_mask)),
      scale_(scale),
      dropout_rate_(dropout_rate),
      seed_(seed),
      is_flash_attention_(is_flash_attention),
      is_causal_mask_(is_causal_mask) {}

torch::lazy::NodePtr ScaledDotProductAttentionBackward::Clone(
    torch::lazy::OpList operands) const {
  absl::optional<torch::lazy::Value> mask;
  if (operands.size() > 6) {
    mask = operands.at(6);
  }

  absl::optional<torch::lazy::Value> bias;
  if (operands.size() > 7) {
    bias = operands.at(7);
  }

  return torch::lazy::MakeNode<ScaledDotProductAttentionBackward>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3),
      operands.at(4), operands.at(5), mask, bias, scale_, dropout_rate_, seed_,
      is_flash_attention_, is_causal_mask_);
}

XlaOpVector ScaledDotProductAttentionBackward::Lower(
    LoweringContext* loctx) const {
  xla::XlaOp q = loctx->GetOutputOp(operand(0));
  xla::XlaOp k = loctx->GetOutputOp(operand(1));
  xla::XlaOp v = loctx->GetOutputOp(operand(2));
  xla::XlaOp activation = loctx->GetOutputOp(operand(3));
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(4));

  bool has_mask = false;
  bool has_bias = false;
  bool has_dropout = dropout_rate_ > 0;

  // {Q, K, V, activation, dO, mask*, bias*, O*}
  std::vector<xla::XlaOp> inputs = {q, k, v, activation, grad_output};
  if (operands().size() > 6) {
    inputs.push_back(loctx->GetOutputOp(operand(6)));
    has_mask = true;
  }
  if (operands().size() > 7 && is_flash_attention_) {
    inputs.push_back(loctx->GetOutputOp(operand(7)));
    has_bias = true;
  }
  if (is_flash_attention_) {
    xla::XlaOp fwd_output = loctx->GetOutputOp(operand(5));
    inputs.push_back(fwd_output);
  }

  const xla::Shape& q_shape = ShapeHelper::ShapeOfXlaOp(q);
  const xla::Shape& k_shape = ShapeHelper::ShapeOfXlaOp(k);
  const xla::Shape& v_shape = ShapeHelper::ShapeOfXlaOp(v);
  const xla::Shape& activation_shape = ShapeHelper::ShapeOfXlaOp(activation);

  // {dQ, dK, dV, d_S*, softmax_sum*, d_Q_accum*, scratch, dbias*}
  xla::Shape custom_call_result_shape =
      GetSDPABackwardOutputShape(q_shape, k_shape, v_shape);
  if (is_flash_attention_) {
    const int64_t batch = q_shape.dimensions(0);
    const int64_t num_heads = q_shape.dimensions(1);
    const int64_t q_seq_len = q_shape.dimensions(2);
    const xla::Shape& softmax_sum_shape = xla::ShapeUtil::MakeShape(
        xla::PrimitiveType::F32, {batch, num_heads, q_seq_len});
    const xla::Shape& d_Q_accum_shape =
        xla::ShapeUtil::ChangeElementType(q_shape, xla::PrimitiveType::F32);
    xla::ShapeUtil::AppendShapeToTuple(softmax_sum_shape,
                                       &custom_call_result_shape);
    xla::ShapeUtil::AppendShapeToTuple(d_Q_accum_shape,
                                       &custom_call_result_shape);
  } else {
    const xla::Shape& dS_shape = activation_shape;
    xla::ShapeUtil::AppendShapeToTuple(dS_shape, &custom_call_result_shape);
  }
  const xla::Shape& scratch_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::U8, {16});
  xla::ShapeUtil::AppendShapeToTuple(scratch_shape, &custom_call_result_shape);

  xla::XlaOp custom_call_result = xla::CustomCall(
      q.builder(),
      GetfMHACustomCallName(/*is_backward*/ true, has_mask, has_bias,
                            has_dropout),
      inputs, custom_call_result_shape,
      GetfMHABackendConfig(
          /*batch*/ q_shape.dimensions(0),
          /*num_heads*/ q_shape.dimensions(1),
          /*q_seq_len*/ q_shape.dimensions(2),
          /*kv_seq_len*/ k_shape.dimensions(2),
          /*dtype*/ q_shape.element_type(), scale_, dropout_rate_, seed_,
          is_flash_attention_, is_causal_mask_,
          /*is_backward*/ true));

  return ReturnOps({xla::GetTupleElement(custom_call_result, 0),
                    xla::GetTupleElement(custom_call_result, 1),
                    xla::GetTupleElement(custom_call_result, 2)},
                   loctx);
}

std::string ScaledDotProductAttentionBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", scale=" << scale_
     << ", dropout_rate=" << dropout_rate_ << ", seed=" << seed_
     << ", is_flash_attention=" << is_flash_attention_
     << ", is_causal_mask=" << is_causal_mask_;
  return ss.str();
}

}  // namespace torch_xla
