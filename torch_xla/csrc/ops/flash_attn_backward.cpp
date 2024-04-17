#include "torch_xla/csrc/ops/flash_attn_backward.h"

#include "torch_xla/csrc/flash_attn_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/service/gpu/gpu_flash_attn.h"
#include "xla/shape_util.h"

namespace torch_xla {

static int64_t RoundMultiple(int64_t x, int64_t m) {
  return (x + m - 1) / m * m;
}

static xla::Shape GetFlashAttnBwdResultShape(
    const xla::Shape& q_shape, const xla::Shape& k_shape,
    const xla::Shape& v_shape,
    const absl::optional<xla::Shape>& cu_seqlens_q_shape,
    const absl::optional<int>& max_seqlen_q) {
  int64_t batch_size, num_heads, seqlen_q;
  if (cu_seqlens_q_shape.has_value()) {  // is_varlen
    batch_size = xla::ShapeUtil::ElementsIn(*cu_seqlens_q_shape) - 1;
    num_heads = q_shape.dimensions(1);
    seqlen_q = max_seqlen_q.value();
  } else {
    batch_size = q_shape.dimensions(0);
    num_heads = q_shape.dimensions(2);
    seqlen_q = q_shape.dimensions(1);
  }

  const int64_t seqlen_q_rounded = RoundMultiple(seqlen_q, 128);

  const xla::Shape& grad_query_shape = q_shape;
  const xla::Shape& grad_key_shape = k_shape;
  const xla::Shape& grad_value_shape = v_shape;
  const xla::Shape& grad_softmax_shape = xla::ShapeUtil::MakeShape(
      xla::PrimitiveType::F32, {batch_size, num_heads, seqlen_q_rounded});

  return xla::ShapeUtil::MakeTupleShape(
      {grad_query_shape, grad_key_shape, grad_value_shape, grad_softmax_shape});
}

FlashAttnBackward::FlashAttnBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& query,
    const torch::lazy::Value& key, const torch::lazy::Value& value,
    const torch::lazy::Value& output, const torch::lazy::Value& softmax_lse,
    const torch::lazy::Value& rng_state,
    const absl::optional<torch::lazy::Value>& cu_seqlens_query,
    const absl::optional<torch::lazy::Value>& cu_seqlens_key,
    const absl::optional<int>& max_seqlen_q,
    const absl::optional<int>& max_seqlen_k, float dropout_rate, float scale,
    bool is_causal, const absl::optional<torch::lazy::Value>& alibi_slopes,
    bool deterministic)
    : XlaNode(
          torch::lazy::OpKind(xla_flash_attn_backward),
          torch_xla::runtime::util::GetValuesVector<torch::lazy::Value>(
              {grad_output, query, key, value, output, softmax_lse, rng_state},
              {&cu_seqlens_query, &cu_seqlens_key, &alibi_slopes}),
          [&]() {
            return GetFlashAttnBwdResultShape(
                GetXlaShape(query), GetXlaShape(key), GetXlaShape(value),
                GetOptionalXlaShape(cu_seqlens_query), max_seqlen_q);
          },
          /*num_outputs=*/4,
          torch::lazy::MHash(dropout_rate, scale, is_causal, deterministic)),
      max_seqlen_q_(max_seqlen_q),
      max_seqlen_k_(max_seqlen_k),
      dropout_rate_(dropout_rate),
      scale_(scale),
      is_causal_(is_causal),
      has_alibi_slopes_(alibi_slopes.has_value()),
      deterministic_(deterministic) {
  is_varlen_ = cu_seqlens_query.has_value();
  XLA_CHECK(is_varlen_ == cu_seqlens_key.has_value() &&
            is_varlen_ == max_seqlen_q.has_value() &&
            is_varlen_ == max_seqlen_k.has_value());
}

torch::lazy::NodePtr FlashAttnBackward::Clone(
    torch::lazy::OpList operands) const {
  absl::optional<torch::lazy::Value> cu_seqlens_query, cu_seqlens_key;
  if (is_varlen_) {
    cu_seqlens_query = operands.at(7);
    cu_seqlens_key = operands.at(8);
  }

  absl::optional<torch::lazy::Value> alibi_slopes;
  if (has_alibi_slopes_) {
    int64_t alibi_slopes_opnd_idx = is_varlen_ ? 9 : 7;
    XLA_CHECK(operands.size() > alibi_slopes_opnd_idx);
    alibi_slopes = operands.at(alibi_slopes_opnd_idx);
  }

  return torch::lazy::MakeNode<FlashAttnBackward>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3),
      operands.at(4), operands.at(5), operands.at(6), cu_seqlens_query,
      cu_seqlens_key, max_seqlen_q_, max_seqlen_k_, dropout_rate_, scale_,
      is_causal_, alibi_slopes, deterministic_);
}

XlaOpVector FlashAttnBackward::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> call_operands;
  for (const auto& opnd : operands()) {
    call_operands.push_back(loctx->GetOutputOp(opnd));
  }

  const xla::Shape& q_shape = ShapeHelper::ShapeOfXlaOp(call_operands[1]);
  const xla::Shape& k_shape = ShapeHelper::ShapeOfXlaOp(call_operands[2]);
  const xla::Shape& v_shape = ShapeHelper::ShapeOfXlaOp(call_operands[3]);
  absl::optional<xla::Shape> cu_seqlens_query_shape;
  if (is_varlen_) {
    cu_seqlens_query_shape = ShapeHelper::ShapeOfXlaOp(call_operands[7]);
  }

  // {grad_query, grad_key, grad_value, grad_softmax}
  const xla::Shape& result_shape = GetFlashAttnBwdResultShape(
      q_shape, k_shape, v_shape, cu_seqlens_query_shape, max_seqlen_q_);

  absl::string_view call_target_name =
      is_varlen_ ? xla::gpu::kGpuFlashAttnVarLenBwdCallTarget
                 : xla::gpu::kGpuFlashAttnBwdCallTarget;

  const std::string& backend_config = GetFlashAttnBackendConfig(
      dropout_rate_, scale_, is_causal_, deterministic_, has_alibi_slopes_,
      max_seqlen_q_, max_seqlen_k_);

  xla::XlaOp custom_call_result =
      xla::CustomCall(loctx->builder(), std::string(call_target_name),
                      call_operands, result_shape, backend_config);

  return ReturnOps(
      {
          xla::GetTupleElement(custom_call_result, 0),  // grad_query
          xla::GetTupleElement(custom_call_result, 1),  // grad_key
          xla::GetTupleElement(custom_call_result, 2),  // grad_value
          xla::GetTupleElement(custom_call_result, 3),  // grad_softmax
      },
      loctx);
}

std::string FlashAttnBackward::ToString() const {
  std::stringstream ss;
  if (is_varlen_) {
    ss << ", max_seqlen_q=" << max_seqlen_q_.value()
       << ", max_seqlen_k=" << max_seqlen_k_.value();
  }
  ss << XlaNode::ToString() << ", dropout_rate=" << dropout_rate_
     << ", scale=" << scale_ << ", is_causal=" << is_causal_
     << ", deterministic=" << deterministic_;
  return ss.str();
}

}  // namespace torch_xla
