#include "torch_xla/csrc/ops/flash_attn_forward.h"

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

static xla::Shape GetFlashAttnFwdResultShape(
    const xla::Shape& q_shape, const xla::Shape& k_shape,
    const absl::optional<xla::Shape>& cu_seqlens_q_shape,
    const absl::optional<int>& max_seqlen_q,
    const absl::optional<int>& max_seqlen_k, bool return_softmax) {
  int64_t batch_size, num_heads, seqlen_q, seqlen_k;
  if (cu_seqlens_q_shape.has_value()) {  // is_varlen
    batch_size = xla::ShapeUtil::ElementsIn(*cu_seqlens_q_shape) - 1;
    num_heads = q_shape.dimensions(1);
    seqlen_q = max_seqlen_q.value();
    seqlen_k = max_seqlen_k.value();
  } else {
    batch_size = q_shape.dimensions(0);
    num_heads = q_shape.dimensions(2);
    seqlen_q = q_shape.dimensions(1);
    seqlen_k = k_shape.dimensions(1);
  }

  const int64_t seqlen_q_rounded = RoundMultiple(seqlen_q, 128);
  const int64_t seqlen_k_rounded = RoundMultiple(seqlen_k, 128);

  // output_shape is same as the q_shape
  const xla::Shape& output_shape = q_shape;
  const xla::Shape& softmax_lse_shape = xla::ShapeUtil::MakeShape(
      xla::PrimitiveType::F32, {batch_size, num_heads, seqlen_q});
  const xla::Shape& rng_state_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::U64, {2});
  if (return_softmax) {
    const xla::Shape& s_dmask_shape = xla::ShapeUtil::MakeShape(
        q_shape.element_type(),
        {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
    // {output, softmax_lse, S_dmask, rng_state}
    return xla::ShapeUtil::MakeTupleShape(
        {output_shape, softmax_lse_shape, s_dmask_shape, rng_state_shape});
  }
  // {output, softmax_lse, rng_state}
  return xla::ShapeUtil::MakeTupleShape(
      {output_shape, softmax_lse_shape, rng_state_shape});
}

FlashAttnForward::FlashAttnForward(
    const torch::lazy::Value& query, const torch::lazy::Value& key,
    const torch::lazy::Value& value,
    const absl::optional<torch::lazy::Value>& cu_seqlens_query,
    const absl::optional<torch::lazy::Value>& cu_seqlens_key,
    const absl::optional<int>& max_seqlen_q,
    const absl::optional<int>& max_seqlen_k, float dropout_rate, float scale,
    bool is_causal, const absl::optional<torch::lazy::Value>& alibi_slopes,
    bool return_softmax)
    : XlaNode(
          torch::lazy::OpKind(xla_flash_attn_forward),
          torch_xla::runtime::util::GetValuesVector<torch::lazy::Value>(
              {query, key, value},
              {&cu_seqlens_query, &cu_seqlens_key, &alibi_slopes}),
          [&]() {
            return GetFlashAttnFwdResultShape(
                GetXlaShape(query), GetXlaShape(key),
                GetOptionalXlaShape(cu_seqlens_query), max_seqlen_q,
                max_seqlen_k, return_softmax);
          },
          /*num_outputs=*/return_softmax ? 4 : 3,
          torch::lazy::MHash(dropout_rate, scale, is_causal, return_softmax)),
      max_seqlen_q_(max_seqlen_q),
      max_seqlen_k_(max_seqlen_k),
      dropout_rate_(dropout_rate),
      scale_(scale),
      is_causal_(is_causal),
      has_alibi_slopes_(alibi_slopes.has_value()),
      return_softmax_(return_softmax) {
  is_varlen_ = cu_seqlens_query.has_value();
  XLA_CHECK(is_varlen_ == cu_seqlens_key.has_value() &&
            is_varlen_ == max_seqlen_q.has_value() &&
            is_varlen_ == max_seqlen_k.has_value());
}

torch::lazy::NodePtr FlashAttnForward::Clone(
    torch::lazy::OpList operands) const {
  absl::optional<torch::lazy::Value> cu_seqlens_query, cu_seqlens_key;
  if (is_varlen_) {
    cu_seqlens_query = operands.at(3);
    cu_seqlens_key = operands.at(4);
  }

  absl::optional<torch::lazy::Value> alibi_slopes;
  if (has_alibi_slopes_) {
    int64_t alibi_slopes_opnd_idx = is_varlen_ ? 5 : 3;
    XLA_CHECK(operands.size() > alibi_slopes_opnd_idx);
    alibi_slopes = operands.at(alibi_slopes_opnd_idx);
  }

  return torch::lazy::MakeNode<FlashAttnForward>(
      operands.at(0), operands.at(1), operands.at(2), cu_seqlens_query,
      cu_seqlens_key, max_seqlen_q_, max_seqlen_k_, dropout_rate_, scale_,
      is_causal_, alibi_slopes, return_softmax_);
}

XlaOpVector FlashAttnForward::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> call_operands;
  for (const auto& opnd : operands()) {
    call_operands.push_back(loctx->GetOutputOp(opnd));
  }

  const xla::Shape& q_shape = ShapeHelper::ShapeOfXlaOp(call_operands[0]);
  const xla::Shape& k_shape = ShapeHelper::ShapeOfXlaOp(call_operands[1]);
  absl::optional<xla::Shape> cu_seqlens_query_shape;
  if (is_varlen_) {
    cu_seqlens_query_shape = ShapeHelper::ShapeOfXlaOp(call_operands[3]);
  }

  const xla::Shape& result_shape =
      GetFlashAttnFwdResultShape(q_shape, k_shape, cu_seqlens_query_shape,
                                 max_seqlen_q_, max_seqlen_k_, return_softmax_);

  absl::string_view call_target_name =
      is_varlen_ ? xla::gpu::kGpuFlashAttnVarLenFwdCallTarget
                 : xla::gpu::kGpuFlashAttnFwdCallTarget;

  const std::string& backend_config = GetFlashAttnBackendConfig(
      dropout_rate_, scale_, is_causal_, /*deterministic=*/true,
      has_alibi_slopes_, max_seqlen_q_, max_seqlen_k_);

  xla::XlaOp custom_call_result =
      xla::CustomCall(loctx->builder(), std::string(call_target_name),
                      call_operands, result_shape, backend_config);

  std::vector<xla::XlaOp> op_result = {
      xla::GetTupleElement(custom_call_result, 0),  // output
      xla::GetTupleElement(custom_call_result, 1),  // softmax_lse
      xla::GetTupleElement(custom_call_result, 2),  // s_dmask or rng_state
  };
  if (return_softmax_) {
    // rng_state
    op_result.push_back(xla::GetTupleElement(custom_call_result, 3));
  }
  return ReturnOps(op_result, loctx);
}

std::string FlashAttnForward::ToString() const {
  std::stringstream ss;
  if (is_varlen_) {
    ss << ", max_seqlen_q=" << max_seqlen_q_.value()
       << ", max_seqlen_k=" << max_seqlen_k_.value();
  }
  ss << XlaNode::ToString() << ", dropout_rate=" << dropout_rate_
     << ", scale=" << scale_ << ", is_causal=" << is_causal_
     << ", return_softmax=" << return_softmax_;
  return ss.str();
}

}  // namespace torch_xla
