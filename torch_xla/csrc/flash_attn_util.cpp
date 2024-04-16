#include "torch_xla/csrc/flash_attn_util.h"

#include "single_include/nlohmann/json.hpp"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "xla/shape_util.h"

namespace torch_xla {

std::string GetFlashAttnBackendConfig(
    float dropout_rate, float scale, bool is_causal, bool deterministic,
    bool has_alibi_slopes, const std::optional<int>& max_seqlen_q,
    const std::optional<int>& max_seqlen_k,
    const std::optional<bool>& return_softmax) {
  using json = nlohmann::json;
  json flash_attn_backend_config = {
      {"dropout_rate", dropout_rate},
      {"scale", scale},
      {"is_causal", is_causal},
      {"deterministic", deterministic},
      {"has_alibi_slopes", has_alibi_slopes},
  };
  bool is_varlen = max_seqlen_q.has_value();
  XLA_CHECK(is_varlen == max_seqlen_k.has_value());
  if (is_varlen) {
    flash_attn_backend_config["max_seqlen_q"] = max_seqlen_q.value();
    flash_attn_backend_config["max_seqlen_k"] = max_seqlen_k.value();
  }
  if (return_softmax.has_value()) {
    flash_attn_backend_config["return_softmax"] = return_softmax.value();
  }
  json backend_config = {
      {"operation_queue_id", "0"},
      {"wait_on_operation_queues", json::array()},
      {"flash_attn_backend_config", flash_attn_backend_config},
  };
  return backend_config.dump();
}

static void CheckFlashAttnCommonOperands(
    const xla::Shape& q_shape, const xla::Shape& k_shape,
    const xla::Shape& v_shape, bool is_varlen,
    const std::optional<xla::Shape>& alibi_slopes_shape) {
  int64_t rank = is_varlen ? 3 : 4;
  XLA_CHECK(q_shape.rank() == rank && k_shape.rank() == rank &&
            v_shape.rank() == rank)
      << "query, key and value should have rank " << rank << '.';

  xla::PrimitiveType q_dtype = q_shape.element_type();
  xla::PrimitiveType k_dtype = k_shape.element_type();
  xla::PrimitiveType v_dtype = v_shape.element_type();

  XLA_CHECK(q_dtype == k_dtype && q_dtype == v_dtype &&
            (q_dtype == xla::PrimitiveType::BF16 ||
             q_dtype == xla::PrimitiveType::F16))
      << "query, key and value should have same dtype and should be float16 or "
         "bfloat16";

  int num_heads_idx = is_varlen ? 1 : 2;
  int head_dim_idx = is_varlen ? 2 : 3;

  const int64_t q_num_heads = q_shape.dimensions(num_heads_idx);
  const int64_t q_head_dim = q_shape.dimensions(head_dim_idx);
  const int64_t k_num_heads = k_shape.dimensions(num_heads_idx);
  const int64_t k_head_dim = k_shape.dimensions(head_dim_idx);
  const int64_t v_num_heads = v_shape.dimensions(num_heads_idx);
  const int64_t v_head_dim = v_shape.dimensions(head_dim_idx);

  XLA_CHECK(k_num_heads == v_num_heads)
      << "Key and value in in FlashAttention should have same number of heads.";
  XLA_CHECK(q_head_dim == k_head_dim && q_head_dim == v_head_dim)
      << "Query, key and value in FlashAttention should have same head "
         "dimension.";

  int64_t q_batch;

  if (is_varlen) {
    const int64_t total_k = k_shape.dimensions(0);
    const int64_t total_v = v_shape.dimensions(0);
    XLA_CHECK(total_k == total_v)
        << "Key and value in FlashAttention should have same total sequence "
           "length.";
  } else {
    q_batch = q_shape.dimensions(0);
    const int64_t k_batch = k_shape.dimensions(0);
    const int64_t k_seq_len = k_shape.dimensions(1);
    const int64_t v_batch = v_shape.dimensions(0);
    const int64_t v_seq_len = v_shape.dimensions(1);
    XLA_CHECK(q_batch == k_batch && q_batch == v_batch)
        << "Query, key and value in FlashAttention should have same batch "
           "size.";
    XLA_CHECK(k_seq_len == v_seq_len) << "Key and value in in FlashAttention "
                                         "should have same sequence length.";
  }

  // num_heads or batch_size x num_heads
  if (alibi_slopes_shape.has_value()) {
    XLA_CHECK(alibi_slopes_shape->element_type() == xla::PrimitiveType::F32)
        << "Alibi slopes should be float32.";
    int64_t rank = alibi_slopes_shape->rank();
    XLA_CHECK(rank == 1 || rank == 2)
        << "Alibi slopes should be a vector or a matrix.";
    XLA_CHECK(alibi_slopes_shape->dimensions(rank - 1) == q_num_heads)
        << "Alibi slopes should have same number of heads as query.";
    if (rank == 2 && !is_varlen) {
      XLA_CHECK(alibi_slopes_shape->dimensions(0) == q_batch)
          << "Alibi slopes should have same batch size as query.";
    }
  }
}

void CheckFlashAttnFwdOperands(const XLATensorPtr& query,
                               const XLATensorPtr& key,
                               const XLATensorPtr& value, bool is_varlen,
                               const XLATensorPtr& alibi_slopes) {
  std::optional<xla::Shape> alibi_slopes_shape;
  if (alibi_slopes) {
    alibi_slopes_shape = alibi_slopes->shape().get();
  }
  CheckFlashAttnCommonOperands(query->shape().get(), key->shape().get(),
                               value->shape().get(), is_varlen,
                               alibi_slopes_shape);
}

void CheckFlashAttnBwdOperands(const XLATensorPtr& grad_output,
                               const XLATensorPtr& query,
                               const XLATensorPtr& key,
                               const XLATensorPtr& value,
                               const XLATensorPtr& output,
                               const XLATensorPtr& softmax_lse,
                               const XLATensorPtr& rng_state, bool is_varlen,
                               const XLATensorPtr& alibi_slopes) {
  const xla::Shape& q_shape = query->shape().get();
  std::optional<xla::Shape> alibi_slopes_shape;
  if (alibi_slopes) {
    alibi_slopes_shape = alibi_slopes->shape().get();
  }
  CheckFlashAttnCommonOperands(q_shape, key->shape().get(),
                               value->shape().get(), is_varlen,
                               alibi_slopes_shape);

  XLA_CHECK(xla::ShapeUtil::Compatible(grad_output->shape().get(), q_shape))
      << "Grad_output and query should have same shape.";
  XLA_CHECK(xla::ShapeUtil::Compatible(output->shape().get(), q_shape))
      << "Forward output and query should have same shape.";

  const xla::Shape& softmax_lse_shape = softmax_lse->shape().get();
  XLA_CHECK(softmax_lse_shape.rank() == 3) << "Softmax LSE should have rank 3.";
  XLA_CHECK(softmax_lse_shape.element_type() == xla::PrimitiveType::F32)
      << "Softmax LSE should be float32.";
  // batch x head_dim x seq_len
  if (is_varlen) {
    XLA_CHECK(softmax_lse_shape.dimensions(1) == q_shape.dimensions(1))
        << "Softmax LSE should have same number of heads as query.";
  } else {
    XLA_CHECK(softmax_lse_shape.dimensions(0) == q_shape.dimensions(0) &&
              softmax_lse_shape.dimensions(1) == q_shape.dimensions(2) &&
              softmax_lse_shape.dimensions(2) == q_shape.dimensions(1))
        << "Softmax LSE should have same batch size, sequence length and "
           "number "
           "of heads as query.";
  }

  const xla::Shape& rng_state_shape = rng_state->shape().get();
  XLA_CHECK(rng_state_shape.rank() == 1 && rng_state_shape.dimensions(0) == 2 &&
            rng_state_shape.element_type() == xla::PrimitiveType::U64)
      << "RNG state should be a u64[2]";
}

}  // namespace torch_xla
