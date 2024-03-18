#include "torch_xla/csrc/attention.h"

#include <map>
#include <string>
#include <tuple>

#include "absl/strings/string_view.h"
#include "single_include/nlohmann/json.hpp"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape_util.h"

namespace torch_xla {

std::string GetfMHACustomCallName(bool is_backward, bool has_mask,
                                  bool has_bias, bool has_dropout) {
  static std::map<std::tuple<bool, bool, bool, bool>, absl::string_view>
      custom_name_maps = {
          /* fMHA forward call targets */
          {
              {false, false, false, false},
              // bmm-softmax-bmm
              xla::gpu::kCudnnfMHASoftmaxCallTarget,
          },
          {
              {false, false, false, true},
              // bmm-softmax-dropout-bmm
              xla::gpu::kCudnnfMHASoftmaxDropoutCallTarget,
          },
          {
              {false, false, true, false},
              // bmm-scale-bias-softmax-bmm
              xla::gpu::kCudnnfMHAScaleBiasSoftmaxCallTarget,
          },
          {
              {false, false, true, true},
              // bmm-scale-bias-softmax-dropout-bmm
              xla::gpu::kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget,
          },
          {
              {false, true, false, false},
              // bmm-scale-mask-softmax-bmm
              xla::gpu::kCudnnfMHAScaleMaskSoftmaxCallTarget,
          },
          {
              {false, true, false, true},
              // bmm-scale-mask-softmax-dropout-bmm
              xla::gpu::kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget,
          },
          {
              {false, true, true, false},
              // bmm-scale-bias-mask-softmax-bmm
              xla::gpu::kCudnnfMHAScaleBiasMaskSoftmaxCallTarget,
          },
          {
              {false, true, true, true},
              // bmm-scale-bias-mask-softmax-dropout-bm
              xla::gpu::kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget,
          },

          /* fMHA backward call targets */
          {
              {true, false, false, false},
              // bmm-softmax-bmm-backward
              xla::gpu::kCudnnfMHASoftmaxBackwardCallTarget,
          },
          {
              {true, false, false, true},
              // bmm-softmax-dropout-bmm-backward
              xla::gpu::kCudnnfMHASoftmaxDropoutBackwardCallTarget,
          },
          {
              {true, false, true, false},
              // bmm-scale-bias-softmax-bmm-backward
              xla::gpu::kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget,
          },
          {
              {true, false, true, true},
              // bmm-scale-bias-softmax-dropout-bmm-backward
              xla::gpu::kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget,
          },
          {
              {true, true, false, false},
              // bmm-scale-mask-softmax-bmm-backward
              xla::gpu::kCudnnfMHAScaleMaskSoftmaxBackwardCallTarget,
          },
          {
              {true, true, false, true},
              // bmm-scale-mask-softmax-dropout-bmm-backward
              xla::gpu::kCudnnfMHAScaleMaskSoftmaxDropoutBackwardCallTarget,
          },
          {
              {true, true, true, false},
              // bmm-scale-bias-mask-softmax-bmm-backward
              xla::gpu::kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget,
          },
          {
              {true, true, true, true},
              // fmha-bmm-scale-bias-mask-softmax-dropout-bmm-backward
              xla::gpu::kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget,
          },
      };

  absl::string_view custom_name = custom_name_maps[std::make_tuple(
      is_backward, has_mask, has_bias, has_dropout)];
  return std::string(custom_name);
}

std::string GetfMHABackendConfig(int64_t batch, int64_t num_heads,
                                 int64_t q_seq_len, int64_t kv_seq_len,
                                 xla::PrimitiveType dtype, double fmha_scale,
                                 double dropout_rate, int64_t seed,
                                 bool is_flash_attention, bool is_causal_mask,
                                 bool is_backward) {
  using json = nlohmann::json;
  json algorithm = {
      {"algo_id", "0"},
      {"math_type", "TENSOR_OP_MATH"},
      {"tuning_knobs", {{"17", "1"}, {"24", "0"}}},
      {"is_cudnn_frontend", true},
      {"workspace_size", "0"},
  };
  json intermediate_tensor_shape = {
      {"element_type", xla::PrimitiveType_Name(dtype)},
      {
          "dimensions",
          {
              std::to_string(batch),
              std::to_string(num_heads),
              std::to_string(q_seq_len),
              std::to_string(kv_seq_len),
          },
      },
      {"tuple_shapes", json::array()},
      {
          "layout",
          {
              {"dim_level_types", json::array()},
              {"dim_unique", json::array()},
              {"dim_ordered", json::array()},
              {"minor_to_major", {"3", "2", "1", "0"}},
              {"tiles", json::array()},
              {"element_size_in_bits", "0"},
              {"memory_space", "0"},
              {"index_primitive_type", "PRIMITIVE_TYPE_INVALID"},
              {"pointer_primitive_type", "PRIMITIVE_TYPE_INVALID"},
              {"dynamic_shape_metadata_prefix_bytes", "0"},
          },
      },
      {"is_dynamic_dimension", {false, false, false, false}},
  };
  json cudnn_fmha_backend_config = {
      {"algorithm", algorithm},
      {"intermediate_tensor_shape", intermediate_tensor_shape},
      {"fmha_scale", fmha_scale},
      {"dropout_rate", dropout_rate},
      {"seed", seed},
      {"is_flash_attention", is_flash_attention},
      {"is_causal_mask", is_causal_mask},
  };
  if (!is_backward) {
    cudnn_fmha_backend_config["bmm1_dot_dimension_numbers"] = {
        {"lhs_contracting_dimensions", {"3"}},
        {"rhs_contracting_dimensions", {"3"}},
        {"lhs_batch_dimensions", {"0", "1"}},
        {"rhs_batch_dimensions", {"0", "1"}},
    };
    cudnn_fmha_backend_config["bmm2_dot_dimension_numbers"] = {
        {"lhs_contracting_dimensions", {"3"}},
        {"rhs_contracting_dimensions", {"2"}},
        {"lhs_batch_dimensions", {"0", "1"}},
        {"rhs_batch_dimensions", {"0", "1"}},
    };
  } else {
    cudnn_fmha_backend_config["bmm1_grad_gemm1_dot_dimension_numbers"] = {
        {"lhs_contracting_dimensions", {"2"}},
        {"rhs_contracting_dimensions", {"2"}},
        {"lhs_batch_dimensions", {"0", "1"}},
        {"rhs_batch_dimensions", {"0", "1"}},
    };
    cudnn_fmha_backend_config["bmm1_grad_gemm2_dot_dimension_numbers"] = {
        {"lhs_contracting_dimensions", {"3"}},
        {"rhs_contracting_dimensions", {"2"}},
        {"lhs_batch_dimensions", {"0", "1"}},
        {"rhs_batch_dimensions", {"0", "1"}},
    };
    cudnn_fmha_backend_config["bmm2_grad_gemm1_dot_dimension_numbers"] = {
        {"lhs_contracting_dimensions", {"2"}},
        {"rhs_contracting_dimensions", {"2"}},
        {"lhs_batch_dimensions", {"0", "1"}},
        {"rhs_batch_dimensions", {"0", "1"}},
    };
    cudnn_fmha_backend_config["bmm2_grad_gemm2_dot_dimension_numbers"] = {
        {"lhs_contracting_dimensions", {"3"}},
        {"rhs_contracting_dimensions", {"3"}},
        {"lhs_batch_dimensions", {"0", "1"}},
        {"rhs_batch_dimensions", {"0", "1"}},
    };
  }
  json backend_config = {
      {"operation_queue_id", "0"},
      {"wait_on_operation_queues", json::array()},
      {"cudnn_fmha_backend_config", cudnn_fmha_backend_config},
  };
  return backend_config.dump();
}

}  // namespace torch_xla