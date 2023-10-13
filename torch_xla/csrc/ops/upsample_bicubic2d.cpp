#include "torch_xla/csrc/ops/upsample_bicubic2d.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"
#include "xla/util.h"

namespace torch_xla {

UpsampleBicubic::UpsampleBicubic(const torch::lazy::Value& input,
                                   std::vector<int64_t> output_size,
                                   bool align_corners)
    : XlaNode(torch::lazy::OpKind(at::aten::upsample_bicubic2d), {input},
              [&]() {
                return resize::GetForwardOutputShape2d(GetXlaShape(input),
                                                       output_size);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(output_size, align_corners)),
      output_size_(std::move(output_size)),
      align_corners_(align_corners) {}

torch::lazy::NodePtr UpsampleBicubic::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<UpsampleBicubic>(operands.at(0), output_size_,
                                                 align_corners_);
}

XlaOpVector UpsampleBicubic::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = resize::LowerForward2d(
      "ResizeBicubic", input, xla_shape(), align_corners_,
      /*half_pixel_centers=*/!align_corners_);
  return ReturnOp(output, loctx);
}

std::string UpsampleBicubic::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ")
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace torch_xla
