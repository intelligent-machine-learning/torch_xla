#ifndef XLA_TORCH_XLA_CSRC_COMMON_LYNX_TYPES_H_
#define XLA_TORCH_XLA_CSRC_COMMON_LYNX_TYPES_H_
#include <unordered_map>
#include <utility>

namespace lynx {
class P2PChannelsMap
    : public std::unordered_map<int64_t, std::pair<int64_t, int64_t>> {}

}  // namespace lynx

#endif