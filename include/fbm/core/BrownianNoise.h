#pragma once
#include "fbm/core/INoise.h"

namespace fbm::core {
class BrownianNoise final : public INoise {
public:
  BrownianNoise() = default;

  void sample(std::span<double> dB,
              std::span<double> dW,
              std::span<double> BH,
              std::size_t m_paths,
              std::size_t N,
              double dt,
              std::uint64_t seed) const override;
};
} // namespace fbm::core
