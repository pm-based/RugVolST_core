#pragma once
#include <span>
#include <cstddef>
#include <cstdint>

namespace fbm::core {
struct INoise {
  virtual ~INoise() = default;
  // dB,dW: [m_paths * N]; BH optional (may be empty).
  virtual void sample(std::span<double> dB,
                      std::span<double> dW,
                      std::span<double> BH,
                      std::size_t m_paths,
                      std::size_t N,
                      double dt,
                      std::uint64_t seed) const = 0;
};
} // namespace fbm::core
