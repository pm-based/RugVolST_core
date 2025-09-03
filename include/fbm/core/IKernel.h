#pragma once
#include <span>
#include <cstddef>
#include <stdexcept>

namespace fbm::core {
struct IKernel {
  virtual ~IKernel() = default;
  // Build K_small (N×N, row-major) for a uniform time grid of size N+1 (t[0]≈0).
  // Throws std::invalid_argument on bad sizes or non-uniform grid; requires 0 < H < 1.
  virtual void build(std::span<const double> time,
                     double H,
                     std::size_t quad_points,
                     std::span<double> K_small) const = 0;
};
} // namespace fbm::core
