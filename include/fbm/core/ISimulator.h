#pragma once
#include <span>
#include <cstddef>
#include <cstdint>

namespace fbm::core {
struct ISimulator {
  virtual ~ISimulator() = default;
  virtual void simulate(std::span<const double> time,
                        std::size_t n_paths,
                        std::uint64_t seed,
                        double S0,
                        std::span<double> S_out) const = 0;
};
} // namespace fbm::core
