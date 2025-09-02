#pragma once
#include <span>
#include <cstddef>

namespace fbm::core {
struct IEvolver {
  virtual ~IEvolver() = default;
  // time: size N+1; S_out: [m_paths * (N+1)]
  virtual void evolve(std::span<const double> time,
                      std::size_t m_paths,
                      std::span<const double> dB,
                      std::span<const double> dW,
                      std::span<const double> BH,
                      double S0,
                      std::span<double> S_out) const = 0;
};
} // namespace fbm::core
