#pragma once
#include "fbm/core/IEvolver.h"

namespace fbm::core {
class GBM_Euler final : public IEvolver {
public:
  GBM_Euler(double mu, double sigma);

  void evolve(std::span<const double> time,
              std::size_t m_paths,
              std::span<const double> dB,
              std::span<const double> dW,
              std::span<const double> BH,
              double S0,
              std::span<double> S_out) const override;

private:
  double mu_;
  double sigma_;
};
} // namespace fbm::core
