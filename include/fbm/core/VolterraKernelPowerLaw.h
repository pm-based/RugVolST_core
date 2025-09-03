#pragma once
#include "fbm/core/IKernel.h"

namespace fbm::core {

// VolterraKernelPowerLaw
// ---------------------
// Implements a power-law Volterra kernel for fractional Brownian motion approximation
// valid for all H ∈ (0,1). Uses stationary weights and normalization to ensure
// Var[BH_t] ≈ t^{2H} on uniform grids.
//
// For H=0.5, reproduces the triangular-ones matrix (backward compatibility).
// For H≠0.5, uses power-law weights: w_k = (k+1)^(H-0.5) - k^(H-0.5)
// with normalization to match theoretical variance.
class VolterraKernelPowerLaw final : public IKernel {
public:
  void build(std::span<const double> time,
             double H,
             std::size_t quad_points,
             std::span<double> K_small) const override;
};

} // namespace fbm::core
