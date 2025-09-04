#include "fbm/core/VolterraKernelBrownian.h"
#include <cmath>
#include <algorithm>

namespace fbm::core {

void VolterraKernelStub::build(std::span<const double> time,
                               double H,
                               std::size_t quad_points,
                               std::span<double> K_small) const {

  // Validate input sizes
  if (time.size() < 2) {
    throw std::invalid_argument("time grid must have at least 2 points");
  }

  const std::size_t N = time.size() - 1;
  if (K_small.size() != N * N) {
    throw std::invalid_argument("K_small size must be N*N where N = time.size()-1");
  }

  // Validate H range
  if (H <= 0.0 || H >= 1.0) {
    throw std::invalid_argument("H must be in range (0, 1)");
  }

  // Validate t[0] ≈ 0
  if (std::abs(time[0]) > 1e-14) {
    throw std::invalid_argument("first time point must be approximately 0");
  }

  // Check uniform grid (constant dt within tolerance)
  const double dt = time[1] - time[0];
  const double rtol = 1e-12;
  for (std::size_t i = 1; i < N; ++i) {
    const double current_dt = time[i + 1] - time[i];
    if (std::abs(current_dt - dt) > rtol * std::max(dt, current_dt)) {
      throw std::invalid_argument("time grid must be uniform");
    }
  }

  // Suppress unused parameter warning
  (void)quad_points;

  // Special case: exact Brownian kernel for H = 0.5
  if (std::abs(H - 0.5) <= 1e-12) {
    // Uniform grid: K(t,s) = 1 for s < t
    // Midpoint discretization ⇒ K[j,i] = 1 for 0 <= j <= i, else 0
    // N = time.size() - 1, row-major K[N*N]
    std::fill(K_small.begin(), K_small.end(), 0.0);
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        K_small[j * N + i] = 1.0;
      }
    }
    return;
  }

  // Fill K with zeros except diagonal = dt (placeholder for H != 0.5)
  std::fill(K_small.begin(), K_small.end(), 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    K_small[i * N + i] = dt;  // row-major indexing
  }
}

} // namespace fbm::core