#include "fbm/core/VolterraKernelPowerLaw.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace fbm::core {

void VolterraKernelPowerLaw::build(std::span<const double> time,
                                   double H,
                                   std::size_t /*quad_points*/,
                                   std::span<double> K_small) const {
  if (time.size() < 2) throw std::invalid_argument("time grid must have at least 2 points");
  const std::size_t N = time.size() - 1;
  if (K_small.size() != N * N) throw std::invalid_argument("K_small size must be N*N");
  if (H <= 0.0 || H >= 1.0) throw std::invalid_argument("H must be in (0,1)");
  if (std::abs(time[0]) > 1e-14) throw std::invalid_argument("t[0] must be ~0");

  const double dt = time[1] - time[0];
  const double rtol = 1e-12;
  for (std::size_t i = 1; i < N; ++i) {
    const double dti = time[i + 1] - time[i];
    if (std::abs(dti - dt) > rtol * std::max(dt, dti))
      throw std::invalid_argument("time grid must be uniform");
  }

  std::fill(K_small.begin(), K_small.end(), 0.0);

  // Exact Brownian case keeps backward tests green
  if (std::abs(H - 0.5) < 1e-15) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        K_small[j * N + i] = 1.0; // row-major: K(row=j,col=i)
    return;
  }

  // Midpoint discretization of the Volterra kernel
  // w_k = (k+0.5)^(H-0.5)
  std::vector<double> w(N);
  const double p = H - 0.5;
  for (std::size_t k = 0; k < N; ++k)
    w[k] = std::pow(double(k) + 0.5, p);

  // Normalization so that Var[BH_T] = T^(2H) with dB ~ N(0, dt)
  // Var = dt * alpha^2 * sum (k+0.5)^(2H-1) = T^(2H), dt=T/N
  double denom = 0.0;
  const double q = 2.0 * H - 1.0;
  for (std::size_t k = 0; k < N; ++k)
    denom += std::pow(double(k) + 0.5, q);

  const double T = time[N];
  const double alpha = std::pow(T, H - 0.5) * std::sqrt(double(N) / denom);

  // Build lower-triangular K: K[j,i] = alpha * w_{i-j}
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      const std::size_t lag = i - j;
      K_small[j * N + i] = alpha * w[lag];
    }
  }
}

} // namespace fbm::core