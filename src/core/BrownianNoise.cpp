#include <random>
#include <stdexcept>
#include <cmath>

#include <fbm/core/INoise.h>
#include <fbm/core/IEvolver.h>
#include <fbm/core/ISimulator.h>
#include <fbm/core/Simulator.h>
#include <fbm/core/GBM_Euler.h>
#include <fbm/core/BrownianNoise.h>

namespace fbm::core {

void BrownianNoise::sample(std::span<double> dB,
                           std::span<double> dW,
                           std::span<double> BH,
                           std::size_t m_paths,
                           std::size_t N,
                           double dt,
                           std::uint64_t seed) const {
  // Validate input sizes
  if (dB.size() != m_paths * N || dW.size() != m_paths * N) {
    throw std::invalid_argument("dB and dW must have size m_paths * N");
  }

  if (!BH.empty() && BH.size() != m_paths * N) {
    throw std::invalid_argument("BH must be empty or have size m_paths * N");
  }

  // Initialize random number generator
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  const double sqrt_dt = std::sqrt(dt);

  // Generate Brownian increments
  for (std::size_t i = 0; i < m_paths * N; ++i) {
    const double z1 = normal(rng);
    const double z2 = normal(rng);

    dB[i] = z1 * sqrt_dt;
    dW[i] = z2 * sqrt_dt;
  }

  // BH is not used for standard Brownian motion, leave empty
  // If BH is provided but not empty, we don't fill it since this is just Brownian noise
}

} // namespace fbm::core
