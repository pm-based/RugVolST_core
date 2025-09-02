#include <stdexcept>
#include <cmath>

#include <fbm/core/INoise.h>
#include <fbm/core/IEvolver.h>
#include <fbm/core/ISimulator.h>
#include <fbm/core/Simulator.h>
#include <fbm/core/GBM_Euler.h>
#include <fbm/core/BrownianNoise.h>

namespace fbm::core {

GBM_Euler::GBM_Euler(double mu, double sigma) : mu_(mu), sigma_(sigma) {
  if (sigma_ < 0.0) {
    throw std::invalid_argument("Volatility sigma must be non-negative");
  }
}

void GBM_Euler::evolve(std::span<const double> time,
                       std::size_t m_paths,
                       std::span<const double> dB,
                       std::span<const double> dW,
                       std::span<const double> BH,
                       double S0,
                       std::span<double> S_out) const {
  const std::size_t N = time.size() - 1;

  // Validate input sizes
  if (dW.size() != m_paths * N) {
    throw std::invalid_argument("dW must have size m_paths * N");
  }

  if (S_out.size() != m_paths * (N + 1)) {
    throw std::invalid_argument("S_out must have size m_paths * (N + 1)");
  }

  const double dt = time[1] - time[0];
  const double drift = (mu_ - 0.5 * sigma_ * sigma_) * dt;

  // Initialize all paths at t=0
  for (std::size_t path = 0; path < m_paths; ++path) {
    S_out[path * (N + 1)] = S0;
  }

  // Evolve each path using Euler scheme for GBM
  // dS/S = mu*dt + sigma*dW, so dlog(S) = (mu - 0.5*sigma^2)*dt + sigma*dW
  for (std::size_t path = 0; path < m_paths; ++path) {
    double log_S = std::log(S0);

    for (std::size_t step = 0; step < N; ++step) {
      const std::size_t idx = path * N + step;

      // Euler step for log(S)
      log_S += drift + sigma_ * dW[idx];

      // Convert back to S
      S_out[path * (N + 1) + step + 1] = std::exp(log_S);
    }
  }
}

} // namespace fbm::core
