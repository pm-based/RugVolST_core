#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>

#include <fbm/core/INoise.h>
#include <fbm/core/IEvolver.h>
#include <fbm/core/ISimulator.h>
#include <fbm/core/Simulator.h>
#include <fbm/core/GBMEuler.h>
#include <fbm/core/BrownianNoise.h>

namespace fbm::core {

Simulator::Simulator(std::shared_ptr<INoise> noise,
                     std::shared_ptr<IEvolver> evolver,
                     std::size_t chunk_paths)
    : noise_(std::move(noise)), evolver_(std::move(evolver)), chunk_(chunk_paths) {
  if (!noise_ || !evolver_) {
    throw std::invalid_argument("Noise and evolver must not be null");
  }
}

void Simulator::simulate(std::span<const double> time,
                         std::size_t n_paths,
                         std::uint64_t seed,
                         double S0,
                         std::span<double> S_out) const {
  // Validate time grid
  if (time.size() < 2) {
    throw std::invalid_argument("time grid must have at least 2 points");
  }

  const std::size_t N = time.size() - 1;
  const double dt = time[1] - time[0];
  constexpr double rtol = 1e-12;

  // Check if time starts near 0
  if (std::abs(time[0]) > rtol) {
    throw std::invalid_argument("time grid must start at t=0");
  }

  // Check uniform spacing
  for (std::size_t i = 1; i < N; ++i) {
    const double dt_i = time[i + 1] - time[i];
    if (std::abs(dt_i - dt) > rtol * std::max(dt, dt_i)) {
      throw std::invalid_argument("time grid must be uniform");
    }
  }

  // Validate output size
  if (S_out.size() != n_paths * (N + 1)) {
    throw std::invalid_argument("S_out size must be n_paths * (N + 1)");
  }

  // Process in chunks
  std::uint64_t current_seed = seed;
  for (std::size_t start = 0; start < n_paths; start += chunk_) {
    const std::size_t m_paths = std::min(chunk_, n_paths - start);

    // Allocate buffers for this chunk
    std::vector<double> dB(m_paths * N);
    std::vector<double> dW(m_paths * N);
    std::vector<double> BH; // Empty for standard Brownian motion

    // Generate noise
    noise_->sample(dB, dW, BH, m_paths, N, dt, current_seed);

    // Evolve paths
    auto S_chunk = S_out.subspan(start * (N + 1), m_paths * (N + 1));
    evolver_->evolve(time, m_paths, dB, dW, BH, S0, S_chunk);

    // Update seed for next chunk
    current_seed += m_paths * N;
  }
}

} // namespace fbm::core
