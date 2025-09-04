#include <fbm/core/RoughBergomiFactor.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fbm::core {

namespace {

inline void validate_time_uniform(std::span<const double> time) {
  if (time.size() < 2) throw std::invalid_argument("time grid must have at least 2 points");
  if (std::abs(time[0]) > 1e-14) throw std::invalid_argument("time[0] must be approximately 0");
  const std::size_t N = time.size() - 1;
  const double dt = time[1] - time[0];
  if (!(dt > 0.0)) throw std::invalid_argument("time grid must be strictly increasing");
  const double rtol = 1e-12;
  for (std::size_t i = 1; i < N; ++i) {
    const double dti = time[i + 1] - time[i];
    if (std::abs(dti - dt) > rtol * std::max(dt, dti)) {
      throw std::invalid_argument("time grid must be uniform");
    }
  }
}

inline void validate_common(std::span<const double> BH_levels,
                            std::span<const double> time,
                            std::size_t m_paths, std::size_t N,
                            double H, double eta,
                            std::span<double> XI) {
  if (m_paths == 0) throw std::invalid_argument("m_paths must be > 0");
  if (N == 0) throw std::invalid_argument("N must be > 0");
  if (BH_levels.size() != m_paths * N) throw std::invalid_argument("BH size must be m_paths*N");
  if (time.size() != N + 1) throw std::invalid_argument("time size must be N+1");
  if (XI.size() != m_paths * N) throw std::invalid_argument("XI size must be m_paths*N");
  if (!(H > 0.0 && H < 1.0)) throw std::invalid_argument("H must be in (0,1)");
  if (eta < 0.0) throw std::invalid_argument("eta must be >= 0");
  validate_time_uniform(time);
}

} // namespace

// xi0 = curva nel tempo xi0(t_i)
void RoughBergomiFactor::compute(std::span<const double> BH_levels,
                                 std::span<const double> time,
                                 std::size_t m_paths, std::size_t N,
                                 double H, std::span<const double> xi0t, double eta,
                                 std::span<double> XI) const {
  validate_common(BH_levels, time, m_paths, N, H, eta, XI);

  // Accetta xi0t di lunghezza N (t1..tN) o N+1 (t0..tN)
  if (xi0t.size() != N && xi0t.size() != N + 1)
    throw std::invalid_argument("xi0t size must be N (t1..tN) or N+1 (t0..tN)");
  for (double v : xi0t) {
    if (v < 0.0) throw std::invalid_argument("xi0t must be non-negative");
  }

  const bool has_t0 = (xi0t.size() == N + 1);

  // Xi(p,i) = xi0(t_{i+1}) * exp( eta*BH_{t_{i+1}} - 0.5*eta^2*t_{i+1}^{2H} )
  for (std::size_t i = 0; i < N; ++i) {
    const double t = time[i + 1];
    const double drift = -0.5 * eta * eta * std::pow(t, 2.0 * H);
    const double xi0_val = has_t0 ? xi0t[i + 1] : xi0t[i];

    for (std::size_t p = 0; p < m_paths; ++p) {
      const double BH = BH_levels[p * N + i]; // livello B^H_t
      XI[p * N + i] = xi0_val * std::exp(eta * BH + drift);
    }
  }
}

// xi0 = costante
void RoughBergomiFactor::compute(std::span<const double> BH_levels,
                                 std::span<const double> time,
                                 std::size_t m_paths, std::size_t N,
                                 double H, double xi0, double eta,
                                 std::span<double> XI) const {
  if (xi0 < 0.0) throw std::invalid_argument("xi0 must be non-negative");
  validate_common(BH_levels, time, m_paths, N, H, eta, XI);

  // Xi(p,i) = xi0 * exp( eta*BH_{t_{i+1}} - 0.5*eta^2*t_{i+1}^{2H} )
  for (std::size_t i = 0; i < N; ++i) {
    const double t = time[i + 1];
    const double drift = -0.5 * eta * eta * std::pow(t, 2.0 * H);

    for (std::size_t p = 0; p < m_paths; ++p) {
      const double BH = BH_levels[p * N + i]; // livello B^H_t
      XI[p * N + i] = xi0 * std::exp(eta * BH + drift);
    }
  }
}

} // namespace fbm::core