#include <fbm/core/VolterraNoise.h>

#include <algorithm>
#include <cmath>
#include <random>

#if defined(FBM_USE_BLAS)
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#if defined(FBM_USE_OPENMP)
#include <omp.h>
#endif

namespace fbm::core {

void VolterraNoise::sample(std::span<double> dB,
                               std::span<double> dW,
                               std::span<double> BH,
                               std::size_t m_paths,
                               std::size_t N,
                               double dt,
                               std::uint64_t seed) const {
  if (N != N_) throw std::invalid_argument("N mismatch with kernel size");
  if (dB.size() != m_paths * N || dW.size() != m_paths * N)
    throw std::invalid_argument("dB/dW size mismatch");
  if (!BH.empty() && BH.size() != m_paths * N)
    throw std::invalid_argument("BH size mismatch");
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");

  // Generate Brownian increments with variance dt
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> N01(0.0, 1.0);
  const double sdt = std::sqrt(dt);
  const double sqrt_one_minus_rho2 = std::sqrt(1.0 - rho_ * rho_);

  if (use_antithetic_ && m_paths % 2 == 0) {
    // Antithetic variates: generate first half, negate for second half
    const std::size_t m2 = m_paths / 2;

    // Generate first half of paths
    for (std::size_t i = 0; i < m2 * N; ++i) {
      dB[i] = N01(rng) * sdt;
      // Generate independent W_perp first, then apply correlation
      const double W_perp = N01(rng) * sdt;
      dW[i] = rho_ * dB[i] + sqrt_one_minus_rho2 * W_perp;
    }

    // Set second half as antithetic (negated)
    for (std::size_t p = 0; p < m2; ++p) {
      for (std::size_t i = 0; i < N; ++i) {
        dB[(m2 + p) * N + i] = -dB[p * N + i];
        dW[(m2 + p) * N + i] = -dW[p * N + i];
      }
    }
  } else {
    // Standard sampling (or fallback for odd m_paths)
    for (std::size_t i = 0; i < m_paths * N; ++i) {
      dB[i] = N01(rng) * sdt;
      // Generate independent W_perp first, then apply correlation
      const double W_perp = N01(rng) * sdt;
      dW[i] = rho_ * dB[i] + sqrt_one_minus_rho2 * W_perp;
    }
  }

  // If BH not requested, stop here
  if (BH.empty()) return;

#if defined(FBM_USE_BLAS)
  // Row-major: (m x N) * (N x N) = (m x N)
  const int M = static_cast<int>(m_paths);
  const int K = static_cast<int>(N);
  const int Nn = static_cast<int>(N);
  const double alpha = 1.0, beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, Nn, K,
              alpha,
              dB.data(), Nn,
              K_small_.data(), Nn,
              beta,
              BH.data(), Nn);
#else
  gemm_fallback(dB, BH, m_paths, N);
#endif
}

void VolterraNoise::gemm_fallback(std::span<const double> dB,
                                      std::span<double> BH,
                                      std::size_t m,
                                      std::size_t N) const {
  std::fill(BH.begin(), BH.end(), 0.0);

#if defined(FBM_USE_OPENMP)
  // Simple OpenMP parallelization over paths
  #pragma omp parallel for schedule(static)
  for (std::size_t p = 0; p < m; ++p) {
    for (std::size_t i = 0; i < N; ++i) {
      double acc = 0.0;
      for (std::size_t j = 0; j <= i; ++j) {
        acc += dB[p * N + j] * K_small_[j * N + i];
      }
      BH[p * N + i] = acc;
    }
  }
#else
  // Blocked triple loop for cache locality (non-OpenMP path)
  const std::size_t BS = 64;

  for (std::size_t p0 = 0; p0 < m; p0 += BS) {
    const std::size_t p1 = std::min(p0 + BS, m);
    for (std::size_t i0 = 0; i0 < N; i0 += BS) {
      const std::size_t i1 = std::min(i0 + BS, N);
      for (std::size_t j0 = 0; j0 < N; j0 += BS) {
        const std::size_t j1 = std::min(j0 + BS, N);
        for (std::size_t p = p0; p < p1; ++p) {
          for (std::size_t i = i0; i < i1; ++i) {
            const std::size_t jmax = std::min(i + 1, j1); // lower-triangular
            for (std::size_t j = j0; j < jmax; ++j) {
              BH[p * N + i] += dB[p * N + j] * K_small_[j * N + i];
            }
          }
        }
      }
    }
  }
#endif
}

} // namespace fbm::core