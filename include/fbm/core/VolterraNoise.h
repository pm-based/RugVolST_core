#pragma once

#include <span>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <fbm/core/INoise.h>

namespace fbm::core {

    // VolterraNoise
    // ------------------
    // Generate Brownian increments dB,dW ~ N(0, dt) and, if requested,
    // compute fractional-noise samples BH = dB @ K_small where K_small is
    // a row-major lower-triangular (N x N) Volterra kernel discretization.
    //
    // When rho != 0, dW is correlated with dB: dW = rho * dB + sqrt(1-rho^2) * W_perp
    // where W_perp is independent of dB. This correlation is with respect to the
    // base Brownian dB, not the fractional Brownian motion BH.
    //
    // Shapes (row-major):
    //   dB, dW, BH : [m_paths * N]
    //   K_small    : [N * N] captured at construction
    //
    // Notes:
    //  - If BH.size() == 0, the method MUST NOT touch BH and returns after
    //    filling dB and dW.
    //  - If BH.size() > 0, it must equal m_paths * N.
    //  - The multiply uses a portable blocked fallback; if FBM_USE_BLAS is
    //    defined at build time, a cblas dgemm path is used instead.
    class VolterraNoise final : public INoise {
    public:
        VolterraNoise(std::vector<double> K_small, std::size_t N, double rho = 0.0, bool use_antithetic = false)
            : K_small_(std::move(K_small)), N_(N), rho_(rho), use_antithetic_(use_antithetic) {
            if (N_ == 0) throw std::invalid_argument("N must be > 0");
            if (K_small_.size() != N_ * N_) throw std::invalid_argument("K_small size must be N*N");
            if (rho_ <= -1.0 || rho_ >= 1.0) throw std::invalid_argument("rho must be in (-1, 1)");
        }

        void sample(std::span<double> dB,
                    std::span<double> dW,
                    std::span<double> BH,
                    std::size_t m_paths,
                    std::size_t N,
                    double dt,
                    std::uint64_t seed) const override;

    private:
        void gemm_fallback(std::span<const double> dB,
                           std::span<double> BH,
                           std::size_t m,
                           std::size_t N) const;

    private:
        std::vector<double> K_small_; // row-major N x N, lower-triangular used
        std::size_t N_{};
        double rho_{0.0};
        bool use_antithetic_{false};
    };

} // namespace fbm::core