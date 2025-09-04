#include "fbm/core/RoughBergomiAssetEuler.h"
#include <cmath>
#include <algorithm>

namespace fbm::core {

void evolve_rb_asset(std::span<const double> XI,
                     std::span<const double> dW,
                     std::size_t m,
                     std::size_t N,
                     double dt,
                     double S0,
                     std::span<double> S_out) noexcept {

    for (std::size_t p = 0; p < m; ++p) {
        // Initialize log price and set initial asset price
        double logS = std::log(S0);
        S_out[p * (N + 1) + 0] = S0;

        // Evolve asset price using log-Euler
        for (std::size_t i = 0; i < N; ++i) {
            const double xi = XI[p * N + i];
            const double inc = -0.5 * xi * dt + std::sqrt(std::max(xi, 0.0)) * dW[p * N + i];
            // Note: dW already has variance dt, so no extra sqrt(dt)
            logS += inc;
            S_out[p * (N + 1) + i + 1] = std::exp(logS);
        }
    }
}

} // namespace fbm::core
