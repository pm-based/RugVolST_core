#include "fbm/core/RoughBergomiAssetEuler.h"
#include <cmath>
#include <algorithm>

namespace fbm::core {

    void RoughBergomiAssetEuler::evolve(std::span<const double> time,
                                        std::size_t m,
                                        std::span<const double> dB,
                                        std::span<const double> dW,
                                        std::span<const double> BH,
                                        std::span<const double> Xi,
                                        double dt,
                                        double S0,
                                        std::span<double> S_out) const noexcept {
        const std::size_t N = time.size() > 0 ? (time.size() - 1) : 0;
        (void)dB; // unused by design
        (void)BH; // unused by design

        for (std::size_t p = 0; p < m; ++p) {
            double logS = std::log(S0);
            S_out[p * (N + 1) + 0] = S0;

            for (std::size_t i = 0; i < N; ++i) {
                const double xi  = Xi[p * N + i];
                const double vol = std::sqrt(std::max(xi, 0.0));
                // dW already ~ N(0, dt)
                logS += (-0.5 * xi * dt) + vol * dW[p * N + i];
                S_out[p * (N + 1) + (i + 1)] = std::exp(logS);
            }
        }
    }

} // namespace fbm::core