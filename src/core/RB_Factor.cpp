#include <fbm/core/RB_Factor.h>
#include <cmath>
#include <algorithm>

namespace fbm::core {

void RB_Factor::compute(std::span<const double> BH,
                        std::span<const double> t,
                        std::size_t m,
                        std::size_t N,
                        double H,
                        double xi0,
                        double eta,
                        std::span<double> XI) const {

    // Validate inputs
    if (BH.size() != m * N) {
        throw std::invalid_argument("BH size must be m * N");
    }
    if (t.size() != N + 1) {
        throw std::invalid_argument("t size must be N + 1");
    }
    if (XI.size() != m * N) {
        throw std::invalid_argument("XI size must be m * N");
    }
    if (H <= 0.0 || H >= 1.0) {
        throw std::invalid_argument("H must be in (0, 1)");
    }
    if (xi0 <= 0.0) {
        throw std::invalid_argument("xi0 must be > 0");
    }
    if (eta < 0.0) {
        throw std::invalid_argument("eta must be >= 0");
    }
    if (N == 0) {
        throw std::invalid_argument("N must be > 0");
    }

    // Check that t is uniform and positive
    if (t[0] != 0.0) {
        throw std::invalid_argument("t[0] must be 0");
    }
    for (std::size_t i = 1; i <= N; ++i) {
        if (t[i] <= t[i-1]) {
            throw std::invalid_argument("t must be strictly increasing");
        }
    }

    // Precompute drift terms: -0.5 * eta^2 * t[i+1]^{2H}
    std::vector<double> drift(N);
    const double eta_sq_half = 0.5 * eta * eta;
    for (std::size_t i = 0; i < N; ++i) {
        const double t_val = t[i + 1];
        drift[i] = -eta_sq_half * std::pow(t_val, 2.0 * H);
    }

    // Compute variance factors for each path and time step
    for (std::size_t p = 0; p < m; ++p) {
        for (std::size_t i = 0; i < N; ++i) {
            const double bh_val = BH[p * N + i];
            const double exponent = eta * bh_val + drift[i];
            XI[p * N + i] = xi0 * std::exp(exponent);
        }
    }
}

} // namespace fbm::core
