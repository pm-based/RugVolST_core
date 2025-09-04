#pragma once

#include <span>
#include <cstddef>

namespace fbm::core {

    /// RoughBergomiAssetEuler
    /// ----------------------
    /// Log-Euler evolution of the asset under rough Bergomi.
    /// Inputs:
    /// - time: size N+1, uniform grid (time[0] = 0)
    /// - m: number of paths
    /// - dB: size m*N, Brownian increments (unused here; kept for interface symmetry)
    /// - dW: size m*N, Brownian increments with Var=dt
    /// - BH: size m*N, fractional Brownian levels (unused here; kept for symmetry)
    /// - Xi: size m*N, variance factors for steps t_1..t_N
    /// - dt: step size
    /// - S0: initial spot
    /// Output:
    /// - S_out: size m*(N+1), path-major (S_out[p*(N+1)+i])
    class RoughBergomiAssetEuler {
    public:
        void evolve(std::span<const double> time,
                    std::size_t m,
                    std::span<const double> dB,
                    std::span<const double> dW,
                    std::span<const double> BH,
                    std::span<const double> Xi,
                    double dt,
                    double S0,
                    std::span<double> S_out) const noexcept;
    };

} // namespace fbm::core