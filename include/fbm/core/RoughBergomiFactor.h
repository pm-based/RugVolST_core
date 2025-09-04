#pragma once

#include <span>
#include <cstddef>

namespace fbm::core {

    /// RoughBergomiFactor
    /// ------------------
    /// Computes variance factors Xi on a uniform time grid for the rough Bergomi model.
    ///
    /// Convention:
    /// - Input BH is the **level** of fractional Brownian motion at t_1..t_N (path-major, size m*N).
    /// - If a term-structure xi0(t) is provided, its size can be N   (values at t_1..t_N)
    ///   or N+1 (values at t_0..t_N; the value at t_0 is ignored).
    ///
    /// Formula at times t_{i+1}, i = 0..N-1:
    ///   Xi[i] = xi0(t_{i+1}) * exp( eta * BH[i] - 0.5 * eta^2 * t_{i+1}^{2H} )
    ///
    /// where H ∈ (0,1), eta ≥ 0, and E[exp(eta * BH_t - 0.5 * eta^2 * t^{2H})] = 1,
    /// hence E[Xi_t] = xi0(t) by construction.
    class RoughBergomiFactor {
    public:
        /// Term-structure overload: xi0(t) provided on either t_1..t_N (size N)
        /// or t_0..t_N (size N+1; t_0 value is ignored).
        void compute(std::span<const double> BH,
                     std::span<const double> time,
                     std::size_t m_paths, std::size_t N,
                     double H, std::span<const double> xi0t, double eta,
                     std::span<double> Xi) const;

        /// Constant xi0 overload.
        void compute(std::span<const double> BH,
                     std::span<const double> time,
                     std::size_t m_paths, std::size_t N,
                     double H, double xi0, double eta,
                     std::span<double> Xi) const;
    };

} // namespace fbm::core