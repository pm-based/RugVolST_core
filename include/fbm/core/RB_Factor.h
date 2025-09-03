#pragma once

#include <span>
#include <cstddef>
#include <stdexcept>

namespace fbm::core {

    // RB_Factor
    // ---------
    // Computes Rough Bergomi variance factor xi_t on a uniform grid using fractional Brownian motion BH.
    //
    // The variance factor follows:
    // xi_t[i] = xi0 * exp( eta * BH[:,i] - 0.5 * eta^2 * t[i+1]^{2H} )
    //
    // This represents the variance process in the Rough Bergomi model where:
    // - xi0 is the initial variance level
    // - eta is the volatility of volatility parameter
    // - BH is fractional Brownian motion with Hurst parameter H
    // - The drift adjustment ensures E[xi_t] = xi0
    class RB_Factor {
    public:
        // Compute variance factor xi_t on a uniform grid using BH
        //
        // Inputs:
        //   BH:  (m x N) row-major fractional Brownian motion increments
        //   t:   (N+1) uniform time grid with t[0] = 0
        //   m:   number of paths
        //   N:   number of time steps
        //   H:   Hurst parameter in (0,1)
        //   xi0: initial variance level > 0
        //   eta: volatility of volatility >= 0
        //
        // Output:
        //   XI:  (m x N) row-major variance factors, i = 0..N-1 corresponds to t[1..N]
        void compute(std::span<const double> BH,
                     std::span<const double> t,
                     std::size_t m,
                     std::size_t N,
                     double H,
                     double xi0,
                     double eta,
                     std::span<double> XI) const;
    };

} // namespace fbm::core
