#pragma once
#include <span>
#include <cstddef>

namespace fbm::core {

// Evolve asset paths under rough Bergomi using log-Euler.
// Inputs:
// - XI:  size m*N, variance factors per (path, step)
// - dW:  size m*N, Brownian increments with Var=dt (from VolterraNoiseGEMM)
// - m:   number of paths
// - N:   time steps
// - dt:  step size T/N
// - S0:  initial price
// Output:
// - S_out: size m*(N+1), row-major per path, writes S0 then path.
void evolve_rb_asset(std::span<const double> XI,
                     std::span<const double> dW,
                     std::size_t m,
                     std::size_t N,
                     double dt,
                     double S0,
                     std::span<double> S_out) noexcept;

} // namespace fbm::core
