#include <catch2/catch_amalgamated.hpp>
#include <fbm/core/VolterraKernelPowerLaw.h>
#include <fbm/core/VolterraNoise.h>
#include <fbm/core/RoughBergomiFactor.h>
#include <vector>
#include <cmath>
#include <numeric>

TEST_CASE("CLI alignment: RoughBergomiFactor expects BH levels (not increments)", "[cli_align]") {
  const std::size_t m = 20000, N = 256;
  const double T = 1.0, H = 0.5, xi0 = 0.04, eta = 1.5, rho = 0.0;
  const double dt = T / double(N);
  const std::uint64_t seed = 7;

  // Time grid
  std::vector<double> time(N + 1);
  for (std::size_t i = 0; i <= N; ++i) time[i] = dt * double(i);

  // Kernel + noise
  fbm::core::VolterraKernelPowerLaw K;
  std::vector<double> Kmat(N * N);
  K.build(time, H, 16, Kmat);
  fbm::core::VolterraNoise noise(Kmat, N, rho, false);

  std::vector<double> dB(m * N), dW(m * N), BH(m * N);
  noise.sample(dB, dW, BH, m, N, dt, seed);

  fbm::core::RoughBergomiFactor f;
  std::vector<double> Xi(m * N);

  // SECTION A: pass BH (levels) -> mean(Xi_t) ≈ xi0
  REQUIRE_NOTHROW(f.compute(std::span<const double>(BH),
                            std::span<const double>(time),
                            m, N, H, xi0, eta,
                            std::span<double>(Xi)));

  for (std::size_t idx : {N/4, N/2, N-1}) {
    double mean = 0.0, ss = 0.0;
    for (std::size_t p = 0; p < m; ++p) {
      const double v = Xi[p * N + idx];
      mean += v; ss += v * v;
    }
    mean /= double(m);
    const double var = (ss - double(m) * mean * mean) / double(m - 1);
    const double sd  = std::sqrt(var);
    const double se  = sd / std::sqrt(double(m));
    REQUIRE(std::abs(mean - xi0) <= 4.0 * se); // tight check
  }

  // SECTION B: misuse — pass dBH (increments) as if they were BH (levels)
  // This should not satisfy E[Xi_t] ≈ xi0, hence the (tight) check fails.
  std::vector<double> Xi_wrong(m * N);
  REQUIRE_NOTHROW(f.compute(std::span<const double>(dB), // WRONG on purpose
                            std::span<const double>(time),
                            m, N, H, xi0, eta,
                            std::span<double>(Xi_wrong)));

  // Expect a clear mismatch at final time (typical when increments are misused).
  {
    const std::size_t idx = N - 1;
    double mean = 0.0, ss = 0.0;
    for (std::size_t p = 0; p < m; ++p) {
      const double v = Xi_wrong[p * N + idx];
      mean += v; ss += v * v;
    }
    mean /= double(m);
    const double var = (ss - double(m) * mean * mean) / double(m - 1);
    const double sd  = std::sqrt(var);
    const double se  = sd / std::sqrt(double(m));

    // Assert the mean is NOT close to xi0 under the same tight band.
    REQUIRE_FALSE(std::abs(mean - xi0) <= 4.0 * se);
  }
}