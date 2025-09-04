#include <catch2/catch_amalgamated.hpp>
#include <fbm/core/VolterraKernelPowerLaw.h>
#include <fbm/core/VolterraNoise.h>
#include <fbm/core/RoughBergomiFactor.h>
#include <fbm/core/RoughBergomiAssetEuler.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

TEST_CASE("Rough Bergomi Asset Evolution", "[rb_asset]") {
    // Test parameters as specified in the prompt
    const std::size_t m = 5000;
    const std::size_t N = 128;
    const double T = 1.0;
    const double H = 0.5;
    const double eta = 1.5;
    const double xi0 = 0.04;
    const double S0 = 1.0;
    const std::uint64_t seed = 7;

    const double dt = T / static_cast<double>(N);

    // Build uniform time grid
    std::vector<double> time(N + 1);
    for (std::size_t i = 0; i <= N; ++i) {
        time[i] = dt * static_cast<double>(i);
    }

    // Build kernel
    fbm::core::VolterraKernelPowerLaw kernel;
    std::vector<double> K_small(N * N, 0.0);
    kernel.build(time, H, 16, K_small); // Using 16 quad points for test

    // Generate Volterra noise
    fbm::core::VolterraNoise noise_gen(std::vector<double>(K_small), N, 0.0, false);
    std::vector<double> dB(m * N), dW(m * N), BH(m * N);
    noise_gen.sample(dB, dW, BH, m, N, dt, seed);

    // Compute variance factors
    fbm::core::RoughBergomiFactor rb_factor;
    std::vector<double> XI(m * N);

    // Create constant xi0 curve for all time steps
    std::vector<double> xi0t(N, xi0);

    rb_factor.compute(std::span<const double>(BH), std::span<const double>(time),
                      m, N, H, std::span<const double>(xi0t), eta, std::span<double>(XI));

    // Evolve asset paths
    std::vector<double> S_out(m * (N + 1));
    fbm::core::evolve_rb_asset(std::span<const double>(XI), std::span<const double>(dW),
                               m, N, dt, S0, std::span<double>(S_out));

    // Test assertions

    // 1. All S_out are finite and > 0
    for (std::size_t i = 0; i < S_out.size(); ++i) {
        REQUIRE(std::isfinite(S_out[i]));
        REQUIRE(S_out[i] > 0.0);
    }

    // 2. Compute sample mean of log returns
    std::vector<double> log_returns(m);
    for (std::size_t p = 0; p < m; ++p) {
        const double ST = S_out[p * (N + 1) + N];
        log_returns[p] = std::log(ST / S0);
    }

    const double sample_mean = std::accumulate(log_returns.begin(), log_returns.end(), 0.0) / static_cast<double>(m);
    const double theoretical_mean = -0.5 * xi0 * T; // = -0.5 * 0.04 * 1.0 = -0.02
    const double abs_error = std::abs(sample_mean - theoretical_mean);

    // 3. Check that sample mean is close to theoretical reference within tolerance
    REQUIRE(abs_error <= 0.01);

    // Additional checks for robustness
    INFO("Sample mean: " << sample_mean);
    INFO("Theoretical mean: " << theoretical_mean);
    INFO("Absolute error: " << abs_error);
}
