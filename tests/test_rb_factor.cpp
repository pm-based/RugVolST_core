#define CATCH_CONFIG_MAIN
#include <catch2/catch_amalgamated.hpp>

#include <fbm/core/RoughBergomiFactor.h>
#include <fbm/core/VolterraKernelPowerLaw.h>
#include <fbm/core/VolterraNoise.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

TEST_CASE("RB_Factor - Rough Bergomi variance factor generation", "[rb_factor]") {
    // Test parameters following the specification
    const std::size_t m = 20000;  // Number of paths
    const std::size_t N = 256;    // Number of time steps
    const double T = 1.0;         // Time horizon
    const double H = 0.5;         // Hurst parameter
    const double eta = 1.5;       // Volatility of volatility
    const std::uint64_t seed = 7; // Random seed
    const double rho = 0.0;       // Correlation

    // xi0(t) curve parameters (constant case)
    const double beta0 = 0.04, beta1 = 0.0, beta2 = 0.0, tau1 = 1.0, tau2 = 1.0;

    // Build uniform time grid
    std::vector<double> time(N + 1);
    const double dt = T / static_cast<double>(N);
    for (std::size_t i = 0; i <= N; ++i) {
        time[i] = dt * static_cast<double>(i);
    }

    // Build power-law kernel
    fbm::core::VolterraKernelPowerLaw kernel;
    std::vector<double> K_small(N * N, 0.0);
    kernel.build(time, H, 16, K_small);

    // Generate Volterra noise
    fbm::core::VolterraNoise noise_gen(std::vector<double>(K_small), N, rho, false);
    std::vector<double> dB(m * N), dW(m * N), BH(m * N);
    noise_gen.sample(dB, dW, BH, m, N, dt, seed);

    // Build xi0(t) curve
    std::vector<double> xi0t(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double t = time[i + 1];
        xi0t[i] = beta0 + beta1 * std::exp(-t / tau1) + beta2 * (t / tau2) * std::exp(-t / tau2);
    }

    // Compute Rough Bergomi variance factors
    fbm::core::RoughBergomiFactor rb_factor;
    std::vector<double> XI(m * N);

    REQUIRE_NOTHROW(rb_factor.compute(
        std::span<const double>(BH),
        std::span<const double>(time),
        m, N, H, std::span<const double>(xi0t), eta,
        std::span<double>(XI)
    ));

    // Test at three time indices as specified
    std::vector<std::size_t> test_indices{N / 4, N / 2, N - 1};

    for (std::size_t idx : test_indices) {
        // Compute sample mean
        double sum = 0.0;
        for (std::size_t p = 0; p < m; ++p) {
            sum += XI[p * N + idx];
        }
        const double sample_mean = sum / static_cast<double>(m);

        // Compute sample standard deviation
        double sum_sq = 0.0;
        for (std::size_t p = 0; p < m; ++p) {
            const double dev = XI[p * N + idx] - sample_mean;
            sum_sq += dev * dev;
        }
        const double sample_var = sum_sq / static_cast<double>(m - 1);
        const double sample_sd = std::sqrt(sample_var);

        // Guard against zero standard deviation
        REQUIRE(sample_sd > 0.0);

        // Standard error of the mean
        const double SE = sample_sd / std::sqrt(static_cast<double>(m));

        // Theory: E[xi_t] = xi0(t) due to drift adjustment
        const double theory = xi0t[idx]; // Use the curve value instead of constant
        const double error = std::abs(sample_mean - theory);

        // Test: |sample_mean - xi0(t)| <= 3 * SE (99.7% confidence)
        REQUIRE(error <= 3.0 * SE);

        // Additional sanity checks
        REQUIRE(sample_mean > 0.0);  // Variance factors should be positive
        REQUIRE(std::isfinite(sample_mean));
        REQUIRE(std::isfinite(sample_sd));
    }

    // Test edge cases for the compute function
    SECTION("Input validation") {
        std::vector<double> small_BH(4), small_t(3), small_XI(4), small_xi0t(2);
        small_t = {0.0, 0.5, 1.0};
        small_xi0t = {0.04, 0.04};

        // Test invalid H values
        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(small_BH),
            std::span<const double>(small_t),
            2, 2, 0.0, std::span<const double>(small_xi0t), eta,
            std::span<double>(small_XI)
        ), std::invalid_argument);

        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(small_BH),
            std::span<const double>(small_t),
            2, 2, 1.0, std::span<const double>(small_xi0t), eta,
            std::span<double>(small_XI)
        ), std::invalid_argument);

        // Test invalid eta
        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(small_BH),
            std::span<const double>(small_t),
            2, 2, H, std::span<const double>(small_xi0t), -1.0,
            std::span<double>(small_XI)
        ), std::invalid_argument);

        // Test size mismatches
        std::vector<double> wrong_size_BH(2);
        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(wrong_size_BH),
            std::span<const double>(small_t),
            2, 2, H, std::span<const double>(small_xi0t), eta,
            std::span<double>(small_XI)
        ), std::invalid_argument);
    }
}
