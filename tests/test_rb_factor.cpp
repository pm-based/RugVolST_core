#define CATCH_CONFIG_MAIN
#include <catch2/catch_amalgamated.hpp>

#include <fbm/core/RoughBergomiFactor.h>
#include <fbm/core/VolterraKernelBrownian.h>
#include <fbm/core/VolterraNoise.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

TEST_CASE("RB_Factor - Rough Bergomi variance factor generation", "[rb_factor]") {
    // Test parameters following the specification
    const std::size_t P = 20000;  // Number of paths
    const std::size_t N = 256;    // Number of time steps
    const double T = 1.0;         // Time horizon
    const double H = 0.5;         // Hurst parameter
    const double xi0 = 0.04;      // Initial variance level
    const double eta = 1.5;       // Volatility of volatility
    const std::uint64_t seed = 7; // Random seed
    const double rho = 0.0;       // Correlation

    // Build uniform time grid
    std::vector<double> t(N + 1);
    const double dt = T / static_cast<double>(N);
    for (std::size_t i = 0; i <= N; ++i) {
        t[i] = dt * static_cast<double>(i);
    }

    // Build Volterra kernel (H=0.5)
    fbm::core::VolterraKernelStub kernel;
    std::vector<double> K(N * N, 0.0);
    kernel.build(t, H, 16, K);

    // Generate fractional Brownian motion BH
    fbm::core::VolterraNoiseGEMM noise_gen(std::vector<double>(K), N, rho, false);
    std::vector<double> dB(P * N), dW(P * N), BH(P * N);
    noise_gen.sample(dB, dW, BH, P, N, dt, seed);

    // Compute Rough Bergomi variance factors
    fbm::core::RB_Factor rb_factor;
    std::vector<double> XI(P * N);

    REQUIRE_NOTHROW(rb_factor.compute(
        std::span<const double>(BH),
        std::span<const double>(t),
        P, N, H, xi0, eta,
        std::span<double>(XI)
    ));

    // Test at three time indices as specified
    std::vector<std::size_t> test_indices{N / 4, N / 2, N - 1};

    for (std::size_t idx : test_indices) {
        // Compute sample mean
        double sum = 0.0;
        for (std::size_t p = 0; p < P; ++p) {
            sum += XI[p * N + idx];
        }
        const double sample_mean = sum / static_cast<double>(P);

        // Compute sample standard deviation
        double sum_sq = 0.0;
        for (std::size_t p = 0; p < P; ++p) {
            const double dev = XI[p * N + idx] - sample_mean;
            sum_sq += dev * dev;
        }
        const double sample_var = sum_sq / static_cast<double>(P - 1);
        const double sample_sd = std::sqrt(sample_var);

        // Guard against zero standard deviation
        REQUIRE(sample_sd > 0.0);

        // Standard error of the mean
        const double SE = sample_sd / std::sqrt(static_cast<double>(P));

        // Theory: E[xi_t] = xi0 due to drift adjustment
        const double theory = xi0;
        const double error = std::abs(sample_mean - theory);

        // Test: |sample_mean - xi0| <= 3 * SE (99.7% confidence)
        REQUIRE(error <= 3.0 * SE);

        // Additional sanity checks
        REQUIRE(sample_mean > 0.0);  // Variance factors should be positive
        REQUIRE(std::isfinite(sample_mean));
        REQUIRE(std::isfinite(sample_sd));
    }

    // Test edge cases for the compute function
    SECTION("Input validation") {
        std::vector<double> small_BH(4), small_t(3), small_XI(4);
        small_t = {0.0, 0.5, 1.0};

        // Test invalid H values
        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(small_BH),
            std::span<const double>(small_t),
            2, 2, 0.0, xi0, eta,
            std::span<double>(small_XI)
        ), std::invalid_argument);

        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(small_BH),
            std::span<const double>(small_t),
            2, 2, 1.0, xi0, eta,
            std::span<double>(small_XI)
        ), std::invalid_argument);

        // Test invalid xi0
        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(small_BH),
            std::span<const double>(small_t),
            2, 2, H, 0.0, eta,
            std::span<double>(small_XI)
        ), std::invalid_argument);

        // Test invalid eta
        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(small_BH),
            std::span<const double>(small_t),
            2, 2, H, xi0, -1.0,
            std::span<double>(small_XI)
        ), std::invalid_argument);

        // Test size mismatches
        std::vector<double> wrong_size_BH(2);
        REQUIRE_THROWS_AS(rb_factor.compute(
            std::span<const double>(wrong_size_BH),
            std::span<const double>(small_t),
            2, 2, H, xi0, eta,
            std::span<double>(small_XI)
        ), std::invalid_argument);
    }
}
