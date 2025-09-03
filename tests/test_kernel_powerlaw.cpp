#include <catch2/catch_amalgamated.hpp>
#include <fbm/core/VolterraKernelPowerLaw.h>
#include <fbm/core/VolterraNoiseGEMM.h>
#include <vector>
#include <cmath>
#include <numeric>

using Catch::Approx;

TEST_CASE("VolterraKernelPowerLaw - Power-law kernel for general H", "[kernel_powerlaw]") {

    SECTION("Backward compatibility: H=0.5 produces triangular-ones matrix") {
        const std::size_t N = 64;
        const double T = 1.0;
        const double H = 0.5;

        // Build uniform time grid
        std::vector<double> time(N + 1);
        const double dt = T / static_cast<double>(N);
        for (std::size_t i = 0; i <= N; ++i) {
            time[i] = dt * static_cast<double>(i);
        }

        // Build kernel
        fbm::core::VolterraKernelPowerLaw kernel;
        std::vector<double> K(N * N);

        REQUIRE_NOTHROW(kernel.build(time, H, 16, K));

        // Check triangular-ones structure
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                if (j <= i) {
                    REQUIRE(K[j * N + i] == Approx(1.0).margin(1e-14));
                } else {
                    REQUIRE(K[j * N + i] == Approx(0.0).margin(1e-14));
                }
            }
        }

        // Check sum equals N(N+1)/2
        const double sum_K = std::accumulate(K.begin(), K.end(), 0.0);
        const double expected_sum = static_cast<double>(N * (N + 1)) / 2.0;
        REQUIRE(sum_K == Approx(expected_sum).epsilon(1e-12));

        // Check Frobenius norm equals sqrt(N(N+1)/2) for triangular ones matrix
        double frob_sq = 0.0;
        for (double v : K) frob_sq += v * v;
        const double frob = std::sqrt(frob_sq);
        const double expected_frob = std::sqrt(static_cast<double>(N * (N + 1)) / 2.0);
        REQUIRE(frob == Approx(expected_frob).epsilon(1e-12));
    }

    SECTION("Variance checks for H=0.3 and H=0.7") {
        const double T = 1.0;
        const std::size_t N = 256;
        const std::size_t m = 10000;
        const std::uint64_t seed = 7;
        const std::vector<double> H_values = {0.3, 0.7};

        for (double H : H_values) {
            // Build uniform time grid
            std::vector<double> time(N + 1);
            const double dt = T / static_cast<double>(N);
            for (std::size_t i = 0; i <= N; ++i) {
                time[i] = dt * static_cast<double>(i);
            }

            // Build power-law kernel
            fbm::core::VolterraKernelPowerLaw kernel;
            std::vector<double> K(N * N);
            kernel.build(time, H, 16, K);

            // Generate fractional Brownian motion using the kernel
            fbm::core::VolterraNoiseGEMM noise_gen(std::vector<double>(K), N, 0.0, false);
            std::vector<double> dB(m * N), dW(m * N), BH(m * N);
            noise_gen.sample(dB, dW, BH, m, N, dt, seed);

            // Test variance at specific indices
            std::vector<std::size_t> test_indices = {N / 4, N / 2, N - 1};

            for (std::size_t idx : test_indices) {
                // Compute sample variance
                double sum = 0.0, sum_sq = 0.0;
                for (std::size_t p = 0; p < m; ++p) {
                    const double val = BH[p * N + idx];
                    sum += val;
                    sum_sq += val * val;
                }
                const double sample_mean = sum / static_cast<double>(m);
                const double sample_var = (sum_sq - static_cast<double>(m) * sample_mean * sample_mean) / static_cast<double>(m - 1);

                // Theoretical variance: Var[BH_t] = t^{2H}
                const double t_i = time[idx + 1];
                const double theoretical_var = std::pow(t_i, 2.0 * H);

                // Check relative error <= 20% (more realistic for power-law approximation)
                const double relative_error = std::abs(sample_var - theoretical_var) / theoretical_var;

                REQUIRE(relative_error <= 0.20);

                // Additional sanity checks
                REQUIRE(sample_var > 0.0);
                REQUIRE(std::isfinite(sample_var));
                REQUIRE(theoretical_var > 0.0);
            }
        }
    }

    SECTION("Input validation") {
        const std::size_t N = 8;
        const double T = 1.0;
        std::vector<double> time(N + 1);
        const double dt = T / static_cast<double>(N);
        for (std::size_t i = 0; i <= N; ++i) {
            time[i] = dt * static_cast<double>(i);
        }

        fbm::core::VolterraKernelPowerLaw kernel;
        std::vector<double> K(N * N);

        // Test invalid H values
        REQUIRE_THROWS_AS(kernel.build(time, 0.0, 16, K), std::invalid_argument);
        REQUIRE_THROWS_AS(kernel.build(time, 1.0, 16, K), std::invalid_argument);
        REQUIRE_THROWS_AS(kernel.build(time, -0.1, 16, K), std::invalid_argument);
        REQUIRE_THROWS_AS(kernel.build(time, 1.1, 16, K), std::invalid_argument);

        // Test invalid time grid
        std::vector<double> bad_time = {0.1, 0.2, 0.3}; // doesn't start at 0
        std::vector<double> K_small(2 * 2);
        REQUIRE_THROWS_AS(kernel.build(bad_time, 0.5, 16, K_small), std::invalid_argument);

        // Test non-uniform grid
        std::vector<double> non_uniform = {0.0, 0.1, 0.25, 0.4}; // non-uniform spacing
        std::vector<double> K_non_uniform(3 * 3);
        REQUIRE_THROWS_AS(kernel.build(non_uniform, 0.5, 16, K_non_uniform), std::invalid_argument);

        // Test size mismatches
        std::vector<double> K_wrong_size(5); // should be N*N = 64
        REQUIRE_THROWS_AS(kernel.build(time, 0.5, 16, K_wrong_size), std::invalid_argument);

        // Test too small time grid
        std::vector<double> tiny_time = {0.0}; // only one point
        std::vector<double> K_tiny(1);
        REQUIRE_THROWS_AS(kernel.build(tiny_time, 0.5, 16, K_tiny), std::invalid_argument);
    }

    SECTION("Kernel properties for general H") {
        const std::size_t N = 32;
        const double T = 1.0;
        const double H = 0.3;

        // Build uniform time grid
        std::vector<double> time(N + 1);
        const double dt = T / static_cast<double>(N);
        for (std::size_t i = 0; i <= N; ++i) {
            time[i] = dt * static_cast<double>(i);
        }

        // Build kernel
        fbm::core::VolterraKernelPowerLaw kernel;
        std::vector<double> K(N * N);
        kernel.build(time, H, 16, K);

        // Check lower-triangular structure
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                if (j > i) {
                    REQUIRE(K[j * N + i] == Approx(0.0).margin(1e-14));
                }
            }
        }

        // Check that kernel entries are finite and non-zero in lower triangle
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                REQUIRE(std::isfinite(K[j * N + i]));
                if (i == j) {
                    REQUIRE(K[j * N + i] > 0.0); // diagonal should be positive
                }
            }
        }
    }
}
