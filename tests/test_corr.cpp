#include <catch2/catch_amalgamated.hpp>
#include <fbm/core/VolterraNoise.h>
#include <fbm/core/VolterraKernelBrownian.h>
#include <vector>
#include <cmath>

TEST_CASE("Correlation between dW and dB", "[correlation]") {
    const std::size_t P = 20000;
    const std::size_t N = 256;
    const double T = 1.0;
    const double H = 0.5;
    const double rho = -0.7;
    const std::uint64_t seed = 7;

    // Build uniform time grid
    std::vector<double> time(N + 1);
    const double dt = T / static_cast<double>(N);
    for (std::size_t i = 0; i <= N; ++i) {
        time[i] = dt * static_cast<double>(i);
    }

    // Build Volterra kernel
    fbm::core::VolterraKernelStub kernel;
    std::vector<double> K_small(N * N, 0.0);
    kernel.build(time, H, 16, K_small);

    // Create noise generator with correlation
    fbm::core::VolterraNoiseGEMM noise_gen(std::move(K_small), N, rho, false);

    // Generate correlated noise
    std::vector<double> dB(P * N), dW(P * N), BH(P * N);
    noise_gen.sample(dB, dW, BH, P, N, dt, seed);

    // Test correlation at three different time indices
    std::vector<std::size_t> test_indices = {N/4, N/2, N-1};

    for (std::size_t idx : test_indices) {
        // Compute sample means
        double mean_dB = 0.0, mean_dW = 0.0;
        for (std::size_t p = 0; p < P; ++p) {
            mean_dB += dB[p * N + idx];
            mean_dW += dW[p * N + idx];
        }
        mean_dB /= static_cast<double>(P);
        mean_dW /= static_cast<double>(P);

        // Compute covariance and variances
        double cov = 0.0, var_dB = 0.0, var_dW = 0.0;
        for (std::size_t p = 0; p < P; ++p) {
            const double dB_dev = dB[p * N + idx] - mean_dB;
            const double dW_dev = dW[p * N + idx] - mean_dW;
            cov += dB_dev * dW_dev;
            var_dB += dB_dev * dB_dev;
            var_dW += dW_dev * dW_dev;
        }
        cov /= static_cast<double>(P - 1);
        var_dB /= static_cast<double>(P - 1);
        var_dW /= static_cast<double>(P - 1);

        // Compute Pearson correlation
        const double corr_hat = cov / std::sqrt(var_dB * var_dW);

        // Test that estimated correlation is close to expected rho
        REQUIRE(std::abs(corr_hat - rho) <= 0.02);
    }
}
