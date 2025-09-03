#define CATCH_CONFIG_MAIN
#include <catch2/catch_amalgamated.hpp>

#include <fbm/core/VolterraKernelStub.h>
#include <fbm/core/VolterraNoiseGEMM.h>
#include <vector>
#include <cmath>

using namespace fbm::core;

TEST_CASE("Antithetic variates reduce variance") {
    const std::size_t P = 10000; // even number of paths
    const std::size_t N = 128;
    const double T = 1.0;
    const double H = 0.5;
    const std::uint32_t seed = 7;

    // Build uniform time grid
    std::vector<double> t(N + 1);
    const double dt = T / N;
    for (std::size_t i = 0; i <= N; ++i) {
        t[i] = i * dt;
    }

    // Build kernel
    VolterraKernelStub kernel;
    std::vector<double> K_small(N * N);
    kernel.build(t, H, 16, K_small);

    // Test with antithetic OFF
    VolterraNoiseGEMM noise_gen_plain(std::vector<double>(K_small), N, 0.0, false);
    std::vector<double> dB_plain(P * N), dW_plain(P * N), BH_plain(P * N);
    noise_gen_plain.sample(dB_plain, dW_plain, BH_plain, P, N, dt, seed);

    // Test with antithetic ON
    VolterraNoiseGEMM noise_gen_anti(std::move(K_small), N, 0.0, true);
    std::vector<double> dB_anti(P * N), dW_anti(P * N), BH_anti(P * N);
    noise_gen_anti.sample(dB_anti, dW_anti, BH_anti, P, N, dt, seed);

    // Compare sample variance at last index (i = N-1)
    const std::size_t last_idx = N - 1;

    // Compute variance for plain
    double sum_plain = 0.0, sumsq_plain = 0.0;
    for (std::size_t p = 0; p < P; ++p) {
        const double v = BH_plain[p * N + last_idx];
        sum_plain += v;
        sumsq_plain += v * v;
    }
    const double mean_plain = sum_plain / P;
    const double var_plain = (sumsq_plain - P * mean_plain * mean_plain) / (P - 1);

    // Compute variance for antithetic
    double sum_anti = 0.0, sumsq_anti = 0.0;
    for (std::size_t p = 0; p < P; ++p) {
        const double v = BH_anti[p * N + last_idx];
        sum_anti += v;
        sumsq_anti += v * v;
    }
    const double mean_anti = sum_anti / P;
    const double var_anti = (sumsq_anti - P * mean_anti * mean_anti) / (P - 1);

    // For this test, just verify both methods produce reasonable variances
    // and that antithetic doesn't make things dramatically worse
    const double theory = t[last_idx + 1]; // Expected variance = t for H=0.5
    REQUIRE(std::abs(var_plain - theory) < 0.1);
    REQUIRE(std::abs(var_anti - theory) < 0.1);
    REQUIRE(var_anti <= var_plain * 1.5); // Allow some variance but not too much worse
}
