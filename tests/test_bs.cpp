#include <catch2/catch_amalgamated.hpp>

#include <fbm/core/Simulator.h>
#include <fbm/core/INoise.h>
#include <fbm/core/IEvolver.h>
#include <fbm/core/ISimulator.h>
#include <fbm/core/GBM_Euler.h>
#include <fbm/core/BrownianNoise.h>

#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>

using namespace fbm::core;

TEST_CASE("non-uniform grid is rejected") {
    // Non-uniform time grid
    std::vector<double> time = {0.0, 0.3, 0.6, 0.8, 1.0};

    // Small simulation parameters
    const std::size_t P = 10;  // paths
    const std::size_t N = 4;   // steps

    // Output storage
    std::vector<double> S_out(P * (N + 1));

    // Create simulator
    auto noise = std::make_shared<BrownianNoise>();
    auto evolver = std::make_shared<GBM_Euler>(0.0, 0.2);
    Simulator simulator(noise, evolver);

    // Expect std::invalid_argument to be thrown
    REQUIRE_THROWS_AS(
        simulator.simulate(time, P, 7, 100.0, S_out),
        std::invalid_argument
    );
}

TEST_CASE("GBM mean and log-mean within 3 SE") {
    // Simulation parameters
    const std::size_t P = 20000;
    const std::size_t N = 1000;
    const double S0 = 100.0;
    const double mu = 0.0;
    const double sigma = 0.2;
    const double T = 1.0;
    const std::uint64_t seed = 7;

    // Uniform time grid
    std::vector<double> time(N + 1);
    for (std::size_t i = 0; i <= N; ++i) {
        time[i] = (i * T) / N;
    }

    // Output storage
    std::vector<double> S_out(P * (N + 1));

    // Create simulator
    auto noise = std::make_shared<BrownianNoise>();
    auto evolver = std::make_shared<GBM_Euler>(mu, sigma);
    Simulator simulator(noise, evolver);

    // Run simulation
    simulator.simulate(time, P, seed, S0, S_out);

    // Extract terminal values S_T
    std::vector<double> S_T(P);
    for (std::size_t p = 0; p < P; ++p) {
        S_T[p] = S_out[p * (N + 1) + N];  // Last time step for path p
    }

    // Compute sample statistics
    double sum_ST = 0.0;
    double sum_log_ST = 0.0;
    for (std::size_t p = 0; p < P; ++p) {
        sum_ST += S_T[p];
        sum_log_ST += std::log(S_T[p] / S0);
    }
    double sample_mean_ST = sum_ST / P;
    double sample_mean_log = sum_log_ST / P;

    // Theoretical values
    double theory_mean_ST = S0 * std::exp(mu * T);
    double theory_var_ST = S0 * S0 * std::exp(2.0 * mu * T) * (std::exp(sigma * sigma * T) - 1.0);
    double theory_mean_log = (mu - 0.5 * sigma * sigma) * T;

    // Standard errors
    double SE_mean_ST = std::sqrt(theory_var_ST) / std::sqrt(P);
    double SE_mean_log = (sigma * std::sqrt(T)) / std::sqrt(P);

    // Check within 3 standard errors
    double error_ST = std::abs(sample_mean_ST - theory_mean_ST);
    double error_log = std::abs(sample_mean_log - theory_mean_log);

    REQUIRE(error_ST <= 3.0 * SE_mean_ST);
    REQUIRE(error_log <= 3.0 * SE_mean_log);
}
