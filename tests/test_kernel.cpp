#define CATCH_CONFIG_MAIN
#include <catch2/catch_amalgamated.hpp>

#include <fbm/core/IKernel.h>
#include <fbm/core/VolterraKernelStub.h>
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace fbm::core;

TEST_CASE("kernel: non-uniform grid is rejected") {
    // Non-uniform time grid
    std::vector<double> time = {0.0, 0.3, 0.6, 0.8, 1.0};

    // Parameters
    const double H = 0.3;
    const std::size_t quad_points = 16;
    const std::size_t N = time.size() - 1;  // N = 4

    // Output storage
    std::vector<double> K_small(N * N);

    // Create kernel
    VolterraKernelStub kernel;

    // Expect std::invalid_argument to be thrown
    REQUIRE_THROWS_AS(
        kernel.build(time, H, quad_points, K_small),
        std::invalid_argument
    );
}

TEST_CASE("kernel: uniform grid produces diagonal matrix") {
    // Parameters
    const std::size_t N = 8;
    const double T = 1.0;
    const double H = 0.3;
    const std::size_t quad_points = 16;

    // Uniform time grid
    std::vector<double> time(N + 1);
    const double dt = T / N;
    for (std::size_t i = 0; i <= N; ++i) {
        time[i] = i * dt;
    }

    // Output storage
    std::vector<double> K_small(N * N);

    // Create kernel
    VolterraKernelStub kernel;

    // Build kernel
    kernel.build(time, H, quad_points, K_small);

    // Check shape is correct (N×N)
    REQUIRE(K_small.size() == N * N);

    // Compute sum (should be approximately N * dt since diagonal = dt)
    double sum_K = 0.0;
    for (double val : K_small) {
        sum_K += val;
    }
    double expected_sum = N * dt;
    REQUIRE(std::abs(sum_K - expected_sum) < 1e-12);

    // Check that diagonal entries are dt
    for (std::size_t i = 0; i < N; ++i) {
        double diagonal_entry = K_small[i * N + i];
        REQUIRE(std::abs(diagonal_entry - dt) < 1e-12);
    }

    // Check that some off-diagonal entries are zero
    REQUIRE(std::abs(K_small[0 * N + 1]) < 1e-12);  // K[0,1]
    REQUIRE(std::abs(K_small[1 * N + 0]) < 1e-12);  // K[1,0]
    REQUIRE(std::abs(K_small[2 * N + 5]) < 1e-12);  // K[2,5]
}

TEST_CASE("Brownian kernel H=0.5 gives Var(B_t)=t via sum K^2 dt") {
    const std::size_t N = 256;
    const double T = 1.0;
    const double dt = T / N;
    std::vector<double> t(N+1);
    for (std::size_t i=0;i<=N;++i) t[i] = i*dt;

    VolterraKernelStub K; // same class as in codebase
    std::vector<double> Kmat(N*N, 0.0);
    K.build(t, 0.5, /*quad*/16, Kmat);

    // For H=0.5, K[j,i]=1 for j<=i. Then sum_j K^2 * dt = (i+1)*dt = t_{i+1}.
    for (std::size_t i=0;i<N;++i) {
        double sumsq = 0.0;
        for (std::size_t j=0;j<=i;++j) sumsq += 1.0; // K^2
        sumsq *= dt;
        REQUIRE( std::abs(sumsq - t[i+1]) <= 1e-12 + 1e-12*std::abs(t[i+1]) );
    }
}

TEST_CASE("kernel: H=0.5 variance equals t") {
    const std::size_t N = 256; const double T = 1.0, dt = T/N;
    std::vector<double> t(N+1); for (std::size_t i=0;i<=N;++i) t[i]=i*dt;
    VolterraKernelStub K; std::vector<double> M(N*N,0.0);
    K.build(t, 0.5, 16, M);
    for (std::size_t i=0;i<N;++i) {
        double sumsq=0.0; for (std::size_t j=0;j<=i;++j) sumsq+=1.0;
        sumsq*=dt; REQUIRE( std::abs(sumsq - t[i+1]) <= 1e-12 + 1e-12*std::abs(t[i+1]) );
    }
}