#include <catch2/catch_amalgamated.hpp>
#include <fbm/core/VolterraKernelPowerLaw.h>
#include <fbm/core/VolterraNoise.h>
#include <fbm/core/RoughBergomiFactor.h>
#include <vector>
#include <cmath>

TEST_CASE("RB factor respects xi0(t) curve mean", "[rb_curve]") {
    const double T = 1.0;
    const std::size_t N = 256, m = 20000;
    const double H = 0.7;
    const double eta = 1.3;
    const std::uint64_t seed = 7;

    // time grid
    std::vector<double> time(N + 1);
    for (std::size_t i = 0; i <= N; ++i) {
        time[i] = T * double(i) / N;
    }

    // power-law kernel + noise
    fbm::core::VolterraKernelPowerLaw K;
    std::vector<double> Kmat(N * N);
    K.build(time, H, 16, Kmat);
    fbm::core::VolterraNoise noise(Kmat, N, 0.0, false);
    std::vector<double> dB(m * N), dW(m * N), BH(m * N);
    noise.sample(dB, dW, BH, m, N, T / double(N), seed);

    // xi0(t) curve (non constant)
    const double beta0 = 0.03, beta1 = 0.02, beta2 = 0.01, tau1 = 0.5, tau2 = 1.5;
    std::vector<double> xi0t(N);
    for (std::size_t i = 0; i < N; ++i) {
        double t = time[i + 1];
        xi0t[i] = beta0 + beta1 * std::exp(-t / tau1) + beta2 * (t / tau2) * std::exp(-t / tau2);
    }

    // RB factor
    fbm::core::RoughBergomiFactor f;
    std::vector<double> XI(m * N);
    f.compute(std::span<const double>(BH), std::span<const double>(time), m, N, H, std::span<const double>(xi0t), eta, std::span<double>(XI));

    // mean(XI_t) ≈ xi0(t)
    for (std::size_t idx : {N/4, N/2, N-1}) {
        double mu = 0.0;
        for (std::size_t p = 0; p < m; ++p) {
            mu += XI[p * N + idx];
        }
        mu /= double(m);
        double th = xi0t[idx];
        REQUIRE(std::abs(mu - th) <= 0.02 * th + 5e-4); // 2% or small abs tol
    }
}
