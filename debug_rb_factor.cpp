#include <iostream>
#include <vector>
#include <fbm/core/RoughBergomiFactor.h>
#include <fbm/core/VolterraKernelBrownian.h>
#include <fbm/core/VolterraNoise.h>

int main() {
    std::cout << "Starting RB_Factor debug test..." << std::endl;

    try {
        // Very minimal test
        const std::size_t N = 4;
        const std::size_t m = 2;
        const double T = 1.0;
        const double H = 0.5;
        const double xi0 = 0.04;
        const double eta = 1.5;
        const double dt = T / N;
        const std::uint64_t seed = 7;
        const double rho = 0.0;

        std::cout << "Creating time grid..." << std::endl;
        std::vector<double> time(N + 1);
        for (std::size_t i = 0; i <= N; ++i) {
            time[i] = dt * static_cast<double>(i);
        }

        std::cout << "Building kernel..." << std::endl;
        fbm::core::VolterraKernelStub kernel;
        std::vector<double> K(N * N, 0.0);
        kernel.build(time, H, 4, K);

        std::cout << "Creating noise generator..." << std::endl;
        fbm::core::VolterraNoiseGEMM noise_gen(std::vector<double>(K), N, rho, false);

        std::cout << "Generating noise..." << std::endl;
        std::vector<double> dB(m * N), dW(m * N), BH(m * N);
        noise_gen.sample(dB, dW, BH, m, N, dt, seed);

        std::cout << "Creating RB_Factor..." << std::endl;
        fbm::core::RB_Factor rb_factor;
        std::vector<double> XI(m * N);

        std::cout << "Computing factors..." << std::endl;
        rb_factor.compute(std::span<const double>(BH), std::span<const double>(time),
                          m, N, H, xi0, eta, std::span<double>(XI));

        std::cout << "Success! First XI value: " << XI[0] << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
