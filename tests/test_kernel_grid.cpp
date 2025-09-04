#include <catch2/catch_amalgamated.hpp>
#include <fbm/core/VolterraKernelPowerLaw.h>
#include <vector>

TEST_CASE("VolterraKernelPowerLaw rejects non-uniform grid","[kernel]") {
    fbm::core::VolterraKernelPowerLaw K;
    std::vector<double> time{0.0, 0.1, 0.25, 0.4};
    std::vector<double> M(9, 0.0);
    REQUIRE_THROWS_AS(K.build(time, 0.5, 16, M), std::invalid_argument);
}