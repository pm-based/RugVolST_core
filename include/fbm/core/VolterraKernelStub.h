#pragma once
#include "fbm/core/IKernel.h"

namespace fbm::core {

class VolterraKernelStub final : public IKernel {
public:
  void build(std::span<const double> time,
             double H,
             std::size_t quad_points,
             std::span<double> K_small) const override;
};

} // namespace fbm::core
