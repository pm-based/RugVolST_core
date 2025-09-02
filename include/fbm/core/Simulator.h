#pragma once
#include <span>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "INoise.h"
#include "IEvolver.h"
#include "ISimulator.h"

namespace fbm::core {
class Simulator final : public ISimulator {
public:
  Simulator(std::shared_ptr<INoise> noise,
            std::shared_ptr<IEvolver> evolver,
            std::size_t chunk_paths = 10000);

  void simulate(std::span<const double> time,
                std::size_t n_paths,
                std::uint64_t seed,
                double S0,
                std::span<double> S_out) const override;

private:
  std::shared_ptr<INoise>   noise_;
  std::shared_ptr<IEvolver> evolver_;
  std::size_t               chunk_;
};
} // namespace fbm::core
