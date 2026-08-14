#pragma once

#include <string>
#include <memory>
#include "LITSystem.hpp"
#include "Algebra/Algebra.hpp"

#include "simulators/open_loop_simulator.hpp"
#include "simulators/closed_loop_setm.hpp"

namespace PeriodicETC
{
  class LITEngine
  {
  private:
    std::unique_ptr<ControlSystems::LITSystem> plant;
    bool system_loaded;
    int state_dim;

  public:
    LITEngine();
    ~LITEngine() = default;

    void loadSystem(const std::string &json_path);
    bool isSystemLoaded() const;
    int getStateDim() const;

    ControlSystems::LITSystem *getPlant() const;

    LIT::OpenLoopResult runOpenLoop(
        const Algebra::Vector &x0,
        std::optional<Algebra::Vector> u = std::nullopt,
        std::optional<Algebra::Vector> w = std::nullopt,
        double duration = 1.0,
        double time_step = 1e-4);

    LIT_SETM::ClosedLoopResult runClosedLoop(
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const LIT_SETM::StaticETMConfig &etm_config,
        double sampling_period,
        double duration = 1.0,
        double time_step = 1e-4,
        const std::string &type = "SETM",
        std::optional<Algebra::Vector> w = std::nullopt);

    LIT_SETM::ExtendedClosedLoopResult runClosedLoopExtended(
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const LIT_SETM::StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        const std::string &type,
        std::optional<Algebra::Vector> w);
  };

} // namespace PeriodicETC