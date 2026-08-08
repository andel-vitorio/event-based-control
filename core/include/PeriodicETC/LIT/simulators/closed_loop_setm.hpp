#pragma once

#include "Algebra/Algebra.hpp"
#include "PeriodicETC/PeriodicETM.hpp"
#include "LITSystem.hpp"
#include <vector>
#include <optional>

namespace PeriodicETC
{
  namespace LIT_SETM
  {
    struct ClosedLoopResult
    {
      std::vector<double> time_data;
      std::vector<double> states_data;
      std::vector<double> control_data;
      std::vector<double> trigger_times;
    };

    class ClosedLoopSimulator
    {
    public:
      static ClosedLoopResult run(
          ControlSystems::LITSystem &plant,
          const Algebra::Vector &x0,
          const Algebra::Matrix &K,
          const StaticETMConfig &etm_config,
          double sampling_period,
          double duration,
          double time_step,
          std::optional<Algebra::Vector> w = std::nullopt);
    };
  } // namespace LIT
} // namespace PeriodicETC