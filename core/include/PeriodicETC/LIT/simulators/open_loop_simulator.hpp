#pragma once

#include "LITSystem.hpp"
#include "Algebra/Algebra.hpp"
#include <vector>
#include <optional>

namespace PeriodicETC
{
  namespace LIT
  {

    struct OpenLoopResult
    {
      std::vector<double> time_data;
      std::vector<double> states_data;
    };

    class OpenLoopSimulator
    {
    public:
      OpenLoopSimulator() = default;
      ~OpenLoopSimulator() = default;

      OpenLoopResult run(
          ControlSystems::LITSystem &plant,
          const Algebra::Vector &x0,
          std::optional<Algebra::Vector> u = std::nullopt,
          std::optional<Algebra::Vector> w = std::nullopt,
          double duration = 1.0,
          double time_step = 0.01);
    };

  } // namespace LIT
} // namespace PeriodicETC