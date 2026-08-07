#pragma once

#include <string>
#include <memory>
#include "LITSystem.hpp"
#include "Algebra/Algebra.hpp"

#include "simulators/open_loop_simulator.hpp"

namespace PeriodicETC
{
  namespace LIT
  {

    using namespace ControlSystems;

    class Engine
    {
    private:
      std::unique_ptr<LITSystem> plant;
      bool system_loaded;
      int state_dim;

    public:
      Engine();
      ~Engine() = default;

      void loadSystem(const std::string &json_path);
      bool isSystemLoaded() const;
      int getStateDim() const;

      OpenLoopResult runOpenLoop(
          const Algebra::Vector &x0,
          std::optional<Algebra::Vector> u = std::nullopt,
          std::optional<Algebra::Vector> w = std::nullopt,
          double duration = 1.0,
          double time_step = 1e-4);
    };

  } // namespace LIT
} // namespace PeriodicETC