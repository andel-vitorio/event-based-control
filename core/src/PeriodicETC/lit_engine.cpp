#include "PeriodicETC/LIT/lit_engine.hpp"
#include "StateSystemModel.hpp"
#include "StateSystemParser.hpp"
#include <stdexcept>

namespace PeriodicETC
{
  namespace LIT
  {
    Engine::Engine() : system_loaded(false), state_dim(0) {}

    void Engine::loadSystem(const std::string &json_path)
    {
      using namespace ControlSystems;
      StateSystemModel model = StateSystemParser::parseFromFile(json_path);
      plant = std::make_unique<LITSystem>(LITSystem::fromModel(model));
      state_dim = static_cast<int>(model.states.size());
      if (state_dim == 0)
      {
        throw std::runtime_error("The loaded system has zero states. Please check the JSON file: " + json_path);
      }
      system_loaded = true;
    }

    bool Engine::isSystemLoaded() const
    {
      return system_loaded;
    }

    int Engine::getStateDim() const
    {
      return state_dim;
    }

    OpenLoopResult Engine::runOpenLoop(
        const Algebra::Vector &x0,
        std::optional<Algebra::Vector> u,
        std::optional<Algebra::Vector> w,
        double duration,
        double time_step)
    {
      if (!system_loaded || !plant)
      {
        throw std::runtime_error("LIT system is not loaded. Please load a system before running open-loop simulation.");
      }

      OpenLoopSimulator simulator;
      return simulator.run(*plant, x0, u, w, duration, time_step);
    }

  } // namespace LIT
} // namespace PeriodicETC