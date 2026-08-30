#include "PeriodicETC/LIT/LITEngine.hpp"
#include "StateSystemModel.hpp"
#include "StateSystemParser.hpp"
#include <stdexcept>

namespace PeriodicETC
{
  LITEngine::LITEngine() : system_loaded(false), state_dim(0) {}

  void LITEngine::loadSystem(const std::string &json_path)
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

  bool LITEngine::isSystemLoaded() const
  {
    return system_loaded;
  }

  int LITEngine::getStateDim() const
  {
    return state_dim;
  }

  LIT::OpenLoopResult LITEngine::runOpenLoop(
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

    LIT::OpenLoopSimulator simulator;
    return simulator.run(*plant, x0, u, w, duration, time_step);
  }

  LIT_SETM::ClosedLoopResult LITEngine::runClosedLoop(
      const Algebra::Vector &x0,
      const Algebra::Matrix &K,
      const LIT_SETM::StaticETMConfig &etm_config,
      double sampling_period,
      double duration,
      double time_step,
      const std::string &type,
      std::optional<Algebra::Vector> w)
  {
    if (!system_loaded || !plant)
    {
      throw std::runtime_error("No LIT system loaded. Please load a system before running closed-loop simulation.");
    }

    if (type == "SETM")
    {
      return LIT_SETM::run_standard_simulation(
          *plant, x0, K, etm_config, sampling_period, duration, time_step, w);
    }

    throw std::invalid_argument("Unknown closed-loop type: " + type);
  }

  LIT_SETM::ExtendedClosedLoopResult LITEngine::runClosedLoopExtended(
      const Algebra::Vector &x0,
      const Algebra::Matrix &K,
      const Algebra::Matrix &L,
      const LIT_SETM::StaticETMConfig &etm_config,
      double sampling_period,
      double duration,
      double time_step,
      const std::string &type,
      std::optional<Algebra::Vector> w)
  {
    if (!system_loaded || !plant)
    {
      throw std::runtime_error("No LIT system loaded. Please load a system before running closed-loop simulation.");
    }

    if (type == "SETM_EVENT_MAP" || type == "EVENT_MAP")
    {
      return LIT_SETM::run_observer_based_petc_simulation(
          *plant, x0, K, L, etm_config, sampling_period, duration, time_step, w);
    }

    throw std::invalid_argument("Unknown extended closed-loop type: " + type);
  }

  LIT_SETM::ExtendedClosedLoopResult LITEngine::runDualChannelClosedLoopExtended(
      const Algebra::Vector &x0,
      const Algebra::Vector &x_hat0,
      const Algebra::Matrix &K,
      const Algebra::Matrix &L,
      const LIT_SETM::StaticETMConfig &etm_sc_config,
      const LIT_SETM::StaticETMConfig &etm_ca_config,
      double sampling_period,
      double duration,
      double time_step,
      const std::string &type,
      std::optional<Algebra::Vector> w,
      double max_iet)
  {
    if (!system_loaded || !plant)
    {
      throw std::runtime_error(
          "No LIT system loaded. Please load a system before running closed-loop simulation.");
    }

    if (type == "DUAL_CHANNEL_SETM" || type == "SETM_EVENT_MAP" || type == "EVENT_MAP")
    {
      return LIT_SETM::run_dual_channel_observer_petc_simulation(
          *plant,
          x0,
          x_hat0,
          K,
          L,
          etm_sc_config,
          etm_ca_config,
          sampling_period,
          duration,
          time_step,
          w,
          max_iet);
    }

    throw std::invalid_argument("Unknown extended closed-loop type: " + type);
  }

  ControlSystems::LITSystem *LITEngine::getPlant() const
  {
    return plant.get();
  }

} // namespace PeriodicETC