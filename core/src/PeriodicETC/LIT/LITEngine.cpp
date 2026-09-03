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

  LIT_SETM::ExtendedClosedLoopResult LITEngine::runDualChannelClosedLoopExtended_old(
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
      double max_iet_sc,
      double max_iet_ca)
  {
    if (!system_loaded || !plant)
    {
      throw std::runtime_error(
          "No LIT system loaded. Please load a system before running closed-loop simulation.");
    }

    if (type == "DUAL_CHANNEL_SETM" || type == "SETM_EVENT_MAP" || type == "EVENT_MAP")
    {
      return LIT_SETM::run_dual_channel_observer_petc_simulation_old(
          *plant, x0, x_hat0, K, L, etm_sc_config, etm_ca_config, sampling_period, duration, time_step, w,
          max_iet_sc, max_iet_ca);
    }

    throw std::invalid_argument("Unknown extended closed-loop type: " + type);
  }

  LIT_SETM::ExtendedClosedLoopResult LITEngine::runDualChannelClosedLoopExtended(
      const Algebra::Vector &x0,
      const Algebra::Vector &x_hat0,
      const Algebra::Vector &x_hat_a0,
      const Algebra::Matrix &K,
      const Algebra::Matrix &L0,
      const Algebra::Matrix &L1,
      const Algebra::Matrix &L2,
      const LIT_SETM::StaticETMConfig &etm_sc_config,
      const LIT_SETM::StaticETMConfig &etm_ca_config,
      double sampling_period,
      double duration,
      double time_step,
      const std::string &type,
      std::optional<Algebra::Vector> w,
      double max_iet_sc,
      double max_iet_ca)
  {
    // -------------------------------------------------------------------------
    // 1. Verificação de Carregamento da Planta
    // -------------------------------------------------------------------------
    if (!system_loaded || !plant)
    {
      throw std::runtime_error(
          "[LITEngine::runDualChannelClosedLoopExtended] Nenhuma planta LIT carregada. "
          "Carregue um sistema antes de executar a simulação em malha fechada.");
    }

    // -------------------------------------------------------------------------
    // 2. Validação Estrutural de Dimensões
    // -------------------------------------------------------------------------
    const std::size_t nx = plant->states();
    const std::size_t nu = plant->inputs();
    const std::size_t ny = plant->outputs();

    if (x0.size() != nx || x_hat0.size() != nx || x_hat_a0.size() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoopExtended] Dimensões incompatíveis dos estados iniciais. "
          "Esperado: " +
          std::to_string(nx) +
          ", Recebido: x0=" + std::to_string(x0.size()) +
          ", x_hat0=" + std::to_string(x_hat0.size()) +
          ", x_hat_a0=" + std::to_string(x_hat_a0.size()));
    }

    if (K.rows() != nu || K.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoopExtended] Dimensão de K inválida. Esperado: (" +
          std::to_string(nu) + "x" + std::to_string(nx) + ")");
    }

    if (L0.rows() != nx || L0.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoopExtended] Dimensão de L0 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(nx) + ")");
    }

    if (L1.rows() != nx || L1.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoopExtended] Dimensão de L1 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + ")");
    }

    if (L2.rows() != nx || L2.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoopExtended] Dimensão de L2 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + ")");
    }

    // -------------------------------------------------------------------------
    // 3. Validação de Parâmetros Temporais
    // -------------------------------------------------------------------------
    if (sampling_period <= 0.0 || time_step <= 0.0 || duration <= 0.0)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoopExtended] Os parâmetros temporais (h, dt, duration) "
          "devem ser estritamente positivos.");
    }

    const double actual_max_iet_sc = (max_iet_sc <= 0.0) ? sampling_period : max_iet_sc;
    const double actual_max_iet_ca = (max_iet_ca <= 0.0) ? sampling_period : max_iet_ca;

    // -------------------------------------------------------------------------
    // 4. Despacho por Tipo de Configuração
    // -------------------------------------------------------------------------
    if (type == "DUAL_CHANNEL_SETM" || type == "SETM_EVENT_MAP" || type == "EVENT_MAP")
    {
      return LIT_SETM::run_dual_channel_augmented_observer_petc_simulation(
          *plant,
          x0,
          x_hat0,
          x_hat_a0,
          K,
          L0,
          L1,
          L2,
          etm_sc_config,
          etm_ca_config,
          sampling_period,
          duration,
          time_step,
          w,
          actual_max_iet_sc,
          actual_max_iet_ca);
    }

    throw std::invalid_argument(
        "[LITEngine::runDualChannelClosedLoopExtended] Tipo de fechamento de malha desconhecido: " + type);
  }

  ControlSystems::LITSystem *LITEngine::getPlant() const
  {
    return plant.get();
  }

} // namespace PeriodicETC