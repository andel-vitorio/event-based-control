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
    state_dim = static_cast<size_t>(model.states.size());
    input_dim = static_cast<size_t>(model.inputs.size());
    output_dim = static_cast<size_t>(model.outputs.size());
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

  size_t LITEngine::getStateDim() const
  {
    return state_dim;
  }

  size_t LITEngine::getInputDim() const
  {
    return input_dim;
  }

  size_t LITEngine::getOutputDim() const
  {
    return output_dim;
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

  LIT_SETM::ClosedLoopWithObserversResult LITEngine::runDualChannelClosedLoop(
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
    if (!system_loaded || !plant)
    {
      throw std::runtime_error(
          "[LITEngine::runDualChannelClosedLoop] Nenhuma planta LIT carregada. "
          "Carregue um sistema antes de executar a simulação em malha fechada.");
    }

    const std::size_t nx = plant->states();
    const std::size_t nu = plant->inputs();
    const std::size_t ny = plant->outputs();

    if (x0.size() != nx || x_hat0.size() != nx || x_hat_a0.size() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoop] Dimensões incompatíveis dos estados iniciais. "
          "Esperado: " +
          std::to_string(nx) +
          ", Recebido: x0=" + std::to_string(x0.size()) +
          ", x_hat0=" + std::to_string(x_hat0.size()) +
          ", x_hat_a0=" + std::to_string(x_hat_a0.size()));
    }

    if (K.rows() != nu || K.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoop] Dimensão de K inválida. Esperado: (" +
          std::to_string(nu) + "x" + std::to_string(nx) + ")");
    }

    if (L0.rows() != nx || L0.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoop] Dimensão de L0 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(nx) + ")");
    }

    if (L1.rows() != nx || L1.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoop] Dimensão de L1 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + ")");
    }

    if (L2.rows() != nx || L2.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoop] Dimensão de L2 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + ")");
    }

    if (sampling_period <= 0.0 || time_step <= 0.0 || duration <= 0.0)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelClosedLoop] Os parâmetros temporais (h, dt, duration) "
          "devem ser estritamente positivos.");
    }

    const double actual_max_iet_sc = (max_iet_sc <= 0.0) ? sampling_period : max_iet_sc;
    const double actual_max_iet_ca = (max_iet_ca <= 0.0) ? sampling_period : max_iet_ca;

    if (type == "DUAL_CHANNEL_SETM" || type == "SETM_EVENT_MAP" || type == "EVENT_MAP")
    {
      return LIT_SETM::run_dual_channel_observer_petc_simulation(
          *plant, x0, x_hat0, x_hat_a0, K, L0, L1, L2, etm_sc_config, etm_ca_config, sampling_period, duration, time_step, w, actual_max_iet_sc, actual_max_iet_ca);
    }

    throw std::invalid_argument(
        "[LITEngine::runDualChannelClosedLoop] Tipo de fechamento de malha desconhecido: " + type);
  }

  LIT_SETM::ClosedLoopWithObserversResult LITEngine::runObserverPETCClosedLoop(
      const Algebra::Vector &x0,
      const Algebra::Vector &x_hat0,
      const Algebra::Matrix &K,
      const Algebra::Matrix &L,
      const LIT_SETM::StaticETMConfig &etm_config,
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
          "[LITEngine::runObserverPETCClosedLoop] Nenhuma planta LIT carregada. "
          "Carregue um sistema antes de executar a simulação em malha fechada.");
    }

    const std::size_t nx = plant->states();
    const std::size_t nu = plant->inputs();
    const std::size_t ny = plant->outputs();

    if (x0.size() != nx || x_hat0.size() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runObserverPETCClosedLoop] Dimensões incompatíveis dos estados iniciais. "
          "Esperado: " +
          std::to_string(nx) +
          ", Recebido: x0=" + std::to_string(x0.size()) +
          ", x_hat0=" + std::to_string(x_hat0.size()));
    }

    if (K.rows() != nu || K.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runObserverPETCClosedLoop] Dimensão do ganho K inválida. Esperado: (" +
          std::to_string(nu) + "x" + std::to_string(nx) + "), Recebido: (" +
          std::to_string(K.rows()) + "x" + std::to_string(K.cols()) + ")");
    }

    if (L.rows() != nx || L.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runObserverPETCClosedLoop] Dimensão do ganho L do observador inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + "), Recebido: (" +
          std::to_string(L.rows()) + "x" + std::to_string(L.cols()) + ")");
    }

    if (sampling_period <= 0.0 || time_step <= 0.0 || duration <= 0.0)
    {
      throw std::invalid_argument(
          "[LITEngine::runObserverPETCClosedLoop] Os parâmetros temporais (h, dt, duration) "
          "devem ser estritamente positivos.");
    }

    if (time_step > sampling_period)
    {
      throw std::invalid_argument(
          "[LITEngine::runObserverPETCClosedLoop] O passo de integração dt não pode ser "
          "maior que o período de amostragem h.");
    }

    const double actual_max_iet = (max_iet <= 0.0) ? 0.0 : max_iet;

    if (type == "OBSERVER")
    {
      return LIT_SETM::run_observer_simulation(
          *plant, x0, x_hat0, K, L, etm_config, sampling_period, duration, time_step, w, actual_max_iet);
    }

    throw std::invalid_argument(
        "[LITEngine::runObserverPETCClosedLoop] Tipo de fechamento de malha desconhecido: " + type);
  }

  LIT_SETM::ClosedLoopUnderAttackResult LITEngine::runDualChannelUnderAttackClosedLoop(
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
      double max_iet_ca,
      std::function<Algebra::Vector(double)> fdi_attack,
      double detection_threshold)
  {
    // -------------------------------------------------------------------------
    // 1. Verificação de Carregamento da Planta
    // -------------------------------------------------------------------------
    if (!system_loaded || !plant)
    {
      throw std::runtime_error(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Nenhuma planta LIT carregada. "
          "Carregue um sistema antes de executar a simulação sob ataques.");
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
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensões incompatíveis dos estados iniciais. "
          "Esperado: " +
          std::to_string(nx) +
          ", Recebido: x0=" + std::to_string(x0.size()) +
          ", x_hat0=" + std::to_string(x_hat0.size()) +
          ", x_hat_a0=" + std::to_string(x_hat_a0.size()));
    }

    if (K.rows() != nu || K.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão de K inválida. Esperado: (" +
          std::to_string(nu) + "x" + std::to_string(nx) + "), Recebido: (" +
          std::to_string(K.rows()) + "x" + std::to_string(K.cols()) + ")");
    }

    if (L0.rows() != nx || L0.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão de L0 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(nx) + "), Recebido: (" +
          std::to_string(L0.rows()) + "x" + std::to_string(L0.cols()) + ")");
    }

    if (L1.rows() != nx || L1.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão de L1 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + "), Recebido: (" +
          std::to_string(L1.rows()) + "x" + std::to_string(L1.cols()) + ")");
    }

    if (L2.rows() != nx || L2.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão de L2 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + "), Recebido: (" +
          std::to_string(L2.rows()) + "x" + std::to_string(L2.cols()) + ")");
    }

    // if (w.has_value() && w->size() != 0 && w->size() != plant->disturbances())
    // {
    //   throw std::invalid_argument(
    //       "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão da perturbação w incompatível. "
    //       "Esperado: " +
    //       std::to_string(plant->disturbances()) +
    //       ", Recebido: " + std::to_string(w->size()));
    // }

    if (detection_threshold < 0.0)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] O limiar de detecção "
          "(detection_threshold) deve ser não-negativo.");
    }

    // -------------------------------------------------------------------------
    // 3. Validação de Parâmetros Temporais
    // -------------------------------------------------------------------------
    if (sampling_period <= 0.0 || time_step <= 0.0 || duration <= 0.0)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Os parâmetros temporais "
          "(sampling_period, time_step, duration) devem ser estritamente positivos.");
    }

    if (time_step > sampling_period)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] O passo de integração time_step "
          "não pode ser maior que o período de amostragem sampling_period.");
    }

    const double actual_max_iet_sc = (max_iet_sc <= 0.0) ? sampling_period : max_iet_sc;
    const double actual_max_iet_ca = (max_iet_ca <= 0.0) ? sampling_period : max_iet_ca;

    if (type == "DUAL_CHANNEL_SETM" || type == "SETM_EVENT_MAP" ||
        type == "EVENT_MAP" || type == "DUAL_CHANNEL_ATTACK" ||
        type == "OBSERVER_ATTACK")
    {
      return LIT_SETM::run_dual_channel_under_attacks_simulation(
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
          actual_max_iet_ca,
          fdi_attack,
          detection_threshold);
    }

    throw std::invalid_argument(
        "[LITEngine::runDualChannelUnderAttackClosedLoop] Tipo de simulação desconhecido: " + type);
  }

  LIT_SETM::ClosedLoopUnderAttackResult LITEngine::runDualChannelUnderAttackClosedLoop(
      const Algebra::Vector &x0,
      const Algebra::Vector &x_hat0,
      const Algebra::Vector &x_hat_a0,
      const Algebra::Vector &tilde_x0,
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
      double max_iet_ca,
      std::function<Algebra::Vector(double)> fdi_attack,
      double epsilon_floor)
  {
    // -------------------------------------------------------------------------
    // 1. Verificação de Carregamento da Planta
    // -------------------------------------------------------------------------
    if (!system_loaded || !plant)
    {
      throw std::runtime_error(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Nenhuma planta LIT carregada. "
          "Carregue um sistema antes de executar a simulação sob ataques.");
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
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensões incompatíveis dos estados iniciais. "
          "Esperado: " +
          std::to_string(nx) +
          ", Recebido: x0=" + std::to_string(x0.size()) +
          ", x_hat0=" + std::to_string(x_hat0.size()) +
          ", x_hat_a0=" + std::to_string(x_hat_a0.size()));
    }

    if (tilde_x0.size() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão da incerteza inicial tilde_x0 incompatível. "
          "Esperado: " +
          std::to_string(nx) +
          ", Recebido: " + std::to_string(tilde_x0.size()));
    }

    for (std::size_t i = 0; i < nx; ++i)
    {
      if (tilde_x0[i] < 0.0)
      {
        throw std::invalid_argument(
            "[LITEngine::runDualChannelUnderAttackClosedLoop] Os elementos de tilde_x0 "
            "devem ser não-negativos. Índice " +
            std::to_string(i) + " = " + std::to_string(tilde_x0[i]));
      }
    }

    if (epsilon_floor < 0.0)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] O piso de tolerância numérica "
          "(epsilon_floor) deve ser não-negativo.");
    }

    if (K.rows() != nu || K.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão de K inválida. Esperado: (" +
          std::to_string(nu) + "x" + std::to_string(nx) + "), Recebido: (" +
          std::to_string(K.rows()) + "x" + std::to_string(K.cols()) + ")");
    }

    if (L0.rows() != nx || L0.cols() != nx)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão de L0 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(nx) + "), Recebido: (" +
          std::to_string(L0.rows()) + "x" + std::to_string(L0.cols()) + ")");
    }

    if (L1.rows() != nx || L1.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão de L1 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + "), Recebido: (" +
          std::to_string(L1.rows()) + "x" + std::to_string(L1.cols()) + ")");
    }

    if (L2.rows() != nx || L2.cols() != ny)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Dimensão de L2 inválida. Esperado: (" +
          std::to_string(nx) + "x" + std::to_string(ny) + "), Recebido: (" +
          std::to_string(L2.rows()) + "x" + std::to_string(L2.cols()) + ")");
    }

    // -------------------------------------------------------------------------
    // 3. Validação de Parâmetros Temporais
    // -------------------------------------------------------------------------
    if (sampling_period <= 0.0 || time_step <= 0.0 || duration <= 0.0)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] Os parâmetros temporais "
          "(sampling_period, time_step, duration) devem ser estritamente positivos.");
    }

    if (time_step > sampling_period)
    {
      throw std::invalid_argument(
          "[LITEngine::runDualChannelUnderAttackClosedLoop] O passo de integração time_step "
          "não pode ser maior que o período de amostragem sampling_period.");
    }

    const double actual_max_iet_sc = (max_iet_sc <= 0.0) ? sampling_period : max_iet_sc;
    const double actual_max_iet_ca = (max_iet_ca <= 0.0) ? sampling_period : max_iet_ca;

    if (type == "DUAL_CHANNEL_SETM" || type == "SETM_EVENT_MAP" ||
        type == "EVENT_MAP" || type == "DUAL_CHANNEL_ATTACK" ||
        type == "OBSERVER_ATTACK")
    {
      return LIT_SETM::run_dual_channel_under_attacks_simulation(
          *plant,
          x0,
          x_hat0,
          x_hat_a0,
          tilde_x0,
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
          actual_max_iet_ca,
          fdi_attack,
          epsilon_floor);
    }

    throw std::invalid_argument(
        "[LITEngine::runDualChannelUnderAttackClosedLoop] Tipo de simulação desconhecido: " + type);
  }

  ControlSystems::LITSystem *LITEngine::getPlant() const
  {
    return plant.get();
  }

} // namespace PeriodicETC