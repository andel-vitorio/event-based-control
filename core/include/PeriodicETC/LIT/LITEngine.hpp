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
    size_t state_dim;
    size_t input_dim;
    size_t output_dim;

  public:
    LITEngine();
    ~LITEngine() = default;

    void loadSystem(const std::string &json_path);
    bool isSystemLoaded() const;
    size_t getStateDim() const;
    size_t getInputDim() const;
    size_t getOutputDim() const;

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

    LIT_SETM::ClosedLoopWithObserversResult runDualChannelClosedLoop(
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
        const std::string &type = "DUAL_CHANNEL_SETM",
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet_sc = std::numeric_limits<double>::infinity(),
        double max_iet_ca = std::numeric_limits<double>::infinity());

    /**
     * @brief Executa a simulação em malha fechada para a arquitetura com observador
     * contínuo de Luenberger e acionamento por eventos periódico (Experimento 2).
     *
     * @param x0              Condição inicial da planta x(0).
     * @param x_hat0          Condição inicial do observador \hat{x}(0).
     * @param K               Ganho de realimentação de controle (nu x nx).
     * @param L               Ganho do observador contínuo de Luenberger (nx x ny).
     * @param etm_config      Configuração das matrizes do ETM (Psi, Xi).
     * @param sampling_period Período base de amostragem h.
     * @param duration        Duração total da simulação.
     * @param time_step       Passo de integração temporal dt.
     * @param type            Identificador da estratégia de simulação.
     * @param w               Vetor de perturbação constante opcional.
     * @param max_iet         Limite superior do intervalo entre eventos (nu_max).
     * @return LIT_SETM::ClosedLoopWithObserversResult Séries temporais e registros de disparo.
     */
    LIT_SETM::ClosedLoopWithObserversResult runObserverPETCClosedLoop(
        const Algebra::Vector &x0,
        const Algebra::Vector &x_hat0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const LIT_SETM::StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        const std::string &type = "OBSERVER_PETC",
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet = std::numeric_limits<double>::infinity());

    LIT_SETM::ClosedLoopUnderAttackResult runDualChannelUnderAttackClosedLoop(
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
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet_sc = std::numeric_limits<double>::infinity(),
        double max_iet_ca = std::numeric_limits<double>::infinity(),
        std::function<Algebra::Vector(double)> fdi_attack = nullptr,
        double detection_threshold = 1e-12);

    LIT_SETM::ClosedLoopUnderAttackResult runDualChannelUnderAttackClosedLoop(
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
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet_sc = std::numeric_limits<double>::infinity(),
        double max_iet_ca = std::numeric_limits<double>::infinity(),
        std::function<Algebra::Vector(double)> fdi_attack = nullptr,
        double epsilon_floor = 1e-5);
  };
} // namespace PeriodicETC