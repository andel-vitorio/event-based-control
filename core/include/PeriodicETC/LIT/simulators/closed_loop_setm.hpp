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
      std::vector<double> sc_trigger_times;
    };

    struct ClosedLoopWithObserversResult : public ClosedLoopResult
    {
      std::vector<double> estimated_states_data;
      std::vector<double> estimation_error_data;
      std::vector<double> ca_trigger_times;
    };

    struct ClosedLoopUnderAttackResult : public ClosedLoopWithObserversResult
    {
      std::vector<double> residual;
      std::vector<double> residual_norm;
      std::vector<bool> alarm_active;
      std::vector<double> alarm_trigger_times;
      std::vector<double> malicious_signal;

      int false_positives = 0;
      int malicious_control_count = 0;
      int malicious_control_steps = 0;
    };

    /**
     * @brief Executa a simulação em malha fechada para o sistema LIT.
     */
    ClosedLoopResult run_standard_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w = std::nullopt);

    ClosedLoopWithObserversResult run_dual_channel_observer_petc_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Vector &x_hat0,
        const Algebra::Vector &x_hat_a0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L0,
        const Algebra::Matrix &L1,
        const Algebra::Matrix &L2,
        const StaticETMConfig &etm_sc_config,
        const StaticETMConfig &etm_ca_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet_sc = std::numeric_limits<double>::infinity(),
        double max_iet_ca = std::numeric_limits<double>::infinity());

    ClosedLoopWithObserversResult run_observer_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Vector &x_hat0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet = std::numeric_limits<double>::infinity());

    ClosedLoopUnderAttackResult run_dual_channel_under_attacks_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Vector &x_hat0,
        const Algebra::Vector &x_hat_a0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L0,
        const Algebra::Matrix &L1,
        const Algebra::Matrix &L2,
        const StaticETMConfig &etm_sc_config,
        const StaticETMConfig &etm_ca_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet_sc = std::numeric_limits<double>::infinity(),
        double max_iet_ca = std::numeric_limits<double>::infinity(),
        std::function<Algebra::Vector(double)> fdi_attack = nullptr,
        double detection_threshold = 1e-9);
  } // namespace LIT_SETM
} // namespace PeriodicETC