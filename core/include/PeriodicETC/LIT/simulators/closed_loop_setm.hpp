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

    struct ExtendedClosedLoopResult : public ClosedLoopResult
    {
      std::vector<double> estimated_states_data;
      std::vector<double> estimation_error_data;
      std::vector<double> ca_trigger_times;
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

    ExtendedClosedLoopResult run_observer_based_petc_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w = std::nullopt);

    ExtendedClosedLoopResult run_dual_channel_observer_petc_simulation_old(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Vector &x_hat0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const StaticETMConfig &etm_sc_config,
        const StaticETMConfig &etm_ca_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet_sc = std::numeric_limits<double>::infinity(),
        double max_iet_ca = std::numeric_limits<double>::infinity());

    ExtendedClosedLoopResult run_dual_channel_augmented_observer_petc_simulation(
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

  } // namespace LIT_SETM
} // namespace PeriodicETC