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

    struct ExtendedClosedLoopResult : public ClosedLoopResult
    {
      std::vector<double> estimated_states_data;
      std::vector<double> estimation_error_data;
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
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet = 0.2);

  } // namespace LIT_SETM
} // namespace PeriodicETC