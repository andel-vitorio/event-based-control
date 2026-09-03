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

    LIT_SETM::ExtendedClosedLoopResult runClosedLoopExtended(
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const LIT_SETM::StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        const std::string &type,
        std::optional<Algebra::Vector> w);

    LIT_SETM::ExtendedClosedLoopResult runDualChannelClosedLoopExtended_old(
        const Algebra::Vector &x0,
        const Algebra::Vector &x_hat0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const LIT_SETM::StaticETMConfig &etm_sc_config,
        const LIT_SETM::StaticETMConfig &etm_ca_config,
        double sampling_period,
        double duration,
        double time_step,
        const std::string &type = "DUAL_CHANNEL_SETM",
        std::optional<Algebra::Vector> w = std::nullopt,
        double max_iet_sc = std::numeric_limits<double>::infinity(),
        double max_iet_ca = std::numeric_limits<double>::infinity());

    LIT_SETM::ExtendedClosedLoopResult runDualChannelClosedLoopExtended(
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
  };
} // namespace PeriodicETC