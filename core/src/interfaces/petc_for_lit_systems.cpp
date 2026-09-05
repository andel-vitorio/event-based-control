#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <optional>

#include "StateSystemModel.hpp"
#include "StateSystemParser.hpp"
#include "LITSystem.hpp"
#include "Algebra/Algebra.hpp"
#include "EDOSolvers/EDOSolvers.hpp"
#include "PeriodicETC/LIT/LITEngine.hpp"

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace PeriodicETC
{
  namespace DualChannel
  {
    class DllEngine
    {
    private:
      PeriodicETC::LITEngine internal_engine;
      PeriodicETC::LIT_SETM::ClosedLoopWithObserversResult last_result;
      bool system_loaded = false;
      int nx = 0;
      int nu = 0;
      int ny = 0;

    public:
      DllEngine() = default;
      ~DllEngine() = default;

      void loadSystem(const std::string &json_path)
      {
        internal_engine.loadSystem(json_path);
        nx = static_cast<int>(internal_engine.getStateDim());
        nu = static_cast<int>(internal_engine.getInputDim());
        ny = static_cast<int>(internal_engine.getOutputDim());
        system_loaded = true;
      }

      void getDimensions(int *out_nx, int *out_nu, int *out_ny) const
      {
        if (!system_loaded)
        {
          throw std::runtime_error("Sistema não carregado.");
        }
        *out_nx = nx;
        *out_nu = nu;
        *out_ny = ny;
      }

      void runSimulation(
          const double *x0_ptr,
          const double *x_hat0_ptr,
          const double *x_hat_a0_ptr,
          const double *K_ptr,
          const double *L0_ptr,
          const double *L1_ptr,
          const double *L2_ptr,
          const double *Psi_sc_ptr,
          const double *Xi_sc_ptr,
          double sigma_sc,
          double threshold_sc,
          const double *Psi_ca_ptr,
          const double *Xi_ca_ptr,
          double sigma_ca,
          double threshold_ca,
          double sampling_period,
          double duration,
          double time_step,
          double max_iet_sc,
          double max_iet_ca)
      {
        using Algebra::Matrix;
        using Algebra::Vector;

        if (!system_loaded)
        {
          throw std::runtime_error("Nenhum sistema linear carregado antes da simulação.");
        }

        Vector x0(nx);
        Vector x_hat0(nx);
        Vector x_hat_a0(nx);
        for (int i = 0; i < nx; ++i)
        {
          x0[i] = x0_ptr[i];
          x_hat0[i] = x_hat0_ptr[i];
          x_hat_a0[i] = x_hat_a0_ptr[i];
        }

        Matrix K(nu, nx);
        for (int r = 0; r < nu; ++r)
          for (int c = 0; c < nx; ++c)
            K(r, c) = K_ptr[r * nx + c];

        Matrix L0(nx, nx);
        for (int r = 0; r < nx; ++r)
          for (int c = 0; c < nx; ++c)
            L0(r, c) = L0_ptr[r * nx + c];

        Matrix L1(nx, ny);
        for (int r = 0; r < nx; ++r)
          for (int c = 0; c < ny; ++c)
            L1(r, c) = L1_ptr[r * ny + c];

        Matrix L2(nx, ny);
        for (int r = 0; r < nx; ++r)
          for (int c = 0; c < ny; ++c)
            L2(r, c) = L2_ptr[r * ny + c];

        Matrix Psi_sc(ny, ny);
        Matrix Xi_sc(ny, ny);
        for (int r = 0; r < ny; ++r)
        {
          for (int c = 0; c < ny; ++c)
          {
            Psi_sc(r, c) = Psi_sc_ptr[r * ny + c];
            Xi_sc(r, c) = Xi_sc_ptr[r * ny + c];
          }
        }
        PeriodicETC::LIT_SETM::StaticETMConfig etm_sc_config;
        etm_sc_config.sigma = sigma_sc;
        etm_sc_config.threshold = threshold_sc;
        etm_sc_config.Psi = Psi_sc;
        etm_sc_config.Xi = Xi_sc;

        Matrix Psi_ca(nx, nx);
        Matrix Xi_ca(nx, nx);
        for (int r = 0; r < nx; ++r)
        {
          for (int c = 0; c < nx; ++c)
          {
            Psi_ca(r, c) = Psi_ca_ptr[r * nx + c];
            Xi_ca(r, c) = Xi_ca_ptr[r * nx + c];
          }
        }
        PeriodicETC::LIT_SETM::StaticETMConfig etm_ca_config;
        etm_ca_config.sigma = sigma_ca;
        etm_ca_config.threshold = threshold_ca;
        etm_ca_config.Psi = Psi_ca;
        etm_ca_config.Xi = Xi_ca;

        last_result = internal_engine.runDualChannelClosedLoop(
            x0, x_hat0, x_hat_a0,
            K, L0, L1, L2,
            etm_sc_config, etm_ca_config,
            sampling_period, duration, time_step,
            "DUAL_CHANNEL_SETM",
            std::nullopt, max_iet_sc, max_iet_ca);
      }

      void getResultSizes(
          int *num_steps,
          int *states_size,
          int *est_states_size,
          int *error_size,
          int *control_size,
          int *sc_events_size,
          int *ca_events_size) const
      {
        *num_steps = static_cast<int>(last_result.time_data.size());
        *states_size = static_cast<int>(last_result.states_data.size());
        *est_states_size = static_cast<int>(last_result.estimated_states_data.size());
        *error_size = static_cast<int>(last_result.estimation_error_data.size());
        *control_size = static_cast<int>(last_result.control_data.size());
        *sc_events_size = static_cast<int>(last_result.sc_trigger_times.size());
        *ca_events_size = static_cast<int>(last_result.ca_trigger_times.size());
      }

      void copyResults(
          double *t_out,
          double *x_out,
          double *x_est_out,
          double *e_out,
          double *u_out,
          double *sc_events_out,
          double *ca_events_out) const
      {
        if (t_out)
          std::copy(last_result.time_data.begin(), last_result.time_data.end(), t_out);
        if (x_out)
          std::copy(last_result.states_data.begin(), last_result.states_data.end(), x_out);
        if (x_est_out)
          std::copy(last_result.estimated_states_data.begin(), last_result.estimated_states_data.end(), x_est_out);
        if (e_out)
          std::copy(last_result.estimation_error_data.begin(), last_result.estimation_error_data.end(), e_out);
        if (u_out)
          std::copy(last_result.control_data.begin(), last_result.control_data.end(), u_out);
        if (sc_events_out)
          std::copy(last_result.sc_trigger_times.begin(), last_result.sc_trigger_times.end(), sc_events_out);
        if (ca_events_out)
          std::copy(last_result.ca_trigger_times.begin(), last_result.ca_trigger_times.end(), ca_events_out);
      }
    };
  } // namespace DualChannel
} // namespace PeriodicETC

// ---------------------------------------------------------------------------
// C API Exportada
// ---------------------------------------------------------------------------

using DualChannelEnginePtr = PeriodicETC::DualChannel::DllEngine *;

DLLEXPORT DualChannelEnginePtr create_dual_channel_engine()
{
  return new PeriodicETC::DualChannel::DllEngine();
}

DLLEXPORT void destroy_dual_channel_engine(DualChannelEnginePtr e)
{
  if (e)
    delete e;
}

DLLEXPORT int load_system_dual_channel(DualChannelEnginePtr e, const char *json_path)
{
  if (!e)
    return -1;
  try
  {
    e->loadSystem(std::string(json_path));
    return 0;
  }
  catch (const std::exception &ex)
  {
    std::cerr << "[DLL Error in load_system_dual_channel]: " << ex.what() << std::endl;
    return -2;
  }
}

DLLEXPORT int get_system_dimensions(DualChannelEnginePtr e, int *nx, int *nu, int *ny)
{
  if (!e)
    return -1;
  try
  {
    e->getDimensions(nx, nu, ny);
    return 0;
  }
  catch (const std::exception &ex)
  {
    std::cerr << "[DLL Error in get_system_dimensions]: " << ex.what() << std::endl;
    return -2;
  }
}

DLLEXPORT int run_dual_channel_simulation(
    DualChannelEnginePtr e,
    const double *x0,
    const double *x_hat0,
    const double *x_hat_a0,
    const double *K,
    const double *L0,
    const double *L1,
    const double *L2,
    const double *Psi_sc,
    const double *Xi_sc,
    double sigma_sc,
    double threshold_sc,
    const double *Psi_ca,
    const double *Xi_ca,
    double sigma_ca,
    double threshold_ca,
    double sampling_period,
    double duration,
    double time_step,
    double max_iet_sc,
    double max_iet_ca)
{
  if (!e)
    return -1;
  try
  {
    e->runSimulation(
        x0, x_hat0, x_hat_a0,
        K, L0, L1, L2,
        Psi_sc, Xi_sc, sigma_sc, threshold_sc,
        Psi_ca, Xi_ca, sigma_ca, threshold_ca,
        sampling_period, duration, time_step,
        max_iet_sc, max_iet_ca);
    return 0;
  }
  catch (const std::exception &ex)
  {
    std::cerr << "[DLL Error in run_dual_channel_simulation]: " << ex.what() << std::endl;
    return -2;
  }
}

DLLEXPORT void get_simulation_result_sizes(
    DualChannelEnginePtr e,
    int *num_steps,
    int *states_sz,
    int *est_states_sz,
    int *error_sz,
    int *control_sz,
    int *sc_events_sz,
    int *ca_events_sz)
{
  if (!e)
    return;
  e->getResultSizes(num_steps, states_sz, est_states_sz, error_sz, control_sz, sc_events_sz, ca_events_sz);
}

DLLEXPORT void copy_simulation_data(
    DualChannelEnginePtr e,
    double *t_out,
    double *x_out,
    double *x_est_out,
    double *e_out,
    double *u_out,
    double *sc_events_out,
    double *ca_events_out)
{
  if (!e)
    return;
  e->copyResults(t_out, x_out, x_est_out, e_out, u_out, sc_events_out, ca_events_out);
}