#include "PeriodicETC/LIT/simulators/closed_loop_setm.hpp"
#include "EDOSolvers/EDOSolvers.hpp"

namespace PeriodicETC
{
  namespace LIT_SETM
  {
    ClosedLoopResult run_standard_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w)
    {
      using Algebra::Vector;

      auto timepts = Algebra::arange(0.0, duration, time_step);

      const int num_steps = static_cast<int>(timepts.size());
      const int state_dim = static_cast<int>(x0.size());
      const int input_dim = static_cast<int>(K.rows());

      ClosedLoopResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);

      Vector actual_w = w.has_value() ? *w : Vector(0);

      StaticETM etm(etm_config, state_dim);
      Sampler sampler(sampling_period, 0.0);

      EDOSolvers::RK5 solver(
          [&plant]([[maybe_unused]] double t,
                   const Vector &x,
                   const Vector &signal)
          {
            std::size_t nu = plant.inputs();
            Vector ut = signal.slice(0, nu);
            Vector wt = signal.slice(nu, signal.size());
            return plant.stateDerivative(x, ut, wt);
          });

      Vector x = x0;
      Vector x_transmitted = x0;
      Vector u = K * x_transmitted;

      for (size_t i = 0; i < timepts.size(); ++i)
      {
        double t = timepts[i];

        if (i == 0)
          result.trigger_times.push_back(t);
        else if (sampler.shouldSample(t))
        {
          if (etm.evaluateEvent(x, x_transmitted))
          {
            result.trigger_times.push_back(t);
            x_transmitted = x;
          }
        }

        u = K * x_transmitted;

        for (int j = 0; j < state_dim; ++j)
          result.states_data.push_back(x[j]);

        for (int j = 0; j < input_dim; ++j)
          result.control_data.push_back(u[j]);

        Vector signal = Vector::concatenate(u, actual_w);
        x = solver.step(t, x, signal, time_step);
      }

      return result;
    }

    ExtendedClosedLoopResult run_observer_based_petc_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w)
    {
      using Algebra::Matrix;
      using Algebra::Vector;

      // ------------------------------------------------------------------
      // Time grid
      // ------------------------------------------------------------------

      auto timepts = Algebra::arange(0.0, duration, time_step);

      const std::size_t num_steps = timepts.size();
      const int state_dim = static_cast<int>(x0.size());
      const int output_dim = state_dim; // Current experiment: y = x
      const int input_dim = static_cast<int>(K.rows());

      // ------------------------------------------------------------------
      // Result initialization
      // ------------------------------------------------------------------

      ExtendedClosedLoopResult result;

      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.estimated_states_data.reserve(num_steps * state_dim);
      result.estimation_error_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);

      // ------------------------------------------------------------------
      // External disturbance
      // ------------------------------------------------------------------
      Vector actual_w = w.has_value() ? *w : Vector(0.0);

      // ------------------------------------------------------------------
      // PETC components
      // ------------------------------------------------------------------
      StaticETM etm(etm_config, output_dim);
      Sampler sampler(sampling_period, 0.0);

      // ------------------------------------------------------------------
      // Plant matrices
      // ------------------------------------------------------------------
      Matrix A_cont = plant.getA();
      Matrix B_cont = plant.getB();
      Matrix Ah = Algebra::matrix_exp(A_cont * sampling_period);
      Matrix Bh = Algebra::integrate_matrix_exp_B(
          A_cont, B_cont, sampling_period, true);

      // ------------------------------------------------------------------
      // Continuous-time solver for the physical plant
      // ------------------------------------------------------------------
      EDOSolvers::RK5 solver(
          [&plant](double /* t */, const Vector &x_sys, const Vector &signal)
          {
            const std::size_t nu = plant.inputs();
            Vector u = signal.slice(0, nu);
            Vector disturbance = signal.slice(nu, signal.size());
            return plant.stateDerivative(
                x_sys,
                u,
                disturbance);
          });

      // ------------------------------------------------------------------
      // System states
      // ------------------------------------------------------------------
      Vector x = x0;
      Vector x_est = x0;
      Vector y_r = x0;
      Vector u_transmitted = K * x_est;

      // ------------------------------------------------------------------
      // Main simulation loop
      // ------------------------------------------------------------------
      for (std::size_t i = 0; i < timepts.size(); ++i)
      {
        const double t = timepts[i];
        Vector x_est_next = x_est;

        if (i == 0)
        {
          result.trigger_times.push_back(t);
          Vector y = x;
          y_r = y;
          Vector innovation = L * (y - x_est);
          x_est_next = x_est + innovation;
          u_transmitted = K * x_est_next;
        }
        else if (sampler.shouldSample(t))
        {
          x_est_next = Ah * x_est + Bh * u_transmitted;
          Vector y = x;
          bool event_triggered = etm.evaluateEvent(y, y_r);
          if (event_triggered)
          {
            y_r = y;
            result.trigger_times.push_back(t);
            Vector innovation = L * (y - x_est_next);
            x_est_next = x_est_next + innovation;
            u_transmitted = K * y;
          }
        }

        Vector estimation_error = x - x_est_next;

        for (int j = 0; j < state_dim; ++j)
        {
          result.states_data.push_back(x[j]);
          result.estimated_states_data.push_back(x_est_next[j]);
          result.estimation_error_data.push_back(estimation_error[j]);
        }

        for (int j = 0; j < input_dim; ++j)
          result.control_data.push_back(u_transmitted[j]);

        x_est = x_est_next;
        Vector signal = Vector::concatenate(u_transmitted, actual_w);
        x = solver.step(t, x, signal, time_step);
      }

      return result;
    }

  } // namespace LIT_SETM
} // namespace PeriodicETC