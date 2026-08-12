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
            x_transmitted = x;
            result.trigger_times.push_back(t);
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

    ExtendedClosedLoopResult run_event_map_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w)
    {
      using Algebra::Matrix;
      using Algebra::Vector;

      auto timepts = Algebra::arange(0.0, duration, time_step);

      const int num_steps = static_cast<int>(timepts.size());
      const int state_dim = static_cast<int>(x0.size());
      const int input_dim = static_cast<int>(K.rows());

      ExtendedClosedLoopResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);
      result.estimated_states_data.reserve(num_steps * state_dim);
      result.estimation_error_data.reserve(num_steps * state_dim);

      Vector actual_w = w.has_value() ? *w : Vector(0);

      StaticETM etm(etm_config, state_dim);
      Sampler sampler(sampling_period, 0.0);

      Matrix A = plant.getA();
      Matrix B = plant.getB();

      // Integrador numérico contínuo para a planta física real
      EDOSolvers::RK5 solver(
          [&plant]([[maybe_unused]] double t,
                   const Vector &x_sys,
                   const Vector &signal)
          {
            std::size_t nu = plant.inputs();
            Vector ut = signal.slice(0, nu);
            Vector wt = signal.slice(nu, signal.size());
            return plant.stateDerivative(x_sys, ut, wt);
          });

      Vector x = x0;             // Estado físico contínuo da planta (real)
      Vector x_transmitted = x0; // Estado transmitido pela rede
      Vector x_est = x0;         // Estado estimado internamente via mapa discreto
      Vector u = K * x_transmitted;

      double t_k = 0.0; // Instante da última transmissão

      for (size_t i = 0; i < timepts.size(); ++i)
      {
        double t = timepts[i];

        if (i == 0)
        {
          result.trigger_times.push_back(t);
          t_k = t;
        }
        else if (sampler.shouldSample(t))
        {
          if (etm.evaluateEvent(x, x_transmitted))
          {
            double nu_k = t - t_k; // \nu_k = t_{k+1} - t_k

            if (nu_k > 0.0)
            {
              // Atualização analítica discreta no instante de disparo t_{k+1}
              Matrix exp_A_nu = Algebra::matrix_exp(A * nu_k);
              Matrix Gamma_u = Algebra::integrate_matrix_exp_B(A, B, nu_k, true);

              // x_{est}(t_{k+1}) = e^{A \nu_k} x_{est}(t_k) + \Gamma_u(\nu_k) u(t_k)
              x_est = exp_A_nu * x_est + Gamma_u * u;
            }

            x_transmitted = x;
            result.trigger_times.push_back(t);
            t_k = t;
          }
        }

        u = K * x_transmitted;

        // Erro entre estado contínuo real e estimativa discreta
        Vector err = x - x_est;

        for (int j = 0; j < state_dim; ++j)
        {
          result.states_data.push_back(x[j]);
          result.estimated_states_data.push_back(x_est[j]);
          result.estimation_error_data.push_back(err[j]);
        }

        for (int j = 0; j < input_dim; ++j)
          result.control_data.push_back(u[j]);

        // Avanço contínuo da planta física via RK5
        Vector signal = Vector::concatenate(u, actual_w);
        x = solver.step(t, x, signal, time_step);
      }

      return result;
    }

  } // namespace LIT_SETM
} // namespace PeriodicETC