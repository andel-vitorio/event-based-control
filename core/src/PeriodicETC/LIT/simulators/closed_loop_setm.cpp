#include "PeriodicETC/LIT/simulators/closed_loop_setm.hpp"
#include "EDOSolvers/EDOSolvers.hpp"
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>

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
          result.sc_trigger_times.push_back(t);
        else if (sampler.shouldSample(t))
        {
          if (etm.evaluateEvent(x, x_transmitted))
          {
            result.sc_trigger_times.push_back(t);
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
        std::optional<Algebra::Vector> w,
        double max_iet_sc,
        double max_iet_ca)
    {
      using Algebra::Matrix;
      using Algebra::Vector;

      const auto timepts = Algebra::arange(0.0, duration, time_step);
      const std::size_t num_steps = timepts.size();

      const int state_dim = static_cast<int>(plant.states());
      const int output_dim = static_cast<int>(plant.outputs());
      const int input_dim = static_cast<int>(plant.inputs());

      ClosedLoopWithObserversResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.estimated_states_data.reserve(num_steps * state_dim);
      result.estimation_error_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);
      result.sc_trigger_times.reserve(num_steps / 2);
      result.ca_trigger_times.reserve(num_steps / 2);

      const Vector actual_w = w.has_value() ? *w : Vector(0);

      StaticETM etm_sc(etm_sc_config, output_dim);
      StaticETM etm_ca(etm_ca_config, state_dim);
      Sampler sampler(sampling_period, 0.0);

      const Matrix A = plant.getA();
      const Matrix B = plant.getB();
      const Matrix C = plant.getC();

      EDOSolvers::RK5 solver(
          [&plant, &A, &B, &C, &L0, &L2, state_dim, input_dim](
              double /* t */,
              const Vector &z_sys,
              const Vector &signal) -> Vector
          {
            const Vector x_sys = z_sys.slice(0, state_dim);
            const Vector x_est = z_sys.slice(state_dim, 2 * state_dim);
            const Vector x_est_a = z_sys.slice(2 * state_dim, 3 * state_dim);
            const Vector u = signal.slice(0, input_dim);
            const Vector disturbance = signal.slice(input_dim, signal.size());
            const Vector x_dot = plant.stateDerivative(x_sys, u, disturbance);
            const Vector Bu = B * u;
            const Vector delta_x_est = x_est - x_est_a;
            const Vector x_est_dot = A * x_est + Bu + L0 * delta_x_est;
            const Vector x_est_a_dot = A * x_est_a + Bu + L2 * (C * delta_x_est);

            return Vector::concatenate(x_dot, Vector::concatenate(x_est_dot, x_est_a_dot));
          });

      Vector x = x0;
      Vector x_est = x_hat0;
      Vector x_est_a = x_hat_a0;
      Vector z = Vector::concatenate(x, Vector::concatenate(x_est, x_est_a));

      Vector y_r_sc = C * x0; // y(t_k^{sc})
      Vector x_r_ca = x_hat0; // \hat{x}(t_\ell^{ca})
      Vector u_actual = K * x_r_ca;

      double last_sc_trigger = 0.0;
      double last_ca_trigger = 0.0;

      for (std::size_t i = 0; i < num_steps; ++i)
      {
        const double t = timepts[i];

        x = z.slice(0, state_dim);
        x_est = z.slice(state_dim, 2 * state_dim);
        x_est_a = z.slice(2 * state_dim, 3 * state_dim);

        const Vector y = C * x;

        if (i == 0)
        {
          result.sc_trigger_times.push_back(t);
          result.ca_trigger_times.push_back(t);
          last_sc_trigger = t;
          last_ca_trigger = t;
          y_r_sc = y;
          x_est = x_est + L1 * (y - C * x_est);
          x_r_ca = x_est;
          z = Vector::concatenate(x, Vector::concatenate(x_est, x_est_a));
          u_actual = K * x_r_ca;
        }
        else if (sampler.shouldSample(t))
        {
          const bool sc_event = etm_sc.evaluateEvent(y, y_r_sc);
          const bool sc_timeout = (t - last_sc_trigger >= max_iet_sc - 1e-9);

          if (sc_event || sc_timeout)
          {
            y_r_sc = y;
            last_sc_trigger = t;
            result.sc_trigger_times.push_back(t);
            x_est = x_est + L1 * (y - C * x_est);
            z = Vector::concatenate(x, Vector::concatenate(x_est, x_est_a));
          }

          const bool ca_event = etm_ca.evaluateEvent(x_est, x_r_ca);
          const bool ca_timeout = (t - last_ca_trigger >= max_iet_ca - 1e-9);

          if (ca_event || ca_timeout)
          {
            x_r_ca = x_est;
            last_ca_trigger = t;
            result.ca_trigger_times.push_back(t);
            u_actual = K * x_r_ca;
          }
        }

        const Vector estimation_error = x - x_est;
        for (int j = 0; j < state_dim; ++j)
        {
          result.states_data.push_back(x[j]);
          result.estimated_states_data.push_back(x_est[j]);
          result.estimation_error_data.push_back(estimation_error[j]);
        }

        for (int j = 0; j < input_dim; ++j)
          result.control_data.push_back(u_actual[j]);

        if (i + 1 < num_steps)
        {
          const Vector signal = Vector::concatenate(u_actual, actual_w);
          z = solver.step(t, z, signal, time_step);
        }
      }

      return result;
    }

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
        std::optional<Algebra::Vector> w,
        double max_iet)
    {
      using Algebra::Matrix;
      using Algebra::Vector;

      const auto timepts = Algebra::arange(0.0, duration, time_step);
      const std::size_t num_steps = timepts.size();

      const int state_dim = static_cast<int>(plant.states());
      const int output_dim = static_cast<int>(plant.outputs());
      const int input_dim = static_cast<int>(plant.inputs());

      ClosedLoopWithObserversResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.estimated_states_data.reserve(num_steps * state_dim);
      result.estimation_error_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);
      result.ca_trigger_times.reserve(num_steps / 2);

      const Vector actual_w = w.has_value() ? *w : Vector(0);

      StaticETM etm_ca(etm_config, state_dim);
      Sampler sampler(sampling_period, 0.0);

      const Matrix A = plant.getA();
      const Matrix B = plant.getB();
      const Matrix C = plant.getC();

      EDOSolvers::RK5 solver(
          [&plant, &A, &B, &C, &L, state_dim, input_dim](
              double /* t */,
              const Vector &z_sys,
              const Vector &signal) -> Vector
          {
            const Vector x_sys = z_sys.slice(0, state_dim);
            const Vector x_est = z_sys.slice(state_dim, 2 * state_dim);
            const Vector u = signal.slice(0, input_dim);
            const Vector disturbance = signal.slice(input_dim, signal.size());

            const Vector x_dot = plant.stateDerivative(x_sys, u, disturbance);
            const Vector y = C * x_sys;
            const Vector Bu = B * u;
            const Vector innovation = y - (C * x_est);
            const Vector x_est_dot = A * x_est + Bu + (L * innovation);

            return Vector::concatenate(x_dot, x_est_dot);
          });

      Vector x = x0;
      Vector x_est = x_hat0;
      Vector z = Vector::concatenate(x, x_est);

      Vector x_r_ca = x_hat0;
      Vector u_actual = K * x_r_ca;
      double last_ca_trigger = 0.0;

      const double effective_max_iet = (max_iet > 0.0) ? max_iet : (duration + 1.0);

      for (std::size_t i = 0; i < num_steps; ++i)
      {
        const double t = timepts[i];

        x = z.slice(0, state_dim);
        x_est = z.slice(state_dim, 2 * state_dim);
        // x_est = x;

        if (i == 0)
        {
          result.ca_trigger_times.push_back(t);
          last_ca_trigger = t;
          x_r_ca = x_est;
          u_actual = K * x_r_ca;
        }
        else if (sampler.shouldSample(t))
        {
          const bool ca_event = etm_ca.evaluateEvent(x_est, x_r_ca);
          const bool ca_timeout = ((t - last_ca_trigger) >= (effective_max_iet - 1e-9));

          if (ca_event || ca_timeout)
          {
            x_r_ca = x_est;
            last_ca_trigger = t;
            result.ca_trigger_times.push_back(t);
            u_actual = K * x_r_ca;
          }
        }

        const Vector estimation_error = x - x_est;
        for (int j = 0; j < state_dim; ++j)
        {
          result.states_data.push_back(x[j]);
          result.estimated_states_data.push_back(x_est[j]);
          result.estimation_error_data.push_back(estimation_error[j]);
        }

        for (int j = 0; j < input_dim; ++j)
        {
          result.control_data.push_back(u_actual[j]);
        }

        if (i + 1 < num_steps)
        {
          const Vector signal = Vector::concatenate(u_actual, actual_w);
          z = solver.step(t, z, signal, time_step);
        }
      }

      return result;
    }

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
        std::optional<Algebra::Vector> w,
        double max_iet_sc,
        double max_iet_ca,
        std::function<Algebra::Vector(double)> fdi_attack,
        double detection_threshold)
    {
      using Algebra::Matrix;
      using Algebra::Vector;

      const auto timepts = Algebra::arange(0.0, duration, time_step);
      const std::size_t num_steps = timepts.size();

      const int state_dim = static_cast<int>(plant.states());
      const int output_dim = static_cast<int>(plant.outputs());
      const int input_dim = static_cast<int>(plant.inputs());

      ClosedLoopUnderAttackResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.estimated_states_data.reserve(num_steps * state_dim);
      result.estimation_error_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);
      result.residual.reserve(num_steps * output_dim);
      result.residual_norm.reserve(num_steps);
      result.sc_trigger_times.reserve(num_steps / 2);
      result.ca_trigger_times.reserve(num_steps / 2);
      result.alarm_active.reserve(num_steps / 2);
      result.malicious_signal.reserve((num_steps / 2) * output_dim);

      const Vector actual_w = w.has_value() ? *w : Vector(0);

      StaticETM etm_sc(etm_sc_config, output_dim);
      StaticETM etm_ca(etm_ca_config, state_dim);
      Sampler sampler(sampling_period, 0.0);

      const Matrix A = plant.getA();
      const Matrix B = plant.getB();
      const Matrix C = plant.getC();

      const Matrix I_ny = Algebra::identity(output_dim);

      EDOSolvers::RK5 solver(
          [&plant, &A, &B, &C, &L0, &L2, state_dim, input_dim](
              double /* t */,
              const Vector &z_sys,
              const Vector &signal) -> Vector
          {
            const Vector x_sys = z_sys.slice(0, state_dim);
            const Vector x_est = z_sys.slice(state_dim, 2 * state_dim);
            const Vector x_est_a = z_sys.slice(2 * state_dim, 3 * state_dim);
            const Vector u = signal.slice(0, input_dim);
            const Vector disturbance = signal.slice(input_dim, signal.size());

            const Vector x_dot = plant.stateDerivative(x_sys, u, disturbance);
            const Vector Bu = B * u;
            const Vector delta_x_est = x_est - x_est_a;

            const Vector x_est_dot = A * x_est + Bu + (L0 * delta_x_est);
            const Vector x_est_a_dot = A * x_est_a + Bu + (L2 * (C * delta_x_est));

            return Vector::concatenate(x_dot, Vector::concatenate(x_est_dot, x_est_a_dot));
          });

      Vector x = x0;
      Vector x_est = x_hat0;
      Vector x_est_a = x_hat_a0;
      Vector z = Vector::concatenate(x, Vector::concatenate(x_est, x_est_a));

      Vector y_r_sc = C * x0;
      Vector x_r_ca = x_hat0;
      Vector u_actual = K * x_r_ca;

      Vector current_residual = C * x0 - C * x_hat0;

      double last_sc_trigger = 0.0;
      double last_ca_trigger = 0.0;

      bool corrupted_state_active = false;

      for (std::size_t i = 0; i < num_steps; ++i)
      {
        const double t = timepts[i];

        x = z.slice(0, state_dim);
        x_est = z.slice(state_dim, 2 * state_dim);
        x_est_a = z.slice(2 * state_dim, 3 * state_dim);

        const Vector y = C * x;

        if (i == 0)
        {
          result.sc_trigger_times.push_back(t);
          result.ca_trigger_times.push_back(t);
          last_sc_trigger = t;
          last_ca_trigger = t;
          y_r_sc = y;

          Vector atk(output_dim);
          bool attack_at_event = false;
          if (fdi_attack)
          {
            atk = fdi_attack(t);
            if (Algebra::quadratic_form(atk, I_ny) > 1e-12)
              attack_at_event = true;
          }

          for (int j = 0; j < output_dim; ++j)
            result.malicious_signal.push_back(atk[j]);

          const Vector y_received = attack_at_event ? (y + atk) : y;
          const Vector innovation_sc = y_received - (C * x_est);
          current_residual = innovation_sc;

          const double r_jump_sq = Algebra::quadratic_form(innovation_sc, I_ny);
          const double r_jump_norm = std::sqrt(std::max(0.0, r_jump_sq));

          const bool current_alarm_active =
              (r_jump_norm > detection_threshold);
          result.alarm_active.push_back(current_alarm_active);

          if (current_alarm_active && !attack_at_event)
            result.false_positives++;
          if (!current_alarm_active && attack_at_event)
          {
            result.malicious_control_count++;
            corrupted_state_active = true;
          }

          const Vector y_for_update = current_alarm_active ? y : y_received;
          x_est = x_est + (L1 * (y_for_update - (C * x_est)));
          x_r_ca = x_est;
          z = Vector::concatenate(x, Vector::concatenate(x_est, x_est_a));
          u_actual = K * x_r_ca;
        }
        else if (sampler.shouldSample(t))
        {
          const bool sc_event = etm_sc.evaluateEvent(y, y_r_sc);
          const bool sc_timeout = ((t - last_sc_trigger) >= (max_iet_sc - 1e-9));

          if (sc_event || sc_timeout)
          {
            y_r_sc = y;
            last_sc_trigger = t;
            result.sc_trigger_times.push_back(t);

            Vector atk(output_dim);
            bool attack_at_event = false;
            if (fdi_attack)
            {
              atk = fdi_attack(t);
              if (Algebra::quadratic_form(atk, I_ny) > 1e-12)
                attack_at_event = true;
            }

            for (int j = 0; j < output_dim; ++j)
              result.malicious_signal.push_back(atk[j]);

            const Vector y_received = attack_at_event ? (y + atk) : y;
            const Vector innovation_sc = y_received - (C * x_est);
            current_residual = innovation_sc;

            const double r_jump_sq = Algebra::quadratic_form(innovation_sc, I_ny);
            const double r_jump_norm = std::sqrt(std::max(0.0, r_jump_sq));

            const bool current_alarm_active = (r_jump_norm > detection_threshold);
            result.alarm_active.push_back(current_alarm_active);

            if (current_alarm_active && !attack_at_event)
              result.false_positives++;

            if (!current_alarm_active && attack_at_event)
            {
              result.malicious_control_count++;
              corrupted_state_active = true;
            }
            else if (current_alarm_active || !attack_at_event)
              corrupted_state_active = false;

            const Vector y_for_update = current_alarm_active ? y : y_received;
            x_est = x_est + (L1 * (y_for_update - (C * x_est)));
            z = Vector::concatenate(x, Vector::concatenate(x_est, x_est_a));
          }
          else
            current_residual = y - (C * x_est);

          const bool ca_event = etm_ca.evaluateEvent(x_est, x_r_ca);
          const bool ca_timeout = ((t - last_ca_trigger) >= (max_iet_ca - 1e-9));

          if (ca_event || ca_timeout)
          {
            x_r_ca = x_est;
            last_ca_trigger = t;
            result.ca_trigger_times.push_back(t);
            u_actual = K * x_r_ca;
          }
        }
        else
          current_residual = y - (C * x_est);

        const double r_norm_sq = Algebra::quadratic_form(current_residual, I_ny);
        const double r_norm = std::sqrt(std::max(0.0, r_norm_sq));

        result.residual_norm.push_back(r_norm);

        for (int j = 0; j < output_dim; ++j)
          result.residual.push_back(current_residual[j]);

        if (corrupted_state_active)
          result.malicious_control_steps++;

        const Vector estimation_error = x - x_est;
        for (int j = 0; j < state_dim; ++j)
        {
          result.states_data.push_back(x[j]);
          result.estimated_states_data.push_back(x_est[j]);
          result.estimation_error_data.push_back(estimation_error[j]);
        }

        for (int j = 0; j < input_dim; ++j)
          result.control_data.push_back(u_actual[j]);

        if (i + 1 < num_steps)
        {
          const Vector signal = Vector::concatenate(u_actual, actual_w);
          z = solver.step(t, z, signal, time_step);
        }
      }

      return result;
    }
  } // namespace LIT_SETM
} // namespace PeriodicETC