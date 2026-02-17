/**
 * @file ETCforLinearSystemSimulator.cpp
 * @brief Event-Triggered Control simulation kernels.
 */

#include "../include/ETCforLinearSystemSimulator.h"
#include "../include/Numeric.h"
#include <cmath>
#include <algorithm>

SimulationResult Simulator::run_open_loop(
    const StateSpace &sys,
    const Numeric::Vector &x0,
    double u_constant,
    double dt,
    int n_steps)
{
  SimulationResult res;
  res.time.reserve(n_steps);
  res.x_hist.reserve(n_steps);
  res.y_hist.reserve(n_steps);
  res.u_hist.reserve(n_steps);

  Numeric::Vector u_vec(sys.nu(), u_constant);
  Numeric::Vector x_curr = x0; // Local copy to prevent side effects

  for (int k = 0; k < n_steps; ++k)
  {
    double t = k * dt;
    res.time.push_back(t);
    res.x_hist.push_back(x_curr);
    res.u_hist.push_back(u_vec);

    Numeric::Vector y_curr = Numeric::vec_add(
        Numeric::mat_vec_mul(sys.C, x_curr),
        Numeric::mat_vec_mul(sys.D, u_vec));
    res.y_hist.push_back(y_curr);

    x_curr = Numeric::rk5_step(sys.A, sys.B, x_curr, u_vec, dt);
  }
  return res;
}

SimulationResult Simulator::run_closed_loop_setm(
    const StateSpace &sys,
    const Control::SETMParams &ctrl,
    const Numeric::Vector &x0,
    double dt,
    int n_steps)
{
  SimulationResult res;
  res.time.reserve(n_steps);
  res.x_hist.reserve(n_steps);
  res.y_hist.reserve(n_steps);
  res.u_hist.reserve(n_steps);

  Numeric::Vector x_curr = x0; // Local copy
  Numeric::Vector x_hat = x0;  // Initial transmission
  Numeric::Vector u_applied(sys.nu(), 0.0);

  int steps_per_sample = static_cast<int>(std::round(ctrl.h / dt));
  double last_event_t = 0.0;

  for (int k = 0; k < n_steps; ++k)
  {
    double t = k * dt;

    if (k % steps_per_sample == 0)
    {
      // Error: e = x_hat - x(k)
      Numeric::Vector error = Numeric::vec_add(x_hat, x_curr, 1.0, -1.0);

      double term_x = Numeric::scalar_quadratic_form(x_curr, ctrl.Psi);
      double term_e = Numeric::scalar_quadratic_form(error, ctrl.Xi);

      // Numerical safety margin to match float precision of Python
      bool iet_violated = (t - last_event_t) >= (ctrl.iet_max - 1e-9);

      if ((term_x - term_e < 0.0) || iet_violated || k == 0)
      {
        x_hat = x_curr;
        u_applied = Numeric::mat_vec_mul(ctrl.K, x_hat);
        last_event_t = t;
        res.event_times.push_back(t);
      }
    }

    res.time.push_back(t);
    res.x_hist.push_back(x_curr);
    res.u_hist.push_back(u_applied);
    res.y_hist.push_back(Numeric::vec_add(
        Numeric::mat_vec_mul(sys.C, x_curr),
        Numeric::mat_vec_mul(sys.D, u_applied)));

    x_curr = Numeric::rk5_step(sys.A, sys.B, x_curr, u_applied, dt);
  }
  return res;
}

std::vector<double> Simulator::run_recurrence_map_setm(
    const StateSpace &sys,
    const Control::SETMParams &ctrl,
    const Numeric::Vector &x0,
    double duration)
{
  std::vector<double> event_times;
  Numeric::Matrix Ad, Bd;
  Numeric::discretize_zoh(sys.A, sys.B, ctrl.h, Ad, Bd);

  const int total_steps = static_cast<int>(std::round(duration / ctrl.h)) + 1;
  const int iet_max_steps = static_cast<int>(std::round(ctrl.iet_max / ctrl.h));

  Numeric::Vector x_tk = x0;
  int current_step_idx = 0;
  event_times.push_back(0.0);

  while (current_step_idx < total_steps)
  {
    const int remaining = total_steps - current_step_idx;
    if (remaining <= 0)
      break;

    const int lookahead = (iet_max_steps < remaining) ? iet_max_steps : remaining;

    Numeric::Vector x_mh = x_tk;
    Numeric::Vector u_hold = Numeric::mat_vec_mul(ctrl.K, x_tk);
    int m_star = lookahead;
    bool triggered = false;

    for (int m = 1; m < lookahead; ++m)
    {
      x_mh = Numeric::vec_add(
          Numeric::mat_vec_mul(Ad, x_mh),
          Numeric::mat_vec_mul(Bd, u_hold));

      Numeric::Vector e = Numeric::vec_add(x_tk, x_mh, 1.0, -1.0);
      double term_x = Numeric::scalar_quadratic_form(x_mh, ctrl.Psi);
      double term_e = Numeric::scalar_quadratic_form(e, ctrl.Xi);

      if ((term_x - term_e) < 0.0)
      {
        m_star = m;
        triggered = true;
        break;
      }
    }

    if (!triggered)
    {
      x_mh = Numeric::vec_add(
          Numeric::mat_vec_mul(Ad, x_mh),
          Numeric::mat_vec_mul(Bd, u_hold));
    }

    current_step_idx += m_star;
    if (current_step_idx < total_steps)
      event_times.push_back(current_step_idx * ctrl.h);

    x_tk = x_mh;
  }
  return event_times;
}

std::vector<std::vector<int>> Simulator::run_parallel_symbolic_sequences(
    const StateSpace &sys,
    const Control::SETMParams &ctrl,
    const std::vector<Numeric::Vector> &initial_states,
    double duration)
{
  int num_samples = initial_states.size();
  std::vector<std::vector<int>> all_sequences(num_samples);

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < num_samples; ++i)
  {
    std::vector<double> event_times = run_recurrence_map_setm(
        sys, ctrl, initial_states[i], duration);

    std::vector<int> k_seq;
    if (event_times.size() > 1)
    {
      for (size_t j = 1; j < event_times.size(); ++j)
      {
        double delta = event_times[j] - event_times[j - 1];
        k_seq.push_back(static_cast<int>(std::round(delta / ctrl.h)));
      }
    }
    all_sequences[i] = k_seq;
  }

  return all_sequences;
}