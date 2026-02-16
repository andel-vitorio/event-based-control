#include "../include/ETCforLinearSystemSimulator.h"
#include "../include/Numeric.h"
#include <cmath>

/**
 * @brief Runs an open-loop simulation with a constant input.
 * * Ensures all history matrices (x, y, u) are populated to maintain
 * consistency with the binary IPC protocol.
 */
SimulationResult Simulator::run_open_loop(
    const StateSpace &sys,
    const Numeric::Vector &x0,
    double u_constant,
    double dt,
    int n_steps)
{
  SimulationResult res;

  // Pre-allocate memory to enhance performance
  res.time.reserve(n_steps);
  res.x_hist.reserve(n_steps);
  res.y_hist.reserve(n_steps);
  res.u_hist.reserve(n_steps);

  Numeric::Vector u_vec(sys.nu(), u_constant);
  Numeric::Vector x_curr = x0;

  for (int k = 0; k < n_steps; ++k)
  {
    double t = k * dt;
    res.time.push_back(t);
    res.x_hist.push_back(x_curr);
    res.u_hist.push_back(u_vec);

    // Compute output: y(t) = C*x(t) + D*u(t)
    Numeric::Vector Cx = Numeric::mat_vec_mul(sys.C, x_curr);
    Numeric::Vector Du = Numeric::mat_vec_mul(sys.D, u_vec);
    Numeric::Vector y_curr = Numeric::vec_add(Cx, Du);

    res.y_hist.push_back(y_curr);

    // Evolve state: x(t+dt) = RK5_step(x(t), u(t))
    x_curr = Numeric::rk5_step(sys.A, sys.B, x_curr, u_vec, dt);
  }

  return res;
}

/**
 * @brief Runs a closed-loop simulation using SETM logic.
 * * Implements the Static Event-Triggered Mechanism and records
 * all states, outputs, control efforts, and triggering instants.
 */
SimulationResult Simulator::run_closed_loop_setm(
    const StateSpace &sys,
    const Control::SETMParams &ctrl,
    const Numeric::Vector &x0,
    double dt,
    int n_steps)
{
  SimulationResult res;

  // Pre-allocate memory
  res.time.reserve(n_steps);
  res.x_hist.reserve(n_steps);
  res.y_hist.reserve(n_steps);
  res.u_hist.reserve(n_steps);

  Numeric::Vector x_curr = x0;
  Numeric::Vector x_hat = x0;
  Numeric::Vector xm = x0;
  Numeric::Vector u_applied(sys.nu(), 0.0);

  int steps_per_sample = static_cast<int>(std::round(ctrl.h / dt));
  double last_event_t = 0.0;

  for (int k = 0; k < n_steps; ++k)
  {
    double t = k * dt;

    // Sampling Logic (SETM)
    if (k % steps_per_sample == 0)
    {
      xm = x_curr;
      Numeric::Vector error = Numeric::vec_add(x_hat, xm, 1.0, -1.0); // e = x_hat - xm

      double term_x = Numeric::scalar_quadratic_form(xm, ctrl.Psi);
      double term_e = Numeric::scalar_quadratic_form(error, ctrl.Xi);
      bool iet_violated = (t - last_event_t) >= (ctrl.iet_max - 1e-9);

      // Check triggering condition
      if ((term_x - term_e < 0) || iet_violated || k == 0)
      {
        x_hat = xm;
        u_applied = Numeric::mat_vec_mul(ctrl.K, x_hat);
        last_event_t = t;
        res.event_times.push_back(t);
      }
    }

    // Data collection
    res.time.push_back(t);
    res.x_hist.push_back(x_curr);
    res.u_hist.push_back(u_applied);

    // Compute output: y = C*x + D*u
    Numeric::Vector Cx = Numeric::mat_vec_mul(sys.C, x_curr);
    Numeric::Vector Du = Numeric::mat_vec_mul(sys.D, u_applied);
    res.y_hist.push_back(Numeric::vec_add(Cx, Du));

    // System Evolution
    x_curr = Numeric::rk5_step(sys.A, sys.B, x_curr, u_applied, dt);
  }
  return res;
}