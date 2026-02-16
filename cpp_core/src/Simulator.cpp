#include "Simulator.h"
#include "Numeric.h"

SimulationResult Simulator::run_open_loop(
    const StateSpace &sys,
    const Numeric::Vector &x0,
    double u_constant,
    double dt,
    int n_steps)
{
  SimulationResult res;

  // Reserve memory to avoid reallocations during the loop
  res.time.reserve(n_steps);
  res.x_hist.reserve(n_steps);
  res.y_hist.reserve(n_steps);

  Numeric::Vector u_vec(sys.nu(), u_constant);
  Numeric::Vector x_curr = x0;

  for (int k = 0; k < n_steps; ++k)
  {
    double t = k * dt;
    res.time.push_back(t);
    res.x_hist.push_back(x_curr);

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