#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "DynamicalSystem.h"
#include <vector>
#include "Numeric.h"

namespace Control
{
  /**
   * @brief Parameters for Static Event-Triggered Mechanism (SETM).
   */
  struct SETMParams
  {
    Numeric::Matrix K;   ///< Controller gain matrix.
    Numeric::Matrix Xi;  ///< Error-related triggering matrix (Ξ).
    Numeric::Matrix Psi; ///< State-related triggering matrix (Ψ).
    double h;            ///< Sampling period.
    double iet_max;      ///< Maximum Inter-Event Time.
  };
}

/**
 * @brief Container for simulation results.
 */
struct SimulationResult
{
  Numeric::Matrix y_hist;          ///< History of outputs (rows=ny, cols=steps).
  Numeric::Matrix x_hist;          ///< History of states (rows=nx, cols=steps).
  Numeric::Matrix u_hist;          ///< History of control inputs (rows=nu, cols=steps).
  std::vector<double> time;        ///< Simulation time vector.
  std::vector<double> event_times; ///< Time instants where events were triggered.
};

/**
 * @brief Static class for running system simulations.
 */
class Simulator
{
public:
  /**
   * @brief Runs an open-loop simulation with a constant input.
   */
  static SimulationResult run_open_loop(
      const StateSpace &sys,
      const Numeric::Vector &x0,
      double u_constant,
      double dt,
      int n_steps);

  /**
   * @brief Runs a closed-loop simulation using SETM.
   * * @param sys The state-space plant model.
   * @param ctrl Control and ETM parameters (K, Xi, Psi, h).
   * @param x0 Initial state vector.
   * @param dt Integration time step.
   * @param n_steps Total number of integration steps.
   */
  static SimulationResult run_closed_loop_setm(
      const StateSpace &sys,
      const Control::SETMParams &ctrl,
      const Numeric::Vector &x0,
      double dt,
      int n_steps);
};

#endif