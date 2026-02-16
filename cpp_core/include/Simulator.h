#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "DynamicalSystem.h"
#include <vector>
#include "Numeric.h"

/**
 * @brief Container for simulation results.
 */
struct SimulationResult
{
  Numeric::Matrix y_hist;   ///< History of outputs (rows=outputs, cols=time steps).
  Numeric::Matrix x_hist;   ///< History of states (rows=states, cols=time steps).
  std::vector<double> time; ///< Time vector.
};

/**
 * @brief Static class for running system simulations.
 */
class Simulator
{
public:
  /**
   * @brief Runs an open-loop simulation with a constant input.
   *
   * Simulates the system response over a specified duration given an initial state
   * and a constant control input.
   *
   * @param sys The state-space system to simulate.
   * @param x0 Initial state vector.
   * @param u_constant Constant input value applied to all input channels.
   * @param dt Integration time step.
   * @param n_steps Number of simulation steps.
   * @return SimulationResult Structure containing time, state, and output histories.
   */
  static SimulationResult run_open_loop(
      const StateSpace &sys,
      const Numeric::Vector &x0,
      double u_constant,
      double dt,
      int n_steps);
};

#endif