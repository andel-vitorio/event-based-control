#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include "DynamicalSystem.h"
#include "Simulator.h"

/**
 * @brief Configuration structure for a simulation experiment.
 *
 * Holds parameters defining the simulation environment, timing,
 * and the dynamical system (plant) to be simulated.
 */
struct ExperimentConfig
{
  std::string name; ///< Name of the experiment or system configuration.
  double duration;  ///< Total duration of the simulation in seconds.
  double dt;        ///< Time step for the numerical integration (simulation step).
  StateSpace plant; ///< The state-space representation of the plant.
  double design_h;  ///< Sampling period (h) used in event-triggered designs.
};

/**
 * @brief Utility class for loading experiment configurations.
 *
 * Provides static methods to parse configuration files (e.g., JSON)
 * and populate ExperimentConfig structures.
 */
class ConfigLoader
{
public:
  /**
   * @brief Loads an experiment configuration from a JSON file.
   *
   * Parses the specified JSON file to extract simulation parameters
   * and system matrices.
   *
   * @param json_path Path to the JSON configuration file.
   * @return ExperimentConfig The populated configuration structure.
   * @throws std::runtime_error if the file cannot be opened or parsing fails.
   */
  static ExperimentConfig load(const std::string &json_path);
};

#endif