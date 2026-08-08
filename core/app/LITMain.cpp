#include <iostream>
#include <variant>
#include <filesystem>
#include <fstream>
#include <vector>

#include "StateSystemModel.hpp"
#include "StateSystemParser.hpp"
#include "BinaryLogger.hpp"
#include "LITSystem.hpp"
#include "LPVSystem.hpp"
#include "Algebra/Algebra.hpp"
#include "EDOSolvers/EDOSolvers.hpp"

#include "PeriodicETC/LIT/LITEngine.hpp"

using namespace Algebra;
using namespace ControlSystems;
namespace fs = std::filesystem;

auto null = std::nullopt;

void run_open_loop_simulation(PeriodicETC::LITEngine &engine)
{
  double duration = 5.0;
  double time_step = 1e-5;

  int state_dim = engine.getStateDim();
  Vector x0(state_dim);
  for (int i = 0; i < state_dim; ++i)
    x0[i] = 1.0;

  Vector u(1);

  auto result = engine.runOpenLoop(x0, u, null, duration, time_step);

  fs::path dir = "simulations/lit-system-open-loop-smooth";
  fs::create_directories(dir);

  BinaryLogger::dump(dir / "time.bin", result.time_data);

  int num_steps = static_cast<int>(result.time_data.size());
  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> state_trajectory;
    state_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      state_trajectory.push_back(result.states_data[step * state_dim + i]);
    std::string filename = "x" + std::to_string(i + 1) + ".bin";
    BinaryLogger::dump(dir / filename, state_trajectory);
  }
  std::cout << "Open-loop simulation completed. Data saved in: " << dir
            << std::endl;
}

void run_closed_loop_setm_simulation(PeriodicETC::LITEngine &engine)
{
  double duration = 15.0;
  double time_step = 1e-4;
  double sampling_period = 0.05;

  int state_dim = engine.getStateDim();
  Vector x0(state_dim);
  for (int i = 0; i < state_dim; ++i)
    x0[i] = 1.0;

  Matrix K(1, 2, {-6.56e-2, -1.08e1});
  Matrix Xi(2, 2, {1.78, 3.89e-1, 3.89e-1, 3.27e1});
  Matrix Psi(2, 2, {1.60, 3.50e-1, 3.50e-1, 2.94e1});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_config;
  etm_config.sigma = 1.0;
  etm_config.threshold = 0.0;

  auto result = engine.runClosedLoop(
      x0, K, etm_config, sampling_period, duration, time_step, "SETM", null);

  fs::path dir = "simulations/lit-system-closed-loop-setm";
  fs::create_directories(dir);

  BinaryLogger::dump(dir / "time.bin", result.time_data);
  BinaryLogger::dump(dir / "trigger_times.bin", result.trigger_times);

  int num_steps = static_cast<int>(result.time_data.size());

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> state_trajectory;
    state_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      state_trajectory.push_back(result.states_data[step * state_dim + i]);
    std::string filename = "x" + std::to_string(i + 1) + ".bin";
    BinaryLogger::dump(dir / filename, state_trajectory);
  }

  int input_dim = 1;
  for (int i = 0; i < input_dim; ++i)
  {
    std::vector<double> control_trajectory;
    control_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      control_trajectory.push_back(result.control_data[step * input_dim + i]);
    std::string filename = "u" + std::to_string(i + 1) + ".bin";
    BinaryLogger::dump(dir / filename, control_trajectory);
  }

  std::cout << "Closed-loop SETM simulation completed. Data saved in: " << dir
            << " | Total triggers: " << result.trigger_times.size() << std::endl;
}

int main()
{
  std::string systems_directory = "../experiments/data/";
  fs::path jsonPath = systems_directory + "sys-01.json";

  try
  {
    PeriodicETC::LITEngine engine;
    engine.loadSystem(jsonPath.string());

    int number_states = engine.getStateDim();
    std::cout << "Number of states in the loaded LIT system: " << number_states << std::endl;

    // run_open_loop_simulation(engine);
    run_closed_loop_setm_simulation(engine);
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Erro durante a execucao: " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}