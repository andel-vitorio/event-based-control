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
  double sampling_period = 0.1;
  // double sampling_period = 0.05;

  int state_dim = engine.getStateDim();
  Vector x0(state_dim);
  for (int i = 0; i < state_dim; ++i)
    x0[i] = 1.0;

  // Matrix K(1, 2, {-6.56e-2, -1.08e1});
  // Matrix Xi(2, 2, {1.78, 3.89e-1, 3.89e-1, 3.27e1});
  // Matrix Psi(2, 2, {1.60, 3.50e-1, 3.50e-1, 2.94e1});
  Matrix K(1, 2, {-3.73e+01, -1.77e+01});
  Matrix Xi(2, 2, {1.07e+06, 5.09e+05, 5.09e+05, 2.41e+05});
  Matrix Psi(2, 2, {3.77e+04, 1.68e+04, 1.68e+04, 1.85e+04});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_config;
  etm_config.sigma = 1.0;
  etm_config.threshold = 0.0;
  etm_config.Psi = Psi;
  etm_config.Xi = Xi;

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

void run_closed_loop_setm_event_map_simulation(PeriodicETC::LITEngine &engine)
{
  double duration = 30.0;
  double time_step = 1e-4;
  double sampling_period = 1e-2;

  int state_dim = engine.getStateDim();
  Vector x0(state_dim);
  x0[0] = 1.0;
  x0[1] = -1.0;

  Matrix K(1, 2, {1.93e+00, -2.59e+00});
  Matrix L(2, 1, {-2.77e-01, 5.72e-01});
  // Matrix L(2, 2, {1.0, 0.0, 0.0, 1.0});
  Matrix Xi(1, 1, {7.98e+00});
  Matrix Psi(1, 1, {6.23e-01});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_config;
  etm_config.sigma = 1.0;
  etm_config.threshold = 0.0;
  etm_config.Psi = Psi;
  etm_config.Xi = Xi;

  // Chamada via runClosedLoopExtended para reter o objeto ExtendedClosedLoopResult
  auto result = engine.runClosedLoopExtended(
      x0, K, L, etm_config, sampling_period, duration, time_step, "SETM_EVENT_MAP", std::nullopt);

  fs::path dir = "simulations/lit-system-closed-loop-setm-event-map";
  fs::create_directories(dir);

  BinaryLogger::dump(dir / "time.bin", result.time_data);
  BinaryLogger::dump(dir / "trigger_times.bin", result.trigger_times);

  int num_steps = static_cast<int>(result.time_data.size());

  // 1. Exportação dos estados reais da planta (x1.bin, x2.bin, ...)
  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> state_trajectory;
    state_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      state_trajectory.push_back(result.states_data[step * state_dim + i]);
    std::string filename = "x" + std::to_string(i + 1) + ".bin";
    BinaryLogger::dump(dir / filename, state_trajectory);
  }

  // 2. Exportação dos estados estimados via mapa de eventos (x_est1.bin, x_est2.bin, ...)
  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> est_state_trajectory;
    est_state_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      est_state_trajectory.push_back(result.estimated_states_data[step * state_dim + i]);
    std::string filename = "x_est" + std::to_string(i + 1) + ".bin";
    BinaryLogger::dump(dir / filename, est_state_trajectory);
  }

  // 3. Exportação do erro de estimação (e1.bin, e2.bin, ...)
  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> error_trajectory;
    error_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      error_trajectory.push_back(result.estimation_error_data[step * state_dim + i]);
    std::string filename = "e" + std::to_string(i + 1) + ".bin";
    BinaryLogger::dump(dir / filename, error_trajectory);
  }

  // 4. Exportação dos sinais de controle (u1.bin, ...)
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

  std::cout << "Closed-loop SETM (Event Map) simulation completed. Data saved in: " << dir
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
    std::cout << "Number of states in the loaded LIT system: "
              << number_states << std::endl;

    ControlSystems::LITSystem *plant = engine.getPlant();
    std::cout << "Plant matrices:" << std::endl;
    std::cout << "A:\n"
              << plant->getA() << std::endl;
    std::cout << "B:\n"
              << plant->getB() << std::endl;
    std::cout << "C:\n"
              << plant->getC() << std::endl;

    // run_open_loop_simulation(engine);
    // run_closed_loop_setm_simulation(engine);
    run_closed_loop_setm_event_map_simulation(engine);
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Erro durante a execucao: " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}