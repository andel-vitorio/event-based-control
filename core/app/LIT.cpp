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

#include "PeriodicETC/LIT/lit_engine.hpp"

using namespace Algebra;
using namespace ControlSystems;
namespace fs = std::filesystem;
namespace LIT = PeriodicETC::LIT;

auto null = std::nullopt;

void run_open_loop_simulation(LIT::Engine &engine)
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

int main()
{
  // std::string systems_directory = "modules/control-systems/system-datas/";
  std::string systems_directory = "../experiments/data/";
  fs::path jsonPath = systems_directory + "sys-01.json";

  try
  {
    LIT::Engine engine;
    engine.loadSystem(jsonPath.string());

    int number_states = engine.getStateDim();
    std::cout << "Number of states in the loaded LIT system: " << number_states << std::endl;

    run_open_loop_simulation(engine);
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Erro durante a execucao: " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}