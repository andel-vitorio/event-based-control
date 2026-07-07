#include <iostream>
#include <variant>
#include <filesystem>
#include <fstream>

#include "StateSystemModel.hpp"
#include "StateSystemParser.hpp"
#include "BinaryLogger.hpp"
#include "LITSystem.hpp"
#include "LPVSystem.hpp"
#include "Algebra/Algebra.hpp"
#include "EDOSolvers/EDOSolvers.hpp"

using namespace Algebra;
using namespace ControlSystems;
namespace fs = std::filesystem;

void simulate_lit_open_loop()
{
  // using Algebra::Matrix;
  using Algebra::Vector;

  Matrix A(2, 2);
  A(0, 0) = 0.0;
  A(0, 1) = 1.0;
  A(1, 0) = -2.0;
  A(1, 1) = -3.0;

  Matrix B(2, 1);
  B(0, 0) = 0.0;
  B(1, 0) = 1.0;

  LITSystem::Configuration config;
  config.matrices.A = A;
  config.matrices.B = B;

  LITSystem system(config);

  Vector u(1);
  u[0] = 0.0;
  Vector w(0);

  EDOSolvers::RK5 solver([&system](double /*t*/, const Vector &x, const Vector &signal)
                         {
      std::size_t num_inputs = system.inputs();
      Vector u = signal.slice(0, num_inputs);
      Vector w = signal.slice(num_inputs, signal.size());
      return system.stateDerivative(x, u, w); });

  Vector x(2);
  x[0] = 1.0;
  x[1] = 0.0; // Condições iniciais
  const double dt = 0.001;
  auto timepts = Algebra::arange(0.0, 10.0, dt);

  std::vector<double> x1, x2;
  x1.reserve(timepts.size());
  x2.reserve(timepts.size());

  for (const auto t : timepts)
  {
    x1.push_back(x[0]);
    x2.push_back(x[1]);

    Vector signal = Vector::concatenate(u, w);
    x = solver.step(t, x, signal, dt);
  }

  std::cout << "Simulação concluída com sucesso!" << std::endl;
}

void simulate()
{
  // using Matrix = Algebra::Matrix;
  using Vector = Algebra::Vector;
  std::filesystem::path jsonPath = "modules/control-systems/system-datas/sys-03.json";
  StateSystemModel model = StateSystemParser::parseFromFile(jsonPath);

  for (const auto &[name, var] : model.states)
  {
    std::cout << "State: " << name << ", Symbol: " << var.symbol << ", Unit: " << var.unit;
    if (var.value.has_value())
      std::cout << ", Value: " << var.value.value();
    std::cout << std::endl;
  }

  for (const auto &[name, var] : model.parameters)
  {
    std::cout << "Parameter: " << name << ", Symbol: " << var.symbol << ", Unit: " << var.unit;
    if (var.value.has_value())
      std::cout << ", Value: " << var.value.value();
    std::cout << std::endl;
  }

  for (const auto &[name, var] : model.disturbances)
  {
    std::cout << "Disturbance: " << name << ", Symbol: " << var.symbol << ", Unit: " << var.unit;
    if (var.value.has_value())
      std::cout << ", Value: " << var.value.value();
    std::cout << std::endl;
  }

  LPVSystem lpvSystem(model);
  Vector u(1);
  u[0] = 0.0;
  Vector w(0);

  EDOSolvers::RK5 solver([&lpvSystem](double t, const Vector &x, const Vector &signal)
                         {
      std::size_t num_inputs = lpvSystem.n_inputs();
      Vector u = signal.slice(0, num_inputs);
      Vector w = signal.slice(num_inputs, signal.size());
      return lpvSystem.stateDerivative(x, u, w, t); });

  Vector x(2);
  x[0] = 1.0;
  x[1] = 0.0;
  const double dt = 0.001;
  auto timepts = Algebra::arange(0.0, 10.0, dt);

  std::vector<double> x1, x2;
  x1.reserve(timepts.size());
  x2.reserve(timepts.size());

  for (const auto t : timepts)
  {
    x1.push_back(x[0]);
    x2.push_back(x[1]);

    Vector signal = Vector::concatenate(u, w);
    x = solver.step(t, x, signal, dt);
  }

  std::filesystem::path dir = "simulations/lpv-system-open-loop";
  std::filesystem::create_directories(dir);
  BinaryLogger::dump(dir / "time.bin", timepts);
  BinaryLogger::dump(dir / "x1.bin", x1);
  BinaryLogger::dump(dir / "x2.bin", x2);

  std::cout << "Simulação concluída com sucesso!" << std::endl;
}

int main()
{
  simulate();
  return 0;
}