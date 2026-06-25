#include <iostream>
#include <variant>
#include <filesystem>
#include <fstream>

#include "libs/control-core/include/SystemParser.hpp"
#include "libs/control-core/include/LITSystem.hpp"
#include "libs/control-core/include/BinaryLogger.hpp"
#include "libs/control-core/include/Algebra/Algebra.hpp"
#include "libs/control-core/include/EDOSolvers/EDOSolvers.hpp"

#include "include/PeriodicETC/LITEngine.hpp"

using namespace Algebra;
using PeriodicETC::LIT::Engine;

namespace fs = std::filesystem;

void matrices_test()
{
  Matrix A(3, 3);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;

  Vector x(3);

  x[0] = 1;
  x[1] = 2;
  x[2] = 3;

  Vector y = A * x;
  std::cout << "Matriz A:" << std::endl;
  std::cout << A << std::endl;
  std::cout << "Vetor x:" << std::endl;
  std::cout << x << std::endl;
  std::cout << "Vetor y = Ax:" << std::endl;
  std::cout << y << std::endl;
}

void system_creation_test()
{
  std::filesystem::path jsonPath = "system-datas/sys-02.json";
  SystemModel model = SystemParser::parseFromFile(jsonPath);
  std::cout << "System successfully parsed." << std::endl;

  if (model.A)
  {
    std::cout << "Matrix A:\n";
    std::cout << *model.A << std::endl;
  }

  if (model.B)
  {
    std::cout << "Matrix B:\n";
    std::cout << *model.B << std::endl;
  }

  if (model.C)
  {
    std::cout << "Matrix C:\n";
    std::cout << *model.C << std::endl;
  }
}

void algebra_expression_test()
{
  std::cout << "=== Algebra Expression Test ===" << std::endl;
  std::string expressionText = "0.4 + 0.6*p1";

  auto expression = Algebra::Parser::parse(expressionText);

  Algebra::Variables parameters;
  parameters["p1"] = 2.0;

  double result = expression->evaluate(parameters, 0.0);

  std::cout << "Expression: " << expressionText << std::endl;
  std::cout << "p1 = " << parameters["p1"] << std::endl;
  std::cout << "Result = " << result << std::endl;
}

void algebra_expression_test_2()
{
  struct TestCase
  {
    std::string expression;
    double expected;
  };

  std::vector<TestCase> tests = {
      {"1 + 2*3", 7.0},
      {"(1+2)*3", 9.0},
      {"2^3", 8.0},
      {"0.4 + 0.6*p1", 1.6}};

  Algebra::Variables vars;
  vars["p1"] = 2.0;

  for (const auto &test : tests)
  {
    auto expr = Algebra::Parser::parse(test.expression);
    double value = expr->evaluate(vars, 0.0);
    std::cout << test.expression << " = " << value << std::endl;
  }
}

void algebra_function_test()
{
  auto expr = Algebra::Parser::parse("sin(t)+2*cos(t)");
  Algebra::Variables vars;

  double result = expr->evaluate(vars, 1.0);
  std::cout << result << std::endl;
}

void algebra_function_test_2()
{
  auto expr = Algebra::Parser::parse("max(2*sin(t),p1)");
  Algebra::Variables vars;
  vars["p1"] = 0.5;
  std::cout << expr->evaluate(vars, 1.0) << std::endl;
}

void algebra_temporal_function()
{
  const auto timepts = Algebra::arange(0.0, 1.0, 1e-6);
  const auto expr = Algebra::Parser::parse("sin(t)+2*cos(t)");
  Algebra::Variables vars;

  std::vector<double> results;
  results.reserve(timepts.size());

  for (const auto t : timepts)
    results.push_back(expr->evaluate(vars, t));

  // Diretório
  const std::filesystem::path dir = "simulations/run-001";
  std::filesystem::create_directories(dir);

  // Salva tudo de forma genérica e concisa
  BinaryLogger::dump(dir / "time.bin", timepts);
  BinaryLogger::dump(dir / "values.bin", results);
}

void EDO_RK5_test()
{
  using Algebra::Matrix;
  using Algebra::Vector;

  Matrix A(2, 2);

  A(0, 0) = 0.0;
  A(0, 1) = 1.0;

  A(1, 0) = -2.0;
  A(1, 1) = -3.0;

  EDOSolvers::RK5 solver(
      [&A](double, const Vector &x, const Vector &)
      {
        return A * x;
      });

  Vector x(2);

  x[0] = 1.0;
  x[1] = 0.0;

  Vector u(1);

  const double dt = 0.001;

  auto timepts = Algebra::arange(0.0, 10.0, dt);

  std::vector<double> x1;
  std::vector<double> x2;

  x1.reserve(timepts.size());
  x2.reserve(timepts.size());

  for (const auto t : timepts)
  {
    x1.push_back(x[0]);
    x2.push_back(x[1]);
    x = solver.step(t, x, u, dt);
  }

  std::filesystem::path dir = "simulations/rk5-general";
  std::filesystem::create_directories(dir);

  BinaryLogger::dump(dir / "time.bin", timepts);
  BinaryLogger::dump(dir / "x1.bin", x1);
  BinaryLogger::dump(dir / "x2.bin", x2);
}

void EDO_RK45_test()
{
  using Algebra::Matrix;
  using Algebra::Vector;

  Matrix A(2, 2);

  A(0, 0) = 0.0;
  A(0, 1) = 1.0;

  A(1, 0) = -2.0;
  A(1, 1) = -3.0;

  EDOSolvers::RK45 solver(
      [&A](
          double,
          const Vector &x,
          const Vector &)
      {
        return A * x;
      });

  Vector x(2);

  x[0] = 1.0;
  x[1] = 0.0;

  Vector u(1);

  const double t0 = 0.0;
  const double tf = 10.0;

  double t = t0;
  double dt = 0.001;

  std::vector<double> time;
  std::vector<double> x1;
  std::vector<double> x2;

  std::vector<double> errors;
  std::vector<double> used_steps;
  std::vector<double> next_steps;
  std::vector<double> confidence;

  time.reserve(10000);
  x1.reserve(10000);
  x2.reserve(10000);

  errors.reserve(10000);
  used_steps.reserve(10000);
  next_steps.reserve(10000);
  confidence.reserve(10000);

  // Initial point

  time.push_back(t);
  x1.push_back(x[0]);
  x2.push_back(x[1]);

  std::size_t accepted = 0;
  std::size_t rejected = 0;

  while (t < tf)
  {
    if (t + dt > tf)
      dt = tf - t;

    auto result = solver.step(t, x, u, dt);
    errors.push_back(result.error_norm);
    next_steps.push_back(result.next_step);
    confidence.push_back(1.0 / (1.0 + result.error_norm));

    if (result.accepted)
    {
      x = result.state;
      t += result.dt_used;

      accepted++;

      time.push_back(t);
      x1.push_back(x[0]);
      x2.push_back(x[1]);

      used_steps.push_back(result.dt_used);
    }
    else
    {
      rejected++;
    }

    dt = result.next_step;
  }

  std::filesystem::path dir = "simulations/rk45-adaptive";
  std::filesystem::create_directories(dir);

  BinaryLogger::dump(dir / "time.bin", time);
  BinaryLogger::dump(dir / "x1.bin", x1);
  BinaryLogger::dump(dir / "x2.bin", x2);

  BinaryLogger::dump(dir / "dt.bin", used_steps);
  BinaryLogger::dump(dir / "next_dt.bin", next_steps);

  BinaryLogger::dump(dir / "error.bin", errors);
  BinaryLogger::dump(dir / "confidence.bin", confidence);

  std::cout
      << "RK45 simulation finished.\n"
      << "Accepted steps: "
      << accepted
      << "\nRejected steps: "
      << rejected
      << "\nTotal attempts: "
      << accepted + rejected
      << "\n";
}

void test_LITEngine_configuration()
{
  const std::filesystem::path json_path = "../experiments/data/sys-01.json";
  Engine engine;

  std::cout << "[TEST] Configuring LITEngine with path: " << json_path << std::endl;

  try
  {
    engine.configure(json_path);
    const auto &model = engine.getModel();

    if (!model.A.has_value() || !model.B.has_value())
    {
      throw std::runtime_error("LITEngine failed validation: A or B matrices missing.");
    }

    std::cout << "[TEST] SUCCESS: LITEngine configured with system: " << model.name << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << "[TEST] FAILED: " << e.what() << std::endl;
    throw; // Propagate to main for program termination
  }
}

int main()
{
  try
  {
    test_LITEngine_configuration();
    std::cout << "All tests passed successfully." << std::endl;
  }
  catch (...)
  {
    return 1;
  }
  return 0;
}