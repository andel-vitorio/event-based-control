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
#include "PeriodicETM.hpp"

using namespace Algebra;
using namespace ControlSystems;
namespace fs = std::filesystem;

struct SimulationStats
{
  std::vector<double> event_times;
  int total_samples = 0;

  void recordEvent(double t)
  {
    event_times.push_back(t);
  }

  void printReport(double t_start, double t_end) const
  {
    if (event_times.size() < 2)
    {
      std::cout << "Eventos insuficientes para cálculo de IET." << std::endl;
      return;
    }

    std::vector<double> iets;
    for (size_t i = 1; i < event_times.size(); ++i)
    {
      iets.push_back(event_times[i] - event_times[i - 1]);
    }

    double sum = std::accumulate(iets.begin(), iets.end(), 0.0);
    double mean = sum / iets.size();

    double sq_sum = std::inner_product(iets.begin(), iets.end(), iets.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / iets.size() - mean * mean);

    double min_iet = *std::min_element(iets.begin(), iets.end());
    double max_iet = *std::max_element(iets.begin(), iets.end());

    std::cout << "\n--- Resultados da Simulação ---" << std::endl;
    std::cout << "Estado Inicial: [ 1. -1. ]" << std::endl; // Hardcoded ou extraia do vetor
    std::cout << "Número de amostras dos estados: " << total_samples << std::endl;
    std::cout << "Número de transmissões de estados realizados: " << event_times.size() << std::endl;
    std::cout << "Menor IET Obtido: " << min_iet << "s" << std::endl;
    std::cout << "Máximo IET Obtido: " << max_iet << "s" << std::endl;
    std::cout << "Média dos Intervalos de Tempo (IET): " << mean << "s" << std::endl;
    std::cout << "Desvio-padrão dos IETs: " << stdev << std::endl;
    std::cout << "Coeficiente de variação dos IETs: " << (mean != 0 ? stdev / mean : 0.0) << std::endl;
    std::cout << "Taxa média de acionamentos: " << (event_times.size() / (t_end - t_start)) << " eventos/s" << std::endl;
  }
};

void simulate()
{
  using Matrix = Algebra::Matrix;
  using Vector = Algebra::Vector;
  std::filesystem::path jsonPath = "modules/control-systems/system-datas/sys-02.json";
  StateSystemModel model = StateSystemParser::parseFromFile(jsonPath);

  LPVSystem lpvSystem(model);
  Vector w(0);

  EDOSolvers::RK5 solver([&lpvSystem](double t, const Vector &x, const Vector &signal)
                         {
      std::size_t num_inputs = lpvSystem.n_inputs();
      Vector u = signal.slice(0, num_inputs);
      Vector w = signal.slice(num_inputs, signal.size());
      return lpvSystem.stateDerivative(x, u, w, t); });

  Vector x(2);
  x[0] = 1.0;
  x[1] = -1.0;
  const double dt = 0.0001;
  auto timepts = Algebra::arange(0.0, 20.0, dt);

  std::vector<double> x1, x2;
  x1.reserve(timepts.size());
  x2.reserve(timepts.size());

  PeriodicETC::Sampler sampler(0.1);
  Matrix K(1, 2);
  K(0, 0) = -2.71;
  K(0, 1) = -4.3;

  Matrix Xi(2, 2);
  Xi(0, 0) = 2.74e5;
  Xi(0, 1) = 4.15e5;
  Xi(1, 0) = 4.15e5;
  Xi(1, 1) = 6.53e5;

  Matrix Psi(2, 2);
  Psi(0, 0) = 6.06e4;
  Psi(0, 1) = 3.61e4;
  Psi(1, 0) = 3.61e4;
  Psi(1, 1) = 9.54e4;

  Vector control_signal = K * x;

  PeriodicETC::StaticSETM etm(Xi, Psi);
  SimulationStats stats;

  for (const auto t : timepts)
  {
    x1.push_back(x[0]);
    x2.push_back(x[1]);

    if (sampler.shouldSample(t) and etm.evaluate(x))
    {
      std::cout << "Evento disparado em t = " << t << "s" << std::endl;
      stats.recordEvent(t);
      control_signal = K * x;
    }

    Vector signal = Vector::concatenate(control_signal, w);
    x = solver.step(t, x, signal, dt);
  }

  stats.printReport(timepts.front(), timepts.back());

  // std::filesystem::path dir = "simulations/lpv-system-open-loop";
  // std::filesystem::create_directories(dir);
  // BinaryLogger::dump(dir / "time.bin", timepts);
  // BinaryLogger::dump(dir / "x1.bin", x1);
  // BinaryLogger::dump(dir / "x2.bin", x2);

  std::cout << "Simulação concluída com sucesso!" << std::endl;
}

int main()
{
  simulate();
  return 0;
}