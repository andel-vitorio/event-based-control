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
#include "MTD.hpp"

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

struct MTDClosedLoopResults
{
  std::vector<double> time;
  std::vector<std::vector<double>> states;  // [passos_de_tempo x nx]
  std::vector<std::vector<double>> control; // [passos_de_tempo x nu]
  std::vector<double> event_times;          // Instantes de acionamento do ETM
  std::vector<int> active_modes;            // Histórico de chaveamento de modo do MTD
  std::vector<int> active_regions;          // Histórico de chaveamento de região do MTD
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

void test_parameters()
{
  using Matrix = Algebra::Matrix;
  using Vector = Algebra::Vector;
  std::filesystem::path jsonPath = "modules/control-systems/system-datas/sys-02.json";
  StateSystemModel model = StateSystemParser::parseFromFile(jsonPath);

  LPVSystem lpvSystem(model);

  // auto rho = lpvSystem.buildVariableParameter(0.0);
  Algebra::Variables rho;
  rho["p1"] = 0.5;
  Matrix A = lpvSystem.getMatrix("A", rho);

  std::cout << "Matriz A em t=0: " << std::endl;
  std::cout << A << std::endl;

  std::cout << "Simulação concluída com sucesso!" << std::endl;
}

MTDClosedLoopResults simulate_closed_loop_mtd(
    LPVSystem &lpvSystem,
    const Algebra::Vector &x0,
    double t_start,
    double t_end,
    double dt,
    double sampling_period,
    MTDManager &mtd,
    const std::vector<std::vector<Algebra::Matrix>> &K_table,
    const std::vector<std::vector<Algebra::Matrix>> &Xi_table,
    const std::vector<std::vector<Algebra::Matrix>> &Psi_table)
{
  using Vector = Algebra::Vector;
  using Matrix = Algebra::Matrix;

  MTDClosedLoopResults results;

  // Definição do Solver RK5 integrado ao sistema LPV[cite: 3]
  EDOSolvers::RK5 solver([&lpvSystem](double t, const Vector &x, const Vector &signal)
                         {
        std::size_t num_inputs = lpvSystem.n_inputs();
        Vector u = signal.slice(0, num_inputs);
        Vector w = signal.slice(num_inputs, signal.size());
        return lpvSystem.stateDerivative(x, u, w, t); });

  Vector x = x0;
  Vector w(0); // Perturbação vazia (ajuste se seu sistema usar perturbação externa)

  // Estado inicialmente transmitido
  Vector x_hat = x;

  PeriodicETC::Sampler sampler(sampling_period);

  // Determina modo e região iniciais em t = 0
  mtd.update(x);
  int current_mode = mtd.getCurrentMode();
  int current_region = mtd.getCurrentRegion();

  // Sinais iniciais de controle
  Matrix K_active = K_table[current_mode][current_region];
  Vector control_signal = K_active * x;

  // Geração do vetor de tempo[cite: 3]
  auto timepts = Algebra::arange(t_start, t_end, dt);

  // Pré-alocação de memória para evitar gargalos de realocação
  results.time.reserve(timepts.size());
  results.states.reserve(timepts.size());
  results.control.reserve(timepts.size());
  results.active_modes.reserve(timepts.size());
  results.active_regions.reserve(timepts.size());

  // O primeiro instante de amostragem sempre registra um acionamento (inicialização)
  results.event_times.push_back(t_start);

  for (const auto t : timepts)
  {
    // 1. Armazenamento dos dados do passo atual
    results.time.push_back(t);

    std::vector<double> x_step(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      x_step[i] = x[i];
    results.states.push_back(x_step);

    std::vector<double> u_step(control_signal.size());
    for (size_t i = 0; i < control_signal.size(); ++i)
      u_step[i] = control_signal[i];
    results.control.push_back(u_step);

    results.active_modes.push_back(current_mode);
    results.active_regions.push_back(current_region);

    // 2. Lógica de Amostragem Periódica
    if (sampler.shouldSample(t))
    {
      // O MTD analisa o estado atual e decide a comutação proativa
      mtd.update(x);
      current_mode = mtd.getCurrentMode();
      current_region = mtd.getCurrentRegion();

      // Carrega as matrizes correspondentes ao novo cenário ativo
      K_active = K_table[current_mode][current_region];
      Matrix Xi_active = Xi_table[current_mode][current_region];
      Matrix Psi_active = Psi_table[current_mode][current_region];

      // Avaliação do erro de transmissão
      Vector error = x_hat - x;

      // Condição elipsoidal adaptativa: val = x^T * Psi * x - e^T * Xi * e
      double x_term = Algebra::Vector::dot(x.T(), Psi_active * x);
      double error_term = Algebra::Vector::dot(error.T(), Xi_active * error);
      double val = x_term - error_term;

      // Se val < 0, a discrepância de erro excedeu o limiar de estabilidade
      if (val < 0)
      {
        x_hat = x;                        // Transmite o estado atualizado
        control_signal = K_active * x;    // Atualiza a ação de controle
        results.event_times.push_back(t); // Registra o instante do evento
      }
    }

    // 3. Integração Numérica do Sistema LPV para o próximo passo[cite: 3]
    Vector signal = Vector::concatenate(control_signal, w);
    x = solver.step(t, x, signal, dt);
  }

  return results;
}

void run_mtd_simulation()
{
  using Matrix = Algebra::Matrix;
  using Vector = Algebra::Vector;

  // 1. Configuração de Dimensões Adaptativas
  const int n_modes = 2;   // Modo 1 (index 0), Modo 2 (index 1)
  const int n_regions = 2; // Região 0, Região 1

  // 2. Alocação de Tabelas Dinâmicas para ETMs e Controladores
  std::vector<std::vector<Matrix>> K_table(n_modes, std::vector<Matrix>(n_regions));
  std::vector<std::vector<Matrix>> Xi_table(n_modes, std::vector<Matrix>(n_regions));
  std::vector<std::vector<Matrix>> Psi_table(n_modes, std::vector<Matrix>(n_regions));

  // --- PREENCHIMENTO DOS DADOS DOS ANEXOS ---

  // MODE 1 (Index 0) - Region 0
  K_table[0][0] = Matrix(1, 2, {-2.71, -4.30});
  Xi_table[0][0] = Matrix(2, 2, {2.74e5, 4.15e5, 4.15e5, 6.52e5});
  Psi_table[0][0] = Matrix(2, 2, {6.05e4, 3.59e4, 3.59e4, 9.52e4});

  // MODE 1 (Index 0) - Region 1
  K_table[0][1] = Matrix(1, 2, {-2.71, -4.30});
  Xi_table[0][1] = Matrix(2, 2, {2.12e4, 1.36e4, 1.36e4, 1.64e4});
  Psi_table[0][1] = Matrix(2, 2, {1.00e6, -3.01, -3.01, 1.00e6});

  // MODE 2 (Index 1) - Region 0
  K_table[1][0] = Matrix(1, 2, {-2.71, -4.30});
  Xi_table[1][0] = Matrix(2, 2, {2.74e5, 4.15e5, 4.15e5, 6.53e5});
  Psi_table[1][0] = Matrix(2, 2, {6.06e4, 3.61e4, 3.61e4, 9.55e4});

  // MODE 2 (Index 1) - Region 1
  K_table[1][1] = Matrix(1, 2, {-2.65, -4.20});
  Xi_table[1][1] = Matrix(2, 2, {5.31e5, 8.21e5, 8.21e5, 1.30e6});
  Psi_table[1][1] = Matrix(2, 2, {3.14e4, 1.94e4, 1.94e4, 5.02e4});

  // 3. Inicialização do MTDManager
  MTDManager mtd;
  mtd.setSeed(12345);

  // Define a matriz de peso de energia baseada no P global dos anexos
  Matrix W = Matrix(2, 2, {1.59e5, 5.61e4, 5.61e4, 9.20e4});
  mtd.setEnergyWeightMatrix(W);

  // Limiares de energia (Defina o limite que separa as regiões)
  double c_data[] = {5.0e4, 1.0e9}; // Exemplo de região [0, c0] e [c0, inf]
  mtd.setEnergyThresholds(c_data, n_regions);

  // Tensor de transição Pi das imagens (achatado na memória em row-major)
  double pi_data[] = {
      // Região 0
      0.15, 0.85,
      0.89, 0.11,
      // Região 1
      0.50, 0.50,
      0.21, 0.79};
  mtd.setTransitionProbabilities(pi_data, n_modes, n_regions);

  // 4. Executa a simulação
  Vector x0(2);
  x0[0] = .5;
  x0[1] = -.75;

  std::filesystem::path jsonPath = "modules/control-systems/system-datas/sys-02.json";
  StateSystemModel model = StateSystemParser::parseFromFile(jsonPath);

  LPVSystem lpvSystem(model);

  MTDClosedLoopResults sim_results = simulate_closed_loop_mtd(
      lpvSystem, x0, 0.0, 20.0, 0.0001, 0.1, mtd, K_table, Xi_table, Psi_table);

  // 2. Processamento Adaptativo para Log
  // Extraímos o número de estados do resultado
  int nx = (int)sim_results.states[0].size();
  std::vector<std::vector<double>> state_cols(nx);

  // Preenche os vetores de cada coluna de estado
  for (const auto &step_states : sim_results.states)
  {
    for (int i = 0; i < nx; ++i)
    {
      state_cols[i].push_back(step_states[i]);
    }
  }

  // 3. Logger
  std::filesystem::path dir = "simulations/lpv-system-mtd-closed-loop";
  std::filesystem::create_directories(dir);

  // Dump dos tempos
  BinaryLogger::dump(dir / "time.bin", sim_results.time);

  // Dump dinâmico dos estados (gera x1.bin, x2.bin, ..., xN.bin)
  for (int i = 0; i < nx; ++i)
  {
    std::string filename = "x" + std::to_string(i + 1) + ".bin";
    BinaryLogger::dump(dir / filename, state_cols[i]);
  }

  // Dica: Dump do histórico de chaveamento (Essencial para debugar a instabilidade)
  // Converter vector<int> para vector<double> para o BinaryLogger, se necessário
  std::vector<double> modes_d(sim_results.active_modes.begin(), sim_results.active_modes.end());
  std::vector<double> regions_d(sim_results.active_regions.begin(), sim_results.active_regions.end());
  BinaryLogger::dump(dir / "modes.bin", modes_d);
  BinaryLogger::dump(dir / "regions.bin", regions_d);

  std::cout << "Simulação MTD concluída e logs gerados em: " << dir << std::endl;
}

int main()
{
  run_mtd_simulation();
  return 0;
}