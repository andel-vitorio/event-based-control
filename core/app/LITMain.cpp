#include <iostream>
#include <variant>
#include <filesystem>
#include <fstream>
#include <vector>
#include <random>

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

class TemporaryOutputNoise
{
public:
  enum class Type
  {
    GAUSSIAN,
    BIAS,
    SINUSOIDAL
  };

  TemporaryOutputNoise(
      int ny,
      double t_start,
      double t_end,
      double amplitude = 1.0,
      Type type = Type::GAUSSIAN,
      unsigned int seed = 42)
      : ny_(ny), t_start_(t_start), t_end_(t_end),
        amplitude_(amplitude), type_(type), rng_(seed), dist_(0.0, amplitude) {}

  Algebra::Vector operator()(double t)
  {
    // Fora da janela de ativação, o ruído é estritamente zero
    if (t < t_start_ || t > t_end_)
    {
      return Algebra::Vector(ny_);
    }

    Algebra::Vector alpha(ny_);
    switch (type_)
    {
    case Type::GAUSSIAN:
      for (int i = 0; i < ny_; ++i)
      {
        alpha[i] = dist_(rng_);
      }
      break;
    case Type::BIAS:
      for (int i = 0; i < ny_; ++i)
      {
        alpha[i] = amplitude_;
      }
      break;
    case Type::SINUSOIDAL:
    {
      double val = amplitude_ * std::sin(2.0 * 3.141592 * 1.0 * t);
      for (int i = 0; i < ny_; ++i)
      {
        alpha[i] = val;
      }
      break;
    }
    }
    return alpha;
  }

private:
  int ny_;
  double t_start_;
  double t_end_;
  double amplitude_;
  Type type_;
  std::mt19937 rng_;
  std::normal_distribution<double> dist_;
};

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
  BinaryLogger::dump(dir / "trigger_times.bin", result.sc_trigger_times);

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
            << " | Total triggers: " << result.sc_trigger_times.size() << std::endl;
}

void run_dual_channel_closed_loop_setm_simulation(PeriodicETC::LITEngine &engine)
{
  using Algebra::Matrix;
  using Algebra::Vector;
  namespace fs = std::filesystem;

  // ------------------------------------------------------------------
  // 1. Parâmetros Temporais
  // ------------------------------------------------------------------
  const double duration = 30.0;
  const double time_step = 1e-4;
  const double sampling_period = 1e-3;

  // ------------------------------------------------------------------
  // 2. Dimensões e Condições Iniciais
  // ------------------------------------------------------------------
  const int state_dim = static_cast<int>(engine.getStateDim());
  const int input_dim = 1;
  const double max_iet_sc = 1000 * sampling_period;
  const double max_iet_ca = 10.0;

  Vector x0(state_dim);
  x0[0] = 1.0;
  x0[1] = 1.0;

  // Estimador Principal \hat{x}(0)
  Vector x_hat0(state_dim);
  x_hat0[0] = 0.0;
  x_hat0[1] = 0.0;

  // Estimador Auxiliar Contínuo \hat{x}_a(0)
  Vector x_hat_a0(state_dim);
  x_hat_a0[0] = 0.0;
  x_hat_a0[1] = 0.0;

  // ------------------------------------------------------------------
  // 3. Ganhos de Controle e do Observador Aumentado (2nx)
  // ------------------------------------------------------------------
  // Ganho do Controlador Nominal K \in R^{1 x 2}
  Matrix K(1, 2, {-2.52e+01, -1.85e+01});

  // Ganho de Amortecimento Interno no Fluxo L0 \in R^{2 x 2} (atua em ker(C))
  Matrix L0(2, 2, {-1.14e+00, 2.28e-01, 2.28e-01, -4.56e-02});

  // Ganho Impulsivo de Salto L1 \in R^{2 x 1} (síntese LMI multimodo)
  Matrix L1(2, 1, {9.39e-01, 1.78e+00});
  // Matrix L1(2, 1, {1.0, 0.0});

  // Ganho de Luenberger Contínuo Virtual L2 \in R^{2 x 1}
  Matrix L2(2, 1, {8.14e+00, 1.04e+01});

  // ------------------------------------------------------------------
  // 4. Configuração do ETM Sensor-Controlador (Canal SC: n_y = 1)
  // ------------------------------------------------------------------
  Matrix Xi_sc(1, 1, {1.06e+01});
  Matrix Psi_sc(1, 1, {9.16e-02});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_sc_config;
  etm_sc_config.sigma = 1.0;
  etm_sc_config.threshold = 0.0;
  etm_sc_config.Psi = Psi_sc;
  etm_sc_config.Xi = Xi_sc;

  // ------------------------------------------------------------------
  // 5. Configuração do ETM Controlador-Atuador (Canal CA: n_x = 2)
  // ------------------------------------------------------------------
  Matrix Xi_ca(2, 2, {2.95e+06, -1.26e+06, -1.26e+06, 5.38e+05});
  Matrix Psi_ca(2, 2, {9.23e+04, -2.91e+04, -2.91e+04, 1.19e+04});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_ca_config;
  etm_ca_config.sigma = 1.0;
  etm_ca_config.threshold = 0.0;
  etm_ca_config.Psi = Psi_ca;
  etm_ca_config.Xi = Xi_ca;

  // ------------------------------------------------------------------
  // 6. Execução da Simulação com Arquitetura Aumentada
  // ------------------------------------------------------------------
  auto result = engine.runDualChannelClosedLoop(
      x0, x_hat0, x_hat_a0,
      K, L0, L1, L2,
      etm_sc_config, etm_ca_config,
      sampling_period, duration, time_step,
      "DUAL_CHANNEL_SETM",
      std::nullopt, max_iet_sc, max_iet_ca);

  // ------------------------------------------------------------------
  // 7. Exportação dos Dados Binários
  // ------------------------------------------------------------------
  fs::path dir = "simulations/lit-system-dual-channel-setm";
  fs::create_directories(dir);

  BinaryLogger::dump(dir / "time.bin", result.time_data);
  BinaryLogger::dump(dir / "sc_trigger_times.bin", result.sc_trigger_times);
  BinaryLogger::dump(dir / "ca_trigger_times.bin", result.ca_trigger_times);

  const int num_steps = static_cast<int>(result.time_data.size());

  // (a) Estados reais da planta (x1.bin, x2.bin, ...)
  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> state_trajectory;
    state_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
    {
      state_trajectory.push_back(result.states_data[step * state_dim + i]);
    }
    BinaryLogger::dump(dir / ("x" + std::to_string(i + 1) + ".bin"), state_trajectory);
  }

  // (b) Estados estimados pelo observador principal (x_est1.bin, x_est2.bin, ...)
  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> est_state_trajectory;
    est_state_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
    {
      est_state_trajectory.push_back(result.estimated_states_data[step * state_dim + i]);
    }
    BinaryLogger::dump(dir / ("x_est" + std::to_string(i + 1) + ".bin"), est_state_trajectory);
  }

  // (c) Erro de estimação principal: e = x - \hat{x} (e1.bin, e2.bin, ...)
  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> error_trajectory;
    error_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
    {
      error_trajectory.push_back(result.estimation_error_data[step * state_dim + i]);
    }
    BinaryLogger::dump(dir / ("e" + std::to_string(i + 1) + ".bin"), error_trajectory);
  }

  // (d) Sinal de controle aplicado pelo ZOH (u1.bin, ...)
  for (int i = 0; i < input_dim; ++i)
  {
    std::vector<double> control_trajectory;
    control_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
    {
      control_trajectory.push_back(result.control_data[step * input_dim + i]);
    }
    BinaryLogger::dump(dir / ("u" + std::to_string(i + 1) + ".bin"), control_trajectory);
  }

  // ------------------------------------------------------------------
  // 8. Relatório da Simulação
  // ------------------------------------------------------------------
  const double total_samples = duration / sampling_period;
  const double sc_reduction = (1.0 - static_cast<double>(result.sc_trigger_times.size()) / total_samples) * 100.0;
  const double ca_reduction = (1.0 - static_cast<double>(result.ca_trigger_times.size()) / total_samples) * 100.0;

  std::cout << "\n======================================================\n"
            << "Simulação Dual-Channel SETM (Ordem Aumentada 2nx)\n"
            << "Diretório de saída: " << dir << "\n"
            << "------------------------------------------------------\n"
            << "Total de amostras periódicas : " << static_cast<int>(total_samples) << "\n"
            << "Disparos no canal SC (Sensor) : " << result.sc_trigger_times.size()
            << " (Redução: " << sc_reduction << "%)\n"
            << "Disparos no canal CA (Atuador): " << result.ca_trigger_times.size()
            << " (Redução: " << ca_reduction << "%)\n"
            << "======================================================\n"
            << std::endl;
}

void run_observer_petc_closed_loop_simulation(PeriodicETC::LITEngine &engine)
{
  using Algebra::Matrix;
  using Algebra::Vector;
  namespace fs = std::filesystem;

  const double duration = 7.5;
  const double time_step = 1e-4;
  const double sampling_period = 1e-1;

  const int state_dim = static_cast<int>(engine.getStateDim());
  const int input_dim = static_cast<int>(engine.getInputDim());
  const double max_iet = 10.0;

  Vector x0(state_dim);
  x0[0] = 1.0;
  x0[1] = 1.0;

  Vector x_hat0(state_dim);
  x_hat0[0] = 0.0;
  x_hat0[1] = 0.0;

  Matrix K(1, 2, {8.38e-01, -2.35e+00});
  Matrix L(2, 1, {3.01e+02, 1.70e+03});
  Matrix Xi(2, 2, {2.93e+03, -7.19e+03, -7.19e+03, 2.04e+04});
  Matrix Psi(2, 2, {5.65e+03, 6.83e+02, 6.83e+02, 6.72e+03});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_config;
  etm_config.sigma = 1.0;
  etm_config.threshold = 0.0;
  etm_config.Psi = Psi;
  etm_config.Xi = Xi;

  auto result = engine.runObserverPETCClosedLoop(
      x0, x_hat0, K, L, etm_config, sampling_period, duration, time_step, "OBSERVER_PETC", std::nullopt, max_iet);

  fs::path dir = "simulations/lit-system-observer-petc";
  fs::create_directories(dir);

  BinaryLogger::dump(dir / "time.bin", result.time_data);
  BinaryLogger::dump(dir / "ca_trigger_times.bin", result.ca_trigger_times);

  const int num_steps = static_cast<int>(result.time_data.size());

  // (a) Trajetória dos estados reais da planta (x1.bin, x2.bin, ...)
  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> state_trajectory;
    state_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
    {
      state_trajectory.push_back(result.states_data[step * state_dim + i]);
    }
    BinaryLogger::dump(dir / ("x" + std::to_string(i + 1) + ".bin"), state_trajectory);
  }

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> est_state_trajectory;
    est_state_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
    {
      est_state_trajectory.push_back(result.estimated_states_data[step * state_dim + i]);
    }
    BinaryLogger::dump(dir / ("x_est" + std::to_string(i + 1) + ".bin"), est_state_trajectory);
  }

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> error_trajectory;
    error_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
    {
      error_trajectory.push_back(result.estimation_error_data[step * state_dim + i]);
    }
    BinaryLogger::dump(dir / ("e" + std::to_string(i + 1) + ".bin"), error_trajectory);
  }

  for (int i = 0; i < input_dim; ++i)
  {
    std::vector<double> control_trajectory;
    control_trajectory.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
    {
      control_trajectory.push_back(result.control_data[step * input_dim + i]);
    }
    BinaryLogger::dump(dir / ("u" + std::to_string(i + 1) + ".bin"), control_trajectory);
  }

  const double total_samples = duration / sampling_period;
  const std::size_t num_transmissions = result.ca_trigger_times.size();
  const double network_reduction = (1.0 - static_cast<double>(num_transmissions) / total_samples) * 100.0;

  double sum_sq_error = 0.0;
  double max_norm_error = 0.0;
  for (int step = 0; step < num_steps; ++step)
  {
    double current_norm_sq = 0.0;
    for (int i = 0; i < state_dim; ++i)
    {
      const double err = result.estimation_error_data[step * state_dim + i];
      current_norm_sq += err * err;
    }
    sum_sq_error += current_norm_sq;
    const double current_norm = std::sqrt(current_norm_sq);
    if (current_norm > max_norm_error)
    {
      max_norm_error = current_norm;
    }
  }
  const double rms_norm_error = std::sqrt(sum_sq_error / static_cast<double>(num_steps));

  std::cout << "\n======================================================\n"
            << "Simulação PETC com Observador Contínuo de Luenberger\n"
            << "Diretório de saída: " << dir << "\n"
            << "------------------------------------------------------\n"
            << "Duração da simulação         : " << duration << " s\n"
            << "Período base de amostragem h : " << sampling_period << " s\n"
            << "Passo de integração RK5 dt   : " << time_step << " s\n"
            << "Total de avaliações do ETM   : " << static_cast<int>(total_samples) << "\n"
            << "Transmissões pela rede (PETC): " << num_transmissions
            << " (Economia: " << network_reduction << "%)\n"
            << "Erro de Estimação - Norma Máxima : " << max_norm_error << "\n"
            << "Erro de Estimação - Norma RMS    : " << rms_norm_error << "\n"
            << "======================================================\n"
            << std::endl;
}

void run_observer_petc_under_attack_simulation(PeriodicETC::LITEngine &engine)
{
  using Algebra::Matrix;
  using Algebra::Vector;
  namespace fs = std::filesystem;

  const double duration = 10.0;
  const double time_step = 1e-4;
  const double sampling_period = 1e-1;

  const int state_dim = static_cast<int>(engine.getStateDim());
  const int output_dim = static_cast<int>(engine.getOutputDim());
  const int input_dim = static_cast<int>(engine.getInputDim());

  const double max_iet_sc = 5.0 * sampling_period;
  const double max_iet_ca = 10.0;
  const double detection_threshold = 1e-3;

  Vector x0(state_dim);
  x0[0] = -1.0;
  x0[1] = 1.0;

  Vector x_hat0(state_dim);
  x_hat0[0] = 0.0;
  x_hat0[1] = 0.0;

  Vector x_hat_a0(state_dim);
  x_hat_a0[0] = 0.0;
  x_hat_a0[1] = 0.0;

  Matrix K(1, 2, {1.39e+00, -3.38e+00});
  Matrix L0(2, 2, {0.00e+00, 0.00e+00, 0.00e+00, -1.50e+00});
  Matrix L1(2, 1, {9.83e-01, -1.13e-01});
  Matrix L2(2, 1, {8.40e+00, 2.46e+01});

  Matrix Xi_sc(1, 1, {8.66e-01});
  Matrix Psi_sc(1, 1, {1.13e+00});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_sc_config;
  etm_sc_config.sigma = 1.0;
  etm_sc_config.threshold = 0.0;
  etm_sc_config.Psi = Psi_sc;
  etm_sc_config.Xi = Xi_sc;

  Matrix Xi_ca(2, 2, {2.73e+05, -5.26e+05, -5.26e+05, 1.30e+06});
  Matrix Psi_ca(2, 2, {1.13e+05, 2.74e+04, 2.74e+04, 1.38e+05});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_ca_config;
  etm_ca_config.sigma = 1.0;
  etm_ca_config.threshold = 0.0;
  etm_ca_config.Psi = Psi_ca;
  etm_ca_config.Xi = Xi_ca;

  TemporaryOutputNoise attack_noise(
      output_dim, 5.0, 6.0, 0.1, TemporaryOutputNoise::Type::BIAS);

  auto result = engine.runDualChannelUnderAttackClosedLoop(
      x0, x_hat0, x_hat_a0,
      K, L0, L1, L2,
      etm_sc_config, etm_ca_config,
      sampling_period, duration, time_step,
      "DUAL_CHANNEL_ATTACK",
      std::nullopt, max_iet_sc, max_iet_ca,
      attack_noise, detection_threshold);

  fs::path dir = "simulations/lit-system-dual-channel-under-attack";
  fs::create_directories(dir);

  const int num_steps = static_cast<int>(result.time_data.size());
  const std::size_t sc_transmissions = result.sc_trigger_times.size();

  BinaryLogger::dump(dir / "time.bin", result.time_data);
  BinaryLogger::dump(dir / "sc_trigger_times.bin", result.sc_trigger_times);
  BinaryLogger::dump(dir / "ca_trigger_times.bin", result.ca_trigger_times);
  BinaryLogger::dump(dir / "residual_norm.bin", result.residual_norm);

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.states_data[step * state_dim + i]);
    BinaryLogger::dump(dir / ("x" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.estimated_states_data[step * state_dim + i]);
    BinaryLogger::dump(dir / ("x_est" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.estimation_error_data[step * state_dim + i]);
    BinaryLogger::dump(dir / ("e" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < input_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.control_data[step * input_dim + i]);
    BinaryLogger::dump(dir / ("u" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < output_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.residual[step * output_dim + i]);
    BinaryLogger::dump(dir / ("r" + std::to_string(i + 1) + ".bin"), traj);
  }

  std::vector<double> alarm_traj;
  alarm_traj.reserve(result.alarm_active.size());
  for (bool val : result.alarm_active)
    alarm_traj.push_back(val ? 1.0 : 0.0);
  BinaryLogger::dump(dir / "alarm_active.bin", alarm_traj);

  for (int i = 0; i < output_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(sc_transmissions);
    for (std::size_t k = 0; k < sc_transmissions; ++k)
      traj.push_back(result.malicious_signal[k * output_dim + i]);
    BinaryLogger::dump(dir / ("malicious_attack" + std::to_string(i + 1) + ".bin"), traj);
  }

  const double total_samples = duration / sampling_period;
  const std::size_t ca_transmissions = result.ca_trigger_times.size();

  const double sc_reduction =
      (total_samples > 0.0)
          ? (1.0 - static_cast<double>(sc_transmissions) / total_samples) * 100.0
          : 0.0;
  const double ca_reduction =
      (total_samples > 0.0)
          ? (1.0 - static_cast<double>(ca_transmissions) / total_samples) * 100.0
          : 0.0;

  double sc_iet_min = 0.0, sc_iet_max = 0.0, sc_iet_mean = 0.0;
  if (sc_transmissions > 1)
  {
    sc_iet_min = result.sc_trigger_times[1] - result.sc_trigger_times[0];
    sc_iet_max = sc_iet_min;
    double sum = 0.0;
    for (std::size_t k = 1; k < sc_transmissions; ++k)
    {
      const double dt_event = result.sc_trigger_times[k] - result.sc_trigger_times[k - 1];
      sum += dt_event;
      if (dt_event < sc_iet_min)
        sc_iet_min = dt_event;
      if (dt_event > sc_iet_max)
        sc_iet_max = dt_event;
    }
    sc_iet_mean = sum / static_cast<double>(sc_transmissions - 1);
  }

  double ca_iet_min = 0.0, ca_iet_max = 0.0, ca_iet_mean = 0.0;
  if (ca_transmissions > 1)
  {
    ca_iet_min = result.ca_trigger_times[1] - result.ca_trigger_times[0];
    ca_iet_max = ca_iet_min;
    double sum = 0.0;
    for (std::size_t k = 1; k < ca_transmissions; ++k)
    {
      const double dt_event = result.ca_trigger_times[k] - result.ca_trigger_times[k - 1];
      sum += dt_event;
      if (dt_event < ca_iet_min)
        ca_iet_min = dt_event;
      if (dt_event > ca_iet_max)
        ca_iet_max = dt_event;
    }
    ca_iet_mean = sum / static_cast<double>(ca_transmissions - 1);
  }

  double sum_sq_err = 0.0, max_norm_err = 0.0;
  for (int step = 0; step < num_steps; ++step)
  {
    double sq = 0.0;
    for (int i = 0; i < state_dim; ++i)
    {
      const double val = result.estimation_error_data[step * state_dim + i];
      sq += val * val;
    }
    sum_sq_err += sq;
    const double norm = std::sqrt(sq);
    if (norm > max_norm_err)
      max_norm_err = norm;
  }
  const double rms_err = std::sqrt(sum_sq_err / static_cast<double>(num_steps));

  double sum_sq_res = 0.0, max_norm_res = 0.0;
  for (double val : result.residual_norm)
  {
    sum_sq_res += val * val;
    if (val > max_norm_res)
      max_norm_res = val;
  }
  const double rms_res = std::sqrt(sum_sq_res / static_cast<double>(num_steps));

  const double total_alarm_duration =
      static_cast<double>(
          std::count(result.alarm_active.begin(), result.alarm_active.end(), true)) *
      time_step;
  const double malicious_control_duration =
      static_cast<double>(result.malicious_control_steps) * time_step;
  const double malicious_exposure_pct =
      (duration > 0.0)
          ? (malicious_control_duration / duration) * 100.0
          : 0.0;
  const double alarm_active_pct =
      (duration > 0.0)
          ? (total_alarm_duration / duration) * 100.0
          : 0.0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout
      << "\n======================================================================\n"
      << "  RELATÓRIO DIAGNÓSTICO: DUAL-CHANNEL PETC SOB ATAQUES FDI (CANAL SC) \n"
      << "======================================================================\n"
      << " 1. PARÂMETROS TEMPORAIS E CONFIGURAÇÃO DA SIMULAÇÃO                  \n"
      << "----------------------------------------------------------------------\n"
      << "  - Duração Total da Simulação (T)       : " << duration << " s\n"
      << "  - Período Fundamental de Amostragem (h): " << sampling_period * 1e3 << " ms\n"
      << "  - Passo de Integração RK5 (dt)         : " << time_step * 1e6 << " µs\n"
      << "  - Limiar de Detecção Adotado (J_th)    : " << detection_threshold << "\n"
      << "----------------------------------------------------------------------\n"
      << " 2. DESEMPENHO DE REDE (DUAL-CHANNEL PETC)                            \n"
      << "----------------------------------------------------------------------\n"
      << "  - Avaliações Periódicas Totais (s_m)   : " << static_cast<int>(total_samples) << "\n"
      << "  - Disparos Canal SC (Sensor-Controlador): " << sc_transmissions
      << " (Redução: " << sc_reduction << " %)\n"
      << "  - IET Canal SC [Mín / Méd / Máx]       : [" << sc_iet_min << " s / " << sc_iet_mean << " s / " << sc_iet_max << " s]\n"
      << "  - Disparos Canal CA (Controlador-Atuador): " << ca_transmissions
      << " (Redução: " << ca_reduction << " %)\n"
      << "  - IET Canal CA [Mín / Méd / Máx]       : [" << ca_iet_min << " s / " << ca_iet_mean << " s / " << ca_iet_max << " s]\n"
      << "----------------------------------------------------------------------\n"
      << " 3. QUALIDADE DINÂMICA DE ESTIMAÇÃO E RESÍDUO                         \n"
      << "----------------------------------------------------------------------\n"
      << "  - Erro de Estimação ||e(t)||_2 (RMS)   : " << rms_err << "\n"
      << "  - Erro de Estimação ||e(t)||_2 (Máximo): " << max_norm_err << "\n"
      << "  - Resíduo de Saída ||r(t)||_2 (RMS)    : " << rms_res << "\n"
      << "  - Resíduo de Saída ||r(t)||_2 (Máximo) : " << max_norm_res << "\n"
      << "----------------------------------------------------------------------\n"
      << " 4. DIAGNÓSTICO DO DETECTOR E RESILIÊNCIA A ATAQUES                   \n"
      << "----------------------------------------------------------------------\n"
      << "  - Falsos Positivos (Alarme sem ataque) : " << result.false_positives << " evento(s)\n"
      << "  - Duração Total com Alarme Ativo       : " << total_alarm_duration << " s (" << alarm_active_pct << " %)\n"
      << "  - Atualizações sob FDI não Mitigado    : " << result.malicious_control_count << " pacote(s)\n"
      << "  - Passos de Integração com Estado Corrompido: " << result.malicious_control_steps << " passo(s)\n"
      << "  - Tempo Total sob Atuação Maliciosa    : " << malicious_control_duration << " s\n"
      << "  - Exposição da Malha ao Sinal Corrompido: " << malicious_exposure_pct << " % da simulação\n"
      << "======================================================================\n"
      << std::endl;
}

void run_observer_petc_under_attack_simulation_2(PeriodicETC::LITEngine &engine)
{
  using Algebra::Matrix;
  using Algebra::Vector;
  namespace fs = std::filesystem;

  const double duration = 10.0;
  const double time_step = 1e-4;
  const double sampling_period = 1e-1;

  const int state_dim = static_cast<int>(engine.getStateDim());
  const int output_dim = static_cast<int>(engine.getOutputDim());
  const int input_dim = static_cast<int>(engine.getInputDim());

  const double max_iet_sc = 5.0 * sampling_period;
  const double max_iet_ca = 10.0;

  Vector tilde_x0(state_dim);
  tilde_x0[0] = 0.05;
  tilde_x0[1] = 0.05;
  const double epsilon_floor = 1e-5;

  Vector x0(state_dim);
  x0[0] = -1.0;
  x0[1] = 1.0;

  Vector x_hat0(state_dim);
  x_hat0[0] = 0.0;
  x_hat0[1] = 0.0;

  Vector x_hat_a0(state_dim);
  x_hat_a0[0] = 0.0;
  x_hat_a0[1] = 0.0;

  Matrix K(1, 2, {1.39e+00, -3.38e+00});
  Matrix L0(2, 2, {0.00e+00, 0.00e+00, 0.00e+00, -1.50e+00});
  Matrix L1(2, 1, {9.83e-01, -1.13e-01});
  Matrix L2(2, 1, {8.40e+00, 2.46e+01});

  Matrix Xi_sc(1, 1, {8.66e-01});
  Matrix Psi_sc(1, 1, {1.13e+00});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_sc_config;
  etm_sc_config.sigma = 1.0;
  etm_sc_config.threshold = 0.0;
  etm_sc_config.Psi = Psi_sc;
  etm_sc_config.Xi = Xi_sc;

  Matrix Xi_ca(2, 2, {2.73e+05, -5.26e+05, -5.26e+05, 1.30e+06});
  Matrix Psi_ca(2, 2, {1.13e+05, 2.74e+04, 2.74e+04, 1.38e+05});

  PeriodicETC::LIT_SETM::StaticETMConfig etm_ca_config;
  etm_ca_config.sigma = 1.0;
  etm_ca_config.threshold = 0.0;
  etm_ca_config.Psi = Psi_ca;
  etm_ca_config.Xi = Xi_ca;

  TemporaryOutputNoise attack_noise(
      output_dim, 0.5, 1.5, 0.1, TemporaryOutputNoise::Type::BIAS);

  auto result = engine.runDualChannelUnderAttackClosedLoop(
      x0, x_hat0, x_hat_a0, tilde_x0, K, L0, L1, L2,
      etm_sc_config, etm_ca_config,
      sampling_period, duration, time_step,
      "DUAL_CHANNEL_ATTACK",
      std::nullopt, max_iet_sc, max_iet_ca,
      attack_noise, epsilon_floor);

  fs::path dir = "simulations/lit-system-dual-channel-under-attack";
  fs::create_directories(dir);

  const int num_steps = static_cast<int>(result.time_data.size());
  const std::size_t sc_transmissions = result.sc_trigger_times.size();

  BinaryLogger::dump(dir / "time.bin", result.time_data);
  BinaryLogger::dump(dir / "sc_trigger_times.bin", result.sc_trigger_times);
  BinaryLogger::dump(dir / "ca_trigger_times.bin", result.ca_trigger_times);
  BinaryLogger::dump(dir / "residual_norm.bin", result.residual_norm);

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.states_data[step * state_dim + i]);
    BinaryLogger::dump(dir / ("x" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.estimated_states_data[step * state_dim + i]);
    BinaryLogger::dump(dir / ("x_est" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < state_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.estimation_error_data[step * state_dim + i]);
    BinaryLogger::dump(dir / ("e" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < input_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.control_data[step * input_dim + i]);
    BinaryLogger::dump(dir / ("u" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < output_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(num_steps);
    for (int step = 0; step < num_steps; ++step)
      traj.push_back(result.residual[step * output_dim + i]);
    BinaryLogger::dump(dir / ("r" + std::to_string(i + 1) + ".bin"), traj);
  }

  std::vector<double> alarm_traj;
  alarm_traj.reserve(result.alarm_active.size());
  for (bool val : result.alarm_active)
    alarm_traj.push_back(val ? 1.0 : 0.0);
  BinaryLogger::dump(dir / "alarm_active.bin", alarm_traj);

  for (int i = 0; i < output_dim; ++i)
  {
    std::vector<double> traj;
    traj.reserve(sc_transmissions);
    for (std::size_t k = 0; k < sc_transmissions; ++k)
      traj.push_back(result.malicious_signal[k * output_dim + i]);
    BinaryLogger::dump(dir / ("malicious_attack" + std::to_string(i + 1) + ".bin"), traj);
  }

  for (int i = 0; i < output_dim; ++i)
  {
    std::vector<double> traj_upper;
    std::vector<double> traj_lower;
    traj_upper.reserve(sc_transmissions);
    traj_lower.reserve(sc_transmissions);
    for (std::size_t k = 0; k < sc_transmissions; ++k)
    {
      traj_upper.push_back(result.threshold_upper[k * output_dim + i]);
      traj_lower.push_back(result.threshold_lower[k * output_dim + i]);
    }
    BinaryLogger::dump(dir / ("threshold_upper" + std::to_string(i + 1) + ".bin"), traj_upper);
    BinaryLogger::dump(dir / ("threshold_lower" + std::to_string(i + 1) + ".bin"), traj_lower);
  }

  const double total_samples = duration / sampling_period;
  const std::size_t ca_transmissions = result.ca_trigger_times.size();

  const double sc_reduction =
      (total_samples > 0.0)
          ? (1.0 - static_cast<double>(sc_transmissions) / total_samples) * 100.0
          : 0.0;
  const double ca_reduction =
      (total_samples > 0.0)
          ? (1.0 - static_cast<double>(ca_transmissions) / total_samples) * 100.0
          : 0.0;

  double sc_iet_min = 0.0, sc_iet_max = 0.0, sc_iet_mean = 0.0;
  if (sc_transmissions > 1)
  {
    sc_iet_min = result.sc_trigger_times[1] - result.sc_trigger_times[0];
    sc_iet_max = sc_iet_min;
    double sum = 0.0;
    for (std::size_t k = 1; k < sc_transmissions; ++k)
    {
      const double dt_event = result.sc_trigger_times[k] - result.sc_trigger_times[k - 1];
      sum += dt_event;
      if (dt_event < sc_iet_min)
        sc_iet_min = dt_event;
      if (dt_event > sc_iet_max)
        sc_iet_max = dt_event;
    }
    sc_iet_mean = sum / static_cast<double>(sc_transmissions - 1);
  }

  double ca_iet_min = 0.0, ca_iet_max = 0.0, ca_iet_mean = 0.0;
  if (ca_transmissions > 1)
  {
    ca_iet_min = result.ca_trigger_times[1] - result.ca_trigger_times[0];
    ca_iet_max = ca_iet_min;
    double sum = 0.0;
    for (std::size_t k = 1; k < ca_transmissions; ++k)
    {
      const double dt_event = result.ca_trigger_times[k] - result.ca_trigger_times[k - 1];
      sum += dt_event;
      if (dt_event < ca_iet_min)
        ca_iet_min = dt_event;
      if (dt_event > ca_iet_max)
        ca_iet_max = dt_event;
    }
    ca_iet_mean = sum / static_cast<double>(ca_transmissions - 1);
  }

  double sum_sq_err = 0.0, max_norm_err = 0.0;
  for (int step = 0; step < num_steps; ++step)
  {
    double sq = 0.0;
    for (int i = 0; i < state_dim; ++i)
    {
      const double val = result.estimation_error_data[step * state_dim + i];
      sq += val * val;
    }
    sum_sq_err += sq;
    const double norm = std::sqrt(sq);
    if (norm > max_norm_err)
      max_norm_err = norm;
  }
  const double rms_err = std::sqrt(sum_sq_err / static_cast<double>(num_steps));

  double sum_sq_res = 0.0, max_norm_res = 0.0;
  for (double val : result.residual_norm)
  {
    sum_sq_res += val * val;
    if (val > max_norm_res)
      max_norm_res = val;
  }
  const double rms_res = std::sqrt(sum_sq_res / static_cast<double>(num_steps));

  const double total_alarm_duration =
      static_cast<double>(
          std::count(result.alarm_active.begin(), result.alarm_active.end(), true)) *
      time_step;
  const double malicious_control_duration =
      static_cast<double>(result.malicious_control_steps) * time_step;
  const double malicious_exposure_pct =
      (duration > 0.0)
          ? (malicious_control_duration / duration) * 100.0
          : 0.0;
  const double alarm_active_pct =
      (duration > 0.0)
          ? (total_alarm_duration / duration) * 100.0
          : 0.0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout
      << "\n======================================================================\n"
      << "  RELATÓRIO DIAGNÓSTICO: DUAL-CHANNEL PETC SOB ATAQUES FDI (CANAL SC) \n"
      << "======================================================================\n"
      << " 1. PARÂMETROS TEMPORAIS E DETECTOR ZONOTÓPICO                        \n"
      << "----------------------------------------------------------------------\n"
      << "  - Duração Total da Simulação (T)       : " << duration << " s\n"
      << "  - Período Fundamental de Amostragem (h): " << sampling_period * 1e3 << " ms\n"
      << "  - Passo de Integração RK5 (dt)         : " << time_step * 1e6 << " µs\n"
      << "  - Detector de Resíduo                  : Zonotopo Dinâmico (Interval Hull)\n"
      << "  - Incerteza Inicial (tilde_x0)         : [" << tilde_x0[0] << ", " << tilde_x0[1] << "]\n"
      << "  - Piso Numérico de Tolerância (eps)    : " << epsilon_floor << "\n"
      << "----------------------------------------------------------------------\n"
      << " 2. DESEMPENHO DE REDE (DUAL-CHANNEL PETC)                            \n"
      << "----------------------------------------------------------------------\n"
      << "  - Avaliações Periódicas Totais (s_m)   : " << static_cast<int>(total_samples) << "\n"
      << "  - Disparos Canal SC (Sensor-Controlador): " << sc_transmissions
      << " (Redução: " << sc_reduction << " %)\n"
      << "  - IET Canal SC [Mín / Méd / Máx]       : [" << sc_iet_min << " s / " << sc_iet_mean << " s / " << sc_iet_max << " s]\n"
      << "  - Disparos Canal CA (Controlador-Atuador): " << ca_transmissions
      << " (Redução: " << ca_reduction << " %)\n"
      << "  - IET Canal CA [Mín / Méd / Máx]       : [" << ca_iet_min << " s / " << ca_iet_mean << " s / " << ca_iet_max << " s]\n"
      << "----------------------------------------------------------------------\n"
      << " 3. QUALIDADE DINÂMICA DE ESTIMAÇÃO E RESÍDUO                         \n"
      << "----------------------------------------------------------------------\n"
      << "  - Erro de Estimação ||e(t)||_2 (RMS)   : " << rms_err << "\n"
      << "  - Erro de Estimação ||e(t)||_2 (Máximo): " << max_norm_err << "\n"
      << "  - Resíduo de Saída ||r(t)||_2 (RMS)    : " << rms_res << "\n"
      << "  - Resíduo de Saída ||r(t)||_2 (Máximo) : " << max_norm_res << "\n"
      << "----------------------------------------------------------------------\n"
      << " 4. DIAGNÓSTICO DO DETECTOR E RESILIÊNCIA A ATAQUES                   \n"
      << "----------------------------------------------------------------------\n"
      << "  - Falsos Positivos (Alarme sem ataque) : " << result.false_positives << " evento(s)\n"
      << "  - Duração Total com Alarme Ativo       : " << total_alarm_duration << " s (" << alarm_active_pct << " %)\n"
      << "  - Atualizações sob FDI não Mitigado    : " << result.malicious_control_count << " pacote(s)\n"
      << "  - Passos de Integração com Estado Corrompido: " << result.malicious_control_steps << " passo(s)\n"
      << "  - Tempo Total sob Atuação Maliciosa    : " << malicious_control_duration << " s\n"
      << "  - Exposição da Malha ao Sinal Corrompido: " << malicious_exposure_pct << " % da simulação\n"
      << "======================================================================\n"
      << std::endl;
}

int main()
{
  std::string systems_directory = "../experiments/data/";
  fs::path jsonPath = systems_directory + "lit-systems/sys-01.json";

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
    // run_closed_loop_setm_event_map_simulation(engine);
    // run_dual_channel_closed_loop_setm_simulation(engine);
    // run_observer_petc_closed_loop_simulation(engine);
    run_observer_petc_under_attack_simulation_2(engine);
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Erro durante a execucao: " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}