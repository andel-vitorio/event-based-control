#include <vector>
#include <string>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <mutex>
#include <future>
#include <thread>

#include "StateSystemModel.hpp"
#include "StateSystemParser.hpp"
#include "BinaryLogger.hpp"
#include "LITSystem.hpp"
#include "LPVSystem.hpp"
#include "Algebra/Algebra.hpp"
#include "EDOSolvers/EDOSolvers.hpp"
#include "PeriodicETM.hpp"

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

namespace PeriodicETC
{
  namespace LIT
  {

    struct ClosedLoopScenario
    {
      Algebra::Vector x0;
      Algebra::Matrix K;
      Algebra::Matrix Xi;
      Algebra::Matrix Psi;
      double sampling_period;
    };

    struct SimulationResult
    {
      std::vector<double> time_data;
      std::vector<double> states_data;
      std::vector<double> control_data;
      std::vector<double> event_times;
    };

    struct SimulationParams
    {
      double time_step;
      double final_time;
    };

    class Engine
    {
    private:
      std::unique_ptr<ControlSystems::LPVSystem> lpvSystem;
      std::unique_ptr<EDOSolvers::RK5> solver;
      std::vector<SimulationResult> multi_thread_results;
      std::mutex results_mutex;

      Algebra::Vector x;
      Algebra::Vector u;
      Algebra::Vector w;

      SimulationParams params;
      bool system_loaded = false;
      bool state_initialized = false;

      std::vector<double> history_buffer;
      std::vector<double> time_buffer;
      int state_dim = 0;
      int num_steps = 0;

    public:
      Engine() : x(0), u(0), w(0) {}
      ~Engine() = default;

      void loadSystem(const std::string &json_path)
      {
        using namespace ControlSystems;
        StateSystemModel model = StateSystemParser::parseFromFile(json_path);

        lpvSystem = std::make_unique<LPVSystem>(model);

        // Determina estritamente a dimensão estrutural a partir do modelo
        state_dim = static_cast<int>(model.states.size());
        if (state_dim == 0)
        {
          throw std::runtime_error("O modelo carregado nao contem definicoes de estados.");
        }

        // Dimensionamento genérico de entradas e perturbações baseados no JSON
        std::size_t num_inputs = lpvSystem->n_inputs();
        std::size_t num_disturbances = model.disturbances.size();

        u = Algebra::Vector(num_inputs);
        for (std::size_t i = 0; i < num_inputs; ++i)
          u[i] = 0.0;

        w = Algebra::Vector(num_disturbances);
        for (std::size_t i = 0; i < num_disturbances; ++i)
          w[i] = 0.0;

        // Configuração matemática do Solver vinculada ao modelo fixo
        solver = std::make_unique<EDOSolvers::RK5>([this](double t, const Algebra::Vector &current_x, const Algebra::Vector &signal)
                                                   {
            std::size_t n_in = lpvSystem->n_inputs();
            Algebra::Vector current_u = signal.slice(0, n_in);
            Algebra::Vector current_w = signal.slice(n_in, signal.size());
            return lpvSystem->stateDerivative(current_x, current_u, current_w, t); });

        system_loaded = true;
        state_initialized = false; // Aguardando x0 externo
      }

      Algebra::Matrix getLPVMatrix(const std::string &name, const Algebra::Variables &rho)
      {
        if (!lpvSystem)
          throw std::runtime_error("Sistema não carregado.");
        return lpvSystem->getMatrix(name, rho);
      }

      // Injeta a condição inicial dinamicamente de forma agnóstica de dimensão
      void setInitialState(const double *initial_x_ptr, int dim)
      {
        if (!system_loaded)
        {
          throw std::runtime_error("Carregue o sistema antes de definir o estado inicial.");
        }
        if (dim != state_dim)
        {
          throw std::runtime_error("Dimensao fornecida incompativel com a dimensao do sistema carregado.");
        }

        x = Algebra::Vector(state_dim);
        for (int i = 0; i < state_dim; ++i)
        {
          x[i] = initial_x_ptr[i];
        }
        state_initialized = true;
      }

      void loadSimulation(const SimulationParams &p)
      {
        params = p;
      }

      void openLoop(double duration)
      {
        if (!system_loaded || !state_initialized || !solver)
        {
          throw std::runtime_error("Engine nao inicializada completamente. Verifique se carregou o sistema e definiu o estado inicial.");
        }

        double dt = params.time_step;
        auto timepts = Algebra::arange(0.0, duration, dt);

        num_steps = static_cast<int>(timepts.size());
        history_buffer.clear();
        time_buffer = timepts;

        history_buffer.reserve(num_steps * state_dim);

        for (const auto t : timepts)
        {
          for (int i = 0; i < state_dim; ++i)
          {
            history_buffer.push_back(x[i]);
          }

          Algebra::Vector signal = Algebra::Vector::concatenate(u, w);
          x = solver->step(t, x, signal, dt);
        }
      }

      void runClosedLoopMultiThread(const std::vector<ClosedLoopScenario> &scenarios, double sample_time, double duration)
      {
        using Vector = Algebra::Vector;
        multi_thread_results.clear();
        multi_thread_results.resize(scenarios.size());

        std::vector<std::future<void>> workers;

        for (size_t i = 0; i < scenarios.size(); ++i)
        {
          workers.push_back(std::async(std::launch::async, [this, i, &scenarios, sample_time, duration]()
                                       {
                const auto& sc = scenarios[i];
                
                ControlSystems::LPVSystem localSystem = *(this->lpvSystem); // Cópia segura para leitura
                Vector x = sc.x0;
                Vector w(0);
                
                EDOSolvers::RK5 solver([&localSystem](double t, const Vector &x, const Vector &signal) {
                    std::size_t num_inputs = localSystem.n_inputs();
                    Vector u = signal.slice(0, num_inputs);
                    Vector w = signal.slice(num_inputs, signal.size());
                    return localSystem.stateDerivative(x, u, w, t);
                });

                Sampler sampler(sc.sampling_period);
                StaticSETM etm(sc.Xi, sc.Psi);
                
                Vector control_signal = sc.K * x;
                auto timepts = Algebra::arange(0.0, duration, sample_time);

                SimulationResult res;
                res.time_data.reserve(timepts.size());
                res.states_data.reserve(timepts.size() * x.size());

                for (const auto t : timepts) {
                    res.time_data.push_back(t);
                    for (size_t j = 0; j < x.size(); ++j) 
                      res.states_data.push_back(x[j]);
                    for (size_t j = 0; j < control_signal.size(); ++j)
                      res.control_data.push_back(control_signal[j]);
                    if (sampler.shouldSample(t) && etm.evaluate(x)) {
                        res.event_times.push_back(t);
                        control_signal = sc.K * x;
                    }

                    Vector signal = Vector::concatenate(control_signal, w);
                    x = solver.step(t, x, signal, sample_time);
                }

                std::lock_guard<std::mutex> lock(results_mutex);
                multi_thread_results[i] = std::move(res); }));
        }

        for (auto &worker : workers)
        {
          worker.get();
        }
      }

      size_t getNumScenariosResults() const { return multi_thread_results.size(); }

      void getScenarioDataSizes(size_t scenario_idx, int *time_size, int *states_size, int *control_size, int *events_size)
      {
        if (scenario_idx >= multi_thread_results.size())
          return;
        const auto &res = multi_thread_results[scenario_idx];
        *time_size = static_cast<int>(res.time_data.size());
        *states_size = static_cast<int>(res.states_data.size());
        *control_size = static_cast<int>(res.control_data.size());
        *events_size = static_cast<int>(res.event_times.size());
      }

      void copyScenarioResults(size_t scenario_idx, double *t_out, double *x_out, double *u_out, double *ev_out)
      {
        if (scenario_idx >= multi_thread_results.size())
          return;
        const auto &res = multi_thread_results[scenario_idx];
        std::copy(res.time_data.begin(), res.time_data.end(), t_out);
        std::copy(res.states_data.begin(), res.states_data.end(), x_out);
        std::copy(res.control_data.begin(), res.control_data.end(), u_out);
        std::copy(res.event_times.begin(), res.event_times.end(), ev_out);
      }

      const double *getTimeData() const { return time_buffer.data(); }
      int getTimeSize() const { return static_cast<int>(time_buffer.size()); }

      const double *getHistoryData() const { return history_buffer.data(); }
      int getHistorySize() const { return static_cast<int>(history_buffer.size()); }
      int getStateDim() const { return state_dim; }
    };

  } // namespace LIT
} // namespace PeriodicETC

// ============================================================================
// INTERFACE EXPORTADA DA DLL (C LINKAGE)
// ============================================================================

using EnginePtr = PeriodicETC::LIT::Engine *;

DLLEXPORT EnginePtr create()
{
  return new PeriodicETC::LIT::Engine();
}

DLLEXPORT void load_system(EnginePtr e, const char *path)
{
  try
  {
    if (e)
      e->loadSystem(std::string(path));
  }
  catch (const std::exception &ex)
  {
    std::cerr << "[C++ DLL Error]: " << ex.what() << std::endl;
  }
}

// Nova função exposta para injetar x0 de qualquer dimensão vindo do Python
DLLEXPORT void set_initial_state(EnginePtr e, const double *initial_x, int dim)
{
  try
  {
    if (e)
      e->setInitialState(initial_x, dim);
  }
  catch (const std::exception &ex)
  {
    std::cerr << "[C++ DLL Error]: " << ex.what() << std::endl;
  }
}

DLLEXPORT void load_sim(EnginePtr e, double dt, double tf)
{
  if (e)
  {
    PeriodicETC::LIT::SimulationParams p;
    p.time_step = dt;
    p.final_time = tf;
    e->loadSimulation(p);
  }
}

DLLEXPORT void open_loop(EnginePtr e, double duration)
{
  try
  {
    if (e)
      e->openLoop(duration);
  }
  catch (const std::exception &ex)
  {
    std::cerr << "[C++ DLL Error in open_loop]: " << ex.what() << std::endl;
  }
}

DLLEXPORT const double *get_history_data(EnginePtr e)
{
  return e ? e->getHistoryData() : nullptr;
}

DLLEXPORT int get_history_size(EnginePtr e)
{
  return e ? e->getHistorySize() : 0;
}

DLLEXPORT int get_state_dim(EnginePtr e)
{
  return e ? e->getStateDim() : 0;
}

DLLEXPORT void destroy(EnginePtr e)
{
  if (e)
    delete e;
}

DLLEXPORT const double *get_time_data(EnginePtr e)
{
  return e ? e->getTimeData() : nullptr;
}

DLLEXPORT int get_time_size(EnginePtr e)
{
  return e ? e->getTimeSize() : 0;
}

DLLEXPORT void get_matrix(EnginePtr e, const char *matrix_name,
                          const char **param_names, const double *param_values, int num_params,
                          double *out_buffer, int rows, int cols)
{
  if (!e)
    return;

  Algebra::Variables rho;
  for (int i = 0; i < num_params; ++i)
  {
    rho[std::string(param_names[i])] = param_values[i];
  }

  try
  {
    Algebra::Matrix mat = e->getLPVMatrix(std::string(matrix_name), rho);

    // 3. Copiar para o buffer
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        out_buffer[i * cols + j] = mat(i, j);
      }
    }
  }
  catch (const std::exception &ex)
  {
    // Opcional: logar o erro
  }
}

DLLEXPORT void run_closed_loop(
    EnginePtr e,
    const double *x0_matrix, int n_states,
    const double *K_matrix, int nu,
    const double *Xi_matrix, const double *Psi_matrix,
    double sampling_period, int num_scenarios,
    double sample_time, double duration)
{
  if (!e)
    return;

  std::vector<PeriodicETC::LIT::ClosedLoopScenario> scenarios(num_scenarios);
  for (int i = 0; i < num_scenarios; ++i)
  {
    // Inicializa x0 para cada cenário
    scenarios[i].x0 = Algebra::Vector(n_states);
    for (int j = 0; j < n_states; ++j)
      scenarios[i].x0[j] = x0_matrix[i * n_states + j];

    // Inicializa K para cada cenário
    scenarios[i].K = Algebra::Matrix(nu, n_states);
    for (int r = 0; r < nu; ++r)
    {
      for (int c = 0; c < n_states; ++c)
      {
        scenarios[i].K(r, c) = K_matrix[i * (nu * n_states) + (r * n_states + c)];
      }
    }

    // Inicializa Xi e Psi (Matrizes n_states x n_states)
    scenarios[i].Xi = Algebra::Matrix(n_states, n_states);
    scenarios[i].Psi = Algebra::Matrix(n_states, n_states);
    for (int r = 0; r < n_states; ++r)
    {
      for (int c = 0; c < n_states; ++c)
      {
        scenarios[i].Xi(r, c) = Xi_matrix[i * (n_states * n_states) + (r * n_states + c)];
        scenarios[i].Psi(r, c) = Psi_matrix[i * (n_states * n_states) + (r * n_states + c)];
      }
    }
    scenarios[i].sampling_period = sampling_period;
  }

  e->runClosedLoopMultiThread(scenarios, sample_time, duration);
}

DLLEXPORT int get_num_scenarios_results(EnginePtr e)
{
  return e ? static_cast<int>(e->getNumScenariosResults()) : 0;
}

DLLEXPORT void get_scenario_data_sizes(EnginePtr e, int idx, int *t_sz, int *x_sz, int *u_sz, int *ev_sz)
{
  if (e)
    e->getScenarioDataSizes(static_cast<size_t>(idx), t_sz, x_sz, u_sz, ev_sz);
}

DLLEXPORT void copy_scenario_results(EnginePtr e, int idx, double *t_out, double *x_out, double *u_out, double *ev_out)
{
  if (e)
    e->copyScenarioResults(static_cast<size_t>(idx), t_out, x_out, u_out, ev_out);
}