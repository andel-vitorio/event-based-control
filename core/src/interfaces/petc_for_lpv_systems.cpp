#include <vector>
#include <string>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <mutex>
#include <future>
#include <thread>
#include <random>
#include <algorithm>

#include "StateSystemModel.hpp"
#include "StateSystemParser.hpp"
#include "BinaryLogger.hpp"
#include "LITSystem.hpp"
#include "LPVSystem.hpp"
#include "Algebra/Algebra.hpp"
#include "EDOSolvers/EDOSolvers.hpp"
#include "PeriodicETM.hpp"
#include "MTD.hpp" // Inclusão da nova classe MTDManager

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
      Algebra::Matrix K, Xi, Psi;
      Algebra::Vector x0;
      double sampling_period;
    };

    struct SimulationResult
    {
      std::vector<double> time_data, states_data, control_data, event_times;
    };

    struct MTDSimulationResult
    {
      std::vector<double> time_data, states_data, control_data, event_times;
      std::vector<int> active_modes, active_regions;
      unsigned int seed;
    };

    struct SimulationParams
    {
      double time_step, final_time;
    };

    class Engine
    {
    private:
      std::unique_ptr<ControlSystems::LPVSystem> lpvSystem;
      std::unique_ptr<EDOSolvers::RK5> solver;

      std::vector<SimulationResult> multi_thread_results;
      std::vector<MTDSimulationResult> mtd_multi_thread_results;
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

        state_dim = static_cast<int>(model.states.size());
        if (state_dim == 0)
        {
          throw std::runtime_error("O modelo carregado nao contem definicoes de estados.");
        }

        std::size_t num_inputs = lpvSystem->n_inputs();
        std::size_t num_disturbances = model.disturbances.size();

        u = Algebra::Vector(num_inputs);
        for (std::size_t i = 0; i < num_inputs; ++i)
          u[i] = 0.0;

        w = Algebra::Vector(num_disturbances);
        for (std::size_t i = 0; i < num_disturbances; ++i)
          w[i] = 0.0;

        solver = std::make_unique<EDOSolvers::RK5>([this](double t, const Algebra::Vector &current_x, const Algebra::Vector &signal)
                                                   {
            std::size_t n_in = lpvSystem->n_inputs();
            Algebra::Vector current_u = signal.slice(0, n_in);
            Algebra::Vector current_w = signal.slice(n_in, signal.size());
            return lpvSystem->stateDerivative(current_x, current_u, current_w, t); });

        system_loaded = true;
        state_initialized = false;
      }

      Algebra::Matrix getLPVMatrix(const std::string &name, const Algebra::Variables &rho)
      {
        if (!lpvSystem)
          throw std::runtime_error("Sistema não carregado.");
        return lpvSystem->getMatrix(name, rho);
      }

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
          throw std::runtime_error("Engine nao inicializada completamente.");
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
                
                ControlSystems::LPVSystem localSystem = *(this->lpvSystem);
                Vector x = sc.x0;
                Vector w(0);
                
                EDOSolvers::RK5 local_solver([&localSystem](double t, const Vector &x, const Vector &signal) {
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
                    x = local_solver.step(t, x, signal, sample_time);
                }

                std::lock_guard<std::mutex> lock(results_mutex);
                multi_thread_results[i] = std::move(res); }));
        }

        for (auto &worker : workers)
        {
          worker.get();
        }
      }

      void runClosedLoopMTDMultiThread(
          const std::vector<Algebra::Vector> &x0_list,
          const std::vector<unsigned int> &seeds_list,
          const std::vector<double> &K_flat_table, int nu,
          const std::vector<double> &Xi_flat_table,
          const std::vector<double> &Psi_flat_table,
          const std::vector<double> &pi_tensor, int n_modes, int n_regions,
          const Algebra::Matrix &W, const std::vector<double> &c_array,
          double sampling_period, double sample_time, double duration)
      {
        using Vector = Algebra::Vector;
        using Matrix = Algebra::Matrix;

        size_t num_x0 = x0_list.size();
        size_t num_seeds = seeds_list.size();
        size_t total_scenarios = num_x0 * num_seeds;

        mtd_multi_thread_results.clear();
        mtd_multi_thread_results.resize(total_scenarios);

        std::vector<std::future<void>> workers;

        for (size_t idx = 0; idx < total_scenarios; ++idx)
        {
          size_t x0_idx = idx / num_seeds;
          size_t seed_idx = idx % num_seeds;

          workers.push_back(std::async(std::launch::async, [this, idx, x0_idx, seed_idx, &x0_list, &seeds_list, &K_flat_table, nu, &Xi_flat_table, &Psi_flat_table, &pi_tensor, n_modes, n_regions, &W, &c_array, sampling_period, sample_time, duration]()
                                       {
                int n_states = W.rows();
                unsigned int scenario_seed = seeds_list[seed_idx];
                Vector x = x0_list[x0_idx];
                Vector w(0);
                Vector x_hat = x;

                ControlSystems::MTDManager local_mtd;
                local_mtd.setSeed(scenario_seed);
                local_mtd.setTransitionProbabilities(pi_tensor.data(), n_modes, n_regions);
                local_mtd.setEnergyWeightMatrix(W);
                local_mtd.setEnergyThresholds(c_array.data(), n_regions);

                ControlSystems::LPVSystem localSystem = *(this->lpvSystem);
                
                EDOSolvers::RK5 local_solver([&localSystem](double t, const Vector &x, const Vector &signal) {
                    std::size_t num_inputs = localSystem.n_inputs();
                    Vector u = signal.slice(0, num_inputs);
                    Vector w = signal.slice(num_inputs, signal.size());
                    return localSystem.stateDerivative(x, u, w, t);
                });

                Sampler sampler(sampling_period);

                local_mtd.update(x);
                int current_mode = local_mtd.getCurrentMode();
                int current_region = local_mtd.getCurrentRegion();

                auto get_K = [&](int m, int r) {
                    Matrix K_mat(nu, n_states);
                    int offset = (m * n_regions + r) * (nu * n_states);
                    for (int row = 0; row < nu; ++row) {
                        for (int col = 0; col < n_states; ++col) {
                            K_mat(row, col) = K_flat_table[offset + row * n_states + col];
                        }
                    }
                    return K_mat;
                };

                auto get_Xi = [&](int m, int r) {
                    Matrix Xi_mat(n_states, n_states);
                    int offset = (m * n_regions + r) * (n_states * n_states);
                    for (int row = 0; row < n_states; ++row) {
                        for (int col = 0; col < n_states; ++col) {
                            Xi_mat(row, col) = Xi_flat_table[offset + row * n_states + col];
                        }
                    }
                    return Xi_mat;
                };

                auto get_Psi = [&](int m, int r) {
                    Matrix Psi_mat(n_states, n_states);
                    int offset = (m * n_regions + r) * (n_states * n_states);
                    for (int row = 0; row < n_states; ++row) {
                        for (int col = 0; col < n_states; ++col) {
                            Psi_mat(row, col) = Psi_flat_table[offset + row * n_states + col];
                        }
                    }
                    return Psi_mat;
                };

                Matrix K_active = get_K(current_mode, current_region);
                Vector control_signal = K_active * x;

                auto timepts = Algebra::arange(0.0, duration, sample_time);

                MTDSimulationResult res;
                res.seed = scenario_seed;
                res.time_data.reserve(timepts.size());
                res.states_data.reserve(timepts.size() * n_states);
                res.control_data.reserve(timepts.size() * nu);
                res.active_modes.reserve(timepts.size());
                res.active_regions.reserve(timepts.size());

                res.event_times.push_back(0.0);

                for (const auto t : timepts) {
                    res.time_data.push_back(t);
                    for (int j = 0; j < n_states; ++j) 
                        res.states_data.push_back(x[j]);
                    for (int j = 0; j < nu; ++j)
                        res.control_data.push_back(control_signal[j]);

                    if (sampler.shouldSample(t)) {
                        local_mtd.update(x);
                        current_mode = local_mtd.getCurrentMode();
                        current_region = local_mtd.getCurrentRegion();

                        res.active_modes.push_back(current_mode);
                        res.active_regions.push_back(current_region);

                        K_active = get_K(current_mode, current_region);
                        Matrix Xi_active = get_Xi(current_mode, current_region);
                        Matrix Psi_active = get_Psi(current_mode, current_region);

                        Vector error = x_hat - x;
                        double x_term = Algebra::Vector::dot(x.T(), Psi_active * x);
                        double error_term = Algebra::Vector::dot(error.T(), Xi_active * error);
                        double val = x_term - error_term;

                        if (val < 0) {
                            x_hat = x;
                            control_signal = K_active * x;
                            res.event_times.push_back(t);
                        }
                    }

                    Vector signal = Vector::concatenate(control_signal, w);
                    x = local_solver.step(t, x, signal, sample_time);
                }

                std::lock_guard<std::mutex> lock(results_mutex);
                mtd_multi_thread_results[idx] = std::move(res); }));
        }

        for (auto &worker : workers)
        {
          worker.get();
        }
      }

      size_t getNumScenariosResults() const { return multi_thread_results.size(); }
      size_t getNumMtdScenariosResults() const { return mtd_multi_thread_results.size(); }

      unsigned int getMtdScenarioSeed(size_t idx) const
      {
        if (idx >= mtd_multi_thread_results.size())
          return 0;
        return mtd_multi_thread_results[idx].seed;
      }

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

      void getMtdScenarioDataSizes(size_t scenario_idx, int *time_size, int *states_size, int *control_size, int *events_size, int *modes_size, int *regions_size)
      {
        if (scenario_idx >= mtd_multi_thread_results.size())
          return;
        const auto &res = mtd_multi_thread_results[scenario_idx];
        *time_size = static_cast<int>(res.time_data.size());
        *states_size = static_cast<int>(res.states_data.size());
        *control_size = static_cast<int>(res.control_data.size());
        *events_size = static_cast<int>(res.event_times.size());
        *modes_size = static_cast<int>(res.active_modes.size());
        *regions_size = static_cast<int>(res.active_regions.size());
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

      void copyMtdScenarioResults(size_t scenario_idx, double *t_out, double *x_out, double *u_out, double *ev_out, int *modes_out, int *regions_out)
      {
        if (scenario_idx >= mtd_multi_thread_results.size())
          return;
        const auto &res = mtd_multi_thread_results[scenario_idx];
        std::copy(res.time_data.begin(), res.time_data.end(), t_out);
        std::copy(res.states_data.begin(), res.states_data.end(), x_out);
        std::copy(res.control_data.begin(), res.control_data.end(), u_out);
        std::copy(res.event_times.begin(), res.event_times.end(), ev_out);
        std::copy(res.active_modes.begin(), res.active_modes.end(), modes_out);
        std::copy(res.active_regions.begin(), res.active_regions.end(), regions_out);
      }

      const double *getTimeData() const { return time_buffer.data(); }
      int getTimeSize() const { return static_cast<int>(time_buffer.size()); }

      const double *getHistoryData() const { return history_buffer.data(); }
      int getHistorySize() const { return static_cast<int>(history_buffer.size()); }
      int getStateDim() const { return state_dim; }
    };

  } // namespace LIT
} // namespace PeriodicETC

using EnginePtr = PeriodicETC::LIT::Engine *;

DLLEXPORT EnginePtr create() { return new PeriodicETC::LIT::Engine(); }

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

DLLEXPORT const double *get_history_data(EnginePtr e) { return e ? e->getHistoryData() : nullptr; }
DLLEXPORT int get_history_size(EnginePtr e) { return e ? e->getHistorySize() : 0; }
DLLEXPORT int get_state_dim(EnginePtr e) { return e ? e->getStateDim() : 0; }
DLLEXPORT void destroy(EnginePtr e)
{
  if (e)
    delete e;
}
DLLEXPORT const double *get_time_data(EnginePtr e) { return e ? e->getTimeData() : nullptr; }
DLLEXPORT int get_time_size(EnginePtr e) { return e ? e->getTimeSize() : 0; }

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
    scenarios[i].x0 = Algebra::Vector(n_states);
    for (int j = 0; j < n_states; ++j)
      scenarios[i].x0[j] = x0_matrix[i * n_states + j];
    scenarios[i].K = Algebra::Matrix(nu, n_states);
    for (int r = 0; r < nu; ++r)
    {
      for (int c = 0; c < n_states; ++c)
        scenarios[i].K(r, c) = K_matrix[i * (nu * n_states) + (r * n_states + c)];
    }
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

DLLEXPORT void run_closed_loop_mtd(
    EnginePtr e,
    const double *x0_matrix, int num_x0, int n_states,
    const unsigned int *seeds_input, int num_seeds, int use_random_seeds,
    const double *K_matrix, int nu,
    const double *Xi_matrix, const double *Psi_matrix,
    const double *pi_tensor, int n_modes, int n_regions,
    const double *W_matrix, const double *c_array,
    double sampling_period, double sample_time, double duration)
{
  if (!e)
    return;

  std::vector<Algebra::Vector> x0_list(num_x0);
  for (int i = 0; i < num_x0; ++i)
  {
    x0_list[i] = Algebra::Vector(n_states);
    for (int j = 0; j < n_states; ++j)
    {
      x0_list[i][j] = x0_matrix[i * n_states + j];
    }
  }

  std::vector<unsigned int> seeds_list(num_seeds);
  if (use_random_seeds)
  {
    std::random_device rd;
    for (int i = 0; i < num_seeds; ++i)
    {
      seeds_list[i] = rd();
    }
  }
  else
  {
    for (int i = 0; i < num_seeds; ++i)
    {
      seeds_list[i] = seeds_input[i];
    }
  }

  Algebra::Matrix W(n_states, n_states);
  for (int r = 0; r < n_states; ++r)
  {
    for (int c = 0; c < n_states; ++c)
    {
      W(r, c) = W_matrix[r * n_states + c];
    }
  }

  std::vector<double> K_vec(K_matrix, K_matrix + (n_modes * n_regions * nu * n_states));
  std::vector<double> Xi_vec(Xi_matrix, Xi_matrix + (n_modes * n_regions * n_states * n_states));
  std::vector<double> Psi_vec(Psi_matrix, Psi_matrix + (n_modes * n_regions * n_states * n_states));
  std::vector<double> pi_vec(pi_tensor, pi_tensor + (n_modes * n_modes * n_regions));
  std::vector<double> c_vec(c_array, c_array + n_regions - 1);

  e->runClosedLoopMTDMultiThread(
      x0_list, seeds_list,
      K_vec, nu, Xi_vec, Psi_vec,
      pi_vec, n_modes, n_regions,
      W, c_vec,
      sampling_period, sample_time, duration);
}

DLLEXPORT int get_num_scenarios_results(EnginePtr e) { return e ? static_cast<int>(e->getNumScenariosResults()) : 0; }
DLLEXPORT int get_num_mtd_scenarios_results(EnginePtr e) { return e ? static_cast<int>(e->getNumMtdScenariosResults()) : 0; }

DLLEXPORT unsigned int get_mtd_scenario_seed(EnginePtr e, int idx)
{
  return e ? e->getMtdScenarioSeed(static_cast<size_t>(idx)) : 0;
}

DLLEXPORT void get_scenario_data_sizes(EnginePtr e, int idx, int *t_sz, int *x_sz, int *u_sz, int *ev_sz)
{
  if (e)
    e->getScenarioDataSizes(static_cast<size_t>(idx), t_sz, x_sz, u_sz, ev_sz);
}

DLLEXPORT void get_mtd_scenario_data_sizes(EnginePtr e, int idx, int *t_sz, int *x_sz, int *u_sz, int *ev_sz, int *modes_sz, int *regions_sz)
{
  if (e)
    e->getMtdScenarioDataSizes(static_cast<size_t>(idx), t_sz, x_sz, u_sz, ev_sz, modes_sz, regions_sz);
}

DLLEXPORT void copy_scenario_results(EnginePtr e, int idx, double *t_out, double *x_out, double *u_out, double *ev_out)
{
  if (e)
    e->copyScenarioResults(static_cast<size_t>(idx), t_out, x_out, u_out, ev_out);
}

DLLEXPORT void copy_mtd_scenario_results(EnginePtr e, int idx, double *t_out, double *x_out, double *u_out, double *ev_out, int *modes_out, int *regions_out)
{
  if (e)
    e->copyMtdScenarioResults(static_cast<size_t>(idx), t_out, x_out, u_out, ev_out, modes_out, regions_out);
}