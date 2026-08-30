#include "PeriodicETC/LIT/simulators/closed_loop_setm.hpp"
#include "EDOSolvers/EDOSolvers.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <cmath>

namespace PeriodicETC
{
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

  namespace LIT_SETM
  {
    ClosedLoopResult run_standard_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w)
    {
      using Algebra::Vector;

      auto timepts = Algebra::arange(0.0, duration, time_step);

      const int num_steps = static_cast<int>(timepts.size());
      const int state_dim = static_cast<int>(x0.size());
      const int input_dim = static_cast<int>(K.rows());

      ClosedLoopResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);

      Vector actual_w = w.has_value() ? *w : Vector(0);

      StaticETM etm(etm_config, state_dim);
      Sampler sampler(sampling_period, 0.0);

      EDOSolvers::RK5 solver(
          [&plant]([[maybe_unused]] double t,
                   const Vector &x,
                   const Vector &signal)
          {
            std::size_t nu = plant.inputs();
            Vector ut = signal.slice(0, nu);
            Vector wt = signal.slice(nu, signal.size());
            return plant.stateDerivative(x, ut, wt);
          });

      Vector x = x0;
      Vector x_transmitted = x0;
      Vector u = K * x_transmitted;

      for (size_t i = 0; i < timepts.size(); ++i)
      {
        double t = timepts[i];

        if (i == 0)
          result.sc_trigger_times.push_back(t);
        else if (sampler.shouldSample(t))
        {
          if (etm.evaluateEvent(x, x_transmitted))
          {
            result.sc_trigger_times.push_back(t);
            x_transmitted = x;
          }
        }

        u = K * x_transmitted;

        for (int j = 0; j < state_dim; ++j)
          result.states_data.push_back(x[j]);

        for (int j = 0; j < input_dim; ++j)
          result.control_data.push_back(u[j]);

        Vector signal = Vector::concatenate(u, actual_w);
        x = solver.step(t, x, signal, time_step);
      }

      return result;
    }

    ExtendedClosedLoopResult run_observer_based_petc_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const StaticETMConfig &etm_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w)
    {
      using Algebra::Matrix;
      using Algebra::Vector;

      // ------------------------------------------------------------------
      // Time grid e Dimensões do Sistema
      // ------------------------------------------------------------------
      auto timepts = Algebra::arange(0.0, duration, time_step);
      const std::size_t num_steps = timepts.size();

      const int state_dim = static_cast<int>(plant.states());
      const int output_dim = static_cast<int>(plant.outputs());
      const int input_dim = static_cast<int>(plant.inputs());
      const int dist_dim = 0;

      // ------------------------------------------------------------------
      // Inicialização das Estruturas de Resultado
      // ------------------------------------------------------------------
      ExtendedClosedLoopResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.estimated_states_data.reserve(num_steps * state_dim);
      result.estimation_error_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);

      // ------------------------------------------------------------------
      // Vetor de Perturbação com Verificação Dimensional
      // ------------------------------------------------------------------
      Vector actual_w = w.has_value() ? *w : Vector(dist_dim);

      // ------------------------------------------------------------------
      // Componentes do PETC (Configurados na Dimensão de Saída Correta)
      // ------------------------------------------------------------------
      StaticETM etm(etm_config, output_dim);
      Sampler sampler(sampling_period, 0.0);

      // ------------------------------------------------------------------
      // Matrizes da Planta e Operadores de Passo Contínuo (dt)
      // ------------------------------------------------------------------
      Matrix A = plant.getA();
      Matrix B = plant.getB();
      Matrix C = plant.getC();

      // Propagação contínua da estimativa entre amostras (passo dt)
      Matrix Adt = Algebra::matrix_exp(A * time_step);
      Matrix Bdt = Algebra::integrate_matrix_exp_B(A, B, time_step, true);

      // ------------------------------------------------------------------
      // Solucionador Numérico (RK5) para a Planta Física
      // ------------------------------------------------------------------
      EDOSolvers::RK5 solver(
          [&plant](double /* t */, const Vector &x_sys, const Vector &signal)
          {
            const std::size_t nu = plant.inputs();
            Vector u = signal.slice(0, nu);
            Vector disturbance = signal.slice(nu, signal.size());
            return plant.stateDerivative(x_sys, u, disturbance);
          });

      // ------------------------------------------------------------------
      // Condições Iniciais dos Estados
      // ------------------------------------------------------------------
      Vector x = x0;
      Vector x_est = x0;
      Vector y_r = C * x0; // Inicializado estritamente em R^{ny}
      Vector u_transmitted = K * x_est;

      // ------------------------------------------------------------------
      // Loop Principal de Simulação
      // ------------------------------------------------------------------
      for (std::size_t i = 0; i < num_steps; ++i)
      {
        const double t = timepts[i];

        // 1. Verificação Periódica e Atualização Impulsiva
        if (i == 0)
        {
          result.sc_trigger_times.push_back(t);
          Vector y = C * x;
          y_r = y;
          Vector innovation = L * (y - C * x_est);
          x_est = x_est + innovation;
          u_transmitted = K * x_est;
        }
        else if (sampler.shouldSample(t))
        {
          Vector y = C * x;
          bool event_triggered = etm.evaluateEvent(y, y_r);

          if (event_triggered)
          {
            y_r = y;
            result.sc_trigger_times.push_back(t);

            // Correção impulsiva da estimativa
            Vector innovation = L * (y - C * x_est);
            x_est = x_est + innovation;

            // Atualização do sinal de controle
            u_transmitted = K * x_est;
          }
        }

        // 2. Registro dos Dados do Instante Atual
        Vector estimation_error = x - x_est;

        for (int j = 0; j < state_dim; ++j)
        {
          result.states_data.push_back(x[j]);
          result.estimated_states_data.push_back(x_est[j]);
          result.estimation_error_data.push_back(estimation_error[j]);
        }

        for (int j = 0; j < input_dim; ++j)
        {
          result.control_data.push_back(u_transmitted[j]);
        }

        // 3. Propagação Contínua da Dinâmica no Intervalo [t, t + dt]
        // Planta física via RK5
        Vector signal = Vector::concatenate(u_transmitted, actual_w);
        x = solver.step(t, x, signal, time_step);

        // Dinâmica contínua do observador: \dot{\hat{x}} = A \hat{x} + B u
        x_est = Adt * x_est + Bdt * u_transmitted;
      }

      return result;
    }

    ExtendedClosedLoopResult run_dual_channel_observer_petc_simulation(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        const Algebra::Vector &x_hat0,
        const Algebra::Matrix &K,
        const Algebra::Matrix &L,
        const StaticETMConfig &etm_sc_config,
        const StaticETMConfig &etm_ca_config,
        double sampling_period,
        double duration,
        double time_step,
        std::optional<Algebra::Vector> w,
        double max_iet)
    {
      using Algebra::Matrix;
      using Algebra::Vector;

      // ------------------------------------------------------------------
      // 1. Grade Temporal e Dimensões do Sistema
      // ------------------------------------------------------------------
      const auto timepts = Algebra::arange(0.0, duration, time_step);
      const std::size_t num_steps = timepts.size();

      const int state_dim = static_cast<int>(plant.states());
      const int output_dim = static_cast<int>(plant.outputs());
      const int input_dim = static_cast<int>(plant.inputs());

      // ------------------------------------------------------------------
      // 2. Alocação e Pré-Reserva de Memória
      // ------------------------------------------------------------------
      ExtendedClosedLoopResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);
      result.estimated_states_data.reserve(num_steps * state_dim);
      result.estimation_error_data.reserve(num_steps * state_dim);
      result.control_data.reserve(num_steps * input_dim);
      result.sc_trigger_times.reserve(num_steps / 2);
      result.ca_trigger_times.reserve(num_steps / 2);

      // ------------------------------------------------------------------
      // 3. Tratamento de Perturbação Externa
      // ------------------------------------------------------------------
      const Vector actual_w = w.has_value() ? *w : Vector(0);

      // ------------------------------------------------------------------
      // 4. Mecanismos ETM Desacoplados (SC e CA) e Amostrador Periódico
      // ------------------------------------------------------------------
      StaticETM etm_sc(etm_sc_config, output_dim);
      StaticETM etm_ca(etm_ca_config, state_dim);
      Sampler sampler(sampling_period, 0.0);

      // ------------------------------------------------------------------
      // 5. Matrizes da Planta
      // ------------------------------------------------------------------
      const Matrix A = plant.getA();
      const Matrix B = plant.getB();
      const Matrix C = plant.getC();

      // ------------------------------------------------------------------
      // 6. Dinâmica Contínua Aumentada: z = [x; \hat{x}]
      //    \dot{x}      = A x + B u + w
      //    \dot{\hat{x}} = A \hat{x} + B u
      // ------------------------------------------------------------------
      EDOSolvers::RK5 solver(
          [&plant, &A, &B, state_dim, input_dim](
              double /* t */,
              const Vector &z_sys,
              const Vector &signal) -> Vector
          {
            const Vector x_sys = z_sys.slice(0, state_dim);
            const Vector x_est = z_sys.slice(state_dim, 2 * state_dim);
            const Vector u = signal.slice(0, input_dim);
            const Vector disturbance = signal.slice(input_dim, signal.size());

            const Vector x_dot = plant.stateDerivative(x_sys, u, disturbance);
            const Vector x_est_dot = A * x_est + B * u;

            return Vector::concatenate(x_dot, x_est_dot);
          });

      // ------------------------------------------------------------------
      // 7. Condições Iniciais e Memórias de Transmissão ZOH
      // ------------------------------------------------------------------
      Vector x = x0;
      Vector x_est = x_hat0;
      Vector z = Vector::concatenate(x, x_est);

      Vector y_r_sc = C * x0; // y(t_k^{sc})
      Vector x_r_ca = x_hat0; // \hat{x}(t_\ell^{ca})
      Vector u_actual = K * x_r_ca;

      double last_sc_trigger = 0.0;
      double last_ca_trigger = 0.0;

      // ------------------------------------------------------------------
      // 8. Laço Principal de Simulação
      // ------------------------------------------------------------------
      for (std::size_t i = 0; i < num_steps; ++i)
      {
        const double t = timepts[i];

        // Extração dos estados contínuos no instante t
        x = z.slice(0, state_dim);
        x_est = z.slice(state_dim, 2 * state_dim);

        const Vector y = C * x;

        // Avaliação Periódica nos Instantes de Amostragem s_m
        if (i == 0)
        {
          // Inicialização síncrona no instante t = 0
          result.sc_trigger_times.push_back(t);
          result.ca_trigger_times.push_back(t);
          last_sc_trigger = t;
          last_ca_trigger = t;

          y_r_sc = y;

          // Salto impulsivo do observador:
          // \hat{x}(0^+) = \hat{x}(0) + L(y(0) - C\hat{x}(0))
          x_est = x_est + L * (y - C * x_est);
          x_r_ca = x_est;

          z = Vector::concatenate(x, x_est);
          u_actual = K * x_r_ca;
        }
        else if (sampler.shouldSample(t))
        {
          // --------------------------------------------------------------
          // (a) Avaliação do Canal Sensor-Controlador (SC)
          // --------------------------------------------------------------
          const bool sc_event = etm_sc.evaluateEvent(y, y_r_sc);
          const bool sc_timeout = (t - last_sc_trigger >= max_iet - 1e-9);

          if (sc_event || sc_timeout)
          {
            y_r_sc = y;
            last_sc_trigger = t;
            result.sc_trigger_times.push_back(t);

            // Salto discreto no observador:
            // \hat{x}(t_k^{sc+}) = \hat{x}(t_k^{sc})
            //                    + L(y(t_k^{sc}) - C\hat{x}(t_k^{sc}))
            x_est = x_est + L * (y - C * x_est);
            z = Vector::concatenate(x, x_est);
          }

          // --------------------------------------------------------------
          // (b) Avaliação do Canal Controlador-Atuador (CA)
          // --------------------------------------------------------------
          const bool ca_event = etm_ca.evaluateEvent(x_est, x_r_ca);
          const bool ca_timeout = (t - last_ca_trigger >= max_iet - 1e-9);

          if (ca_event || ca_timeout)
          {
            x_r_ca = x_est;
            last_ca_trigger = t;
            result.ca_trigger_times.push_back(t);

            // Atualização do sinal de controle mantido pelo ZOH do atuador:
            // u(t) = K * \hat{x}(t_\ell^{ca})
            u_actual = K * x_r_ca;
          }
        }

        // ------------------------------------------------------------------
        // 9. Registro de Telemetria
        // ------------------------------------------------------------------
        const Vector estimation_error = x - x_est;

        for (int j = 0; j < state_dim; ++j)
        {
          result.states_data.push_back(x[j]);
          result.estimated_states_data.push_back(x_est[j]);
          result.estimation_error_data.push_back(estimation_error[j]);
        }

        for (int j = 0; j < input_dim; ++j)
        {
          result.control_data.push_back(u_actual[j]);
        }

        // ------------------------------------------------------------------
        // 10. Integração Numérica Contínua no Intervalo [t, t + dt]
        // ------------------------------------------------------------------
        if (i + 1 < num_steps)
        {
          const Vector signal = Vector::concatenate(u_actual, actual_w);
          z = solver.step(t, z, signal, time_step);
        }
      }

      return result;
    }

  } // namespace LIT_SETM
} // namespace PeriodicETC