#include "PeriodicETC/LIT/simulators/open_loop_simulator.hpp"
#include "EDOSolvers/EDOSolvers.hpp"

namespace PeriodicETC
{
  namespace LIT
  {

    OpenLoopResult OpenLoopSimulator::run(
        ControlSystems::LITSystem &plant,
        const Algebra::Vector &x0,
        std::optional<Algebra::Vector> u,
        std::optional<Algebra::Vector> w,
        double duration,
        double time_step)
    {
      auto timepts = Algebra::arange(0.0, duration, time_step);
      int num_steps = static_cast<int>(timepts.size());
      int state_dim = static_cast<int>(x0.size());

      using Vector = Algebra::Vector;

      OpenLoopResult result;
      result.time_data = timepts;
      result.states_data.reserve(num_steps * state_dim);

      Vector actual_u = u.has_value() ? *u : Vector(plant.inputs());
      Vector actual_w = w.has_value() ? *w : Vector(0);

      EDOSolvers::RK5 solver(
          [&plant]([[maybe_unused]] double t, const Vector &x,
                   const Vector &signal)
          {
            std::size_t nu = plant.inputs();
            Vector ut = signal.slice(0, nu);
            Vector wt = signal.slice(nu, signal.size());
            return plant.stateDerivative(x, ut, wt);
          });

      Vector x = x0;
      Vector signal = Vector::concatenate(actual_u, actual_w);

      for (const auto t : timepts)
      {
        for (int i = 0; i < state_dim; ++i)
        {
          result.states_data.push_back(x[i]);
        }
        x = solver.step(t, x, signal, time_step);
      }

      return result;
    }

  } // namespace LIT
} // namespace PeriodicETC