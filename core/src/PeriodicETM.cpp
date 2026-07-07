#include "PeriodicETM.hpp"
#include "Algebra/Algebra.hpp"

namespace PeriodicETC
{
  bool Sampler::shouldSample(double t)
  {
    if (t >= (next_sample_time - eps))
    {
      next_sample_time += sampling_time;
      return true;
    }
    return false;
  }

  void Sampler::reset(double t0)
  {
    next_sample_time = t0;
  }

  StaticSETM::StaticSETM(const Algebra::Matrix &Xi, const Algebra::Matrix &Psi)
      : Xi(Xi), Psi(Psi), initialized(false) {}

  bool StaticSETM::evaluate(const Algebra::Vector &current_states)
  {
    using Vector = Algebra::Vector;
    if (!initialized)
    {
      transmitted_states = current_states;
      initialized = true;
      return true;
    }

    Algebra::Vector error = transmitted_states - current_states;

    double x_norm = Vector::dot(current_states.T(), Psi * current_states);
    double epsilon_norm = Vector::dot(error.T(), Psi * error);
    double val = x_norm - epsilon_norm;

    if (val < 0.0)
    {
      transmitted_states = current_states;
      return true;
    }

    return false;
  }

  void StaticSETM::reset()
  {
    initialized = false;
  }
}