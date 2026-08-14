#include "PeriodicETC/PeriodicETM.hpp"
#include <iostream>

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

  namespace LIT_SETM
  {
    using namespace Algebra;

    StaticETM::StaticETM(const StaticETMConfig &config, int n_states) : config_(config)
    {
      if (config_.Psi.rows() == 0 || config_.Psi.cols() == 0)
      {
        std::cerr
            << "[WARNING] StaticETM: Psi was not provided. "
            << "Using the identity matrix as default."
            << std::endl;

        config_.Psi = identity(n_states);
      }

      if (config_.Xi.rows() == 0 || config_.Xi.cols() == 0)
      {
        std::cerr
            << "[WARNING] StaticETM: Xi was not provided. "
            << "Using the identity matrix as default."
            << std::endl;

        config_.Xi = identity(n_states);
      }
    }

    bool StaticETM::evaluateEvent(const Vector &current_state,
                                  const Vector &last_sampled_state) const
    {
      Vector epsilon = last_sampled_state - current_state;
      double state_term = quadratic_form(current_state, config_.Psi);
      double error_term = quadratic_form(epsilon, config_.Xi);
      double gamma = config_.sigma * state_term - error_term;
      return gamma < config_.threshold;
    }
  } // namespace LIT
} // namespace PeriodicETC