#pragma once

#include "Algebra/Algebra.hpp"

namespace PeriodicETC
{
  class Sampler
  {
  private:
    double sampling_time;
    double next_sample_time;
    const double eps = 1e-9;

  public:
    explicit Sampler(double interval, double t0 = 0.0)
        : sampling_time(interval), next_sample_time(t0) {}

    bool shouldSample(double t);
    void reset(double t0);
  };

  namespace LIT_SETM
  {
    struct StaticETMConfig
    {
      double sigma = 1.0;
      double threshold = 0.0;
      Algebra::Matrix Psi;
      Algebra::Matrix Xi;
    };

    class StaticETM
    {
    private:
      StaticETMConfig config_;

    public:
      explicit StaticETM(const StaticETMConfig &config, int n_states);
      ~StaticETM() = default;

      bool evaluateEvent(const Algebra::Vector &current_state, const Algebra::Vector &last_sampled_state) const;
    };
  } // namespace LIT_SETM
} // namespace PeriodicETC