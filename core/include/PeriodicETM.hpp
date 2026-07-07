#pragma once
#include <vector>

#include "Algebra/Algebra.hpp"

namespace PeriodicETC
{
  struct SimulationParams
  {
    double time_step;
    double final_time;
    Algebra::Vector initial_state;
  };

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

  class StaticSETM
  {
  private:
    Algebra::Matrix Xi;
    Algebra::Matrix Psi;
    Algebra::Vector transmitted_states;
    bool initialized;

  public:
    StaticSETM(const Algebra::Matrix &Xi, const Algebra::Matrix &Psi);
    bool evaluate(const Algebra::Vector &current_x);
    void reset();
  };
}