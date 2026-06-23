#pragma once

#include <algorithm>
#include <cmath>

namespace EDOSolvers
{

  class AdaptiveSolver
  {

  public:
    struct Options
    {
      double absolute_tolerance = 1e-12;
      double relative_tolerance = 1e-10;

      double safety_factor = 0.9;

      double min_factor = 0.2;
      double max_factor = 5.0;

      double min_step = 1e-12;
      double max_step = 1.0;
    };

    // Overload 1: Default constructor uses default-initialized Options
    AdaptiveSolver()
        : options_()
    {
    }

    // Overload 2: Explicit constructor allowing custom Options pass
    explicit AdaptiveSolver(const Options &options)
        : options_(options)
    {
    }

  protected:
    /**
     * @brief Computes normalized local error.
     *
     * Values below 1 indicate that the step
     * satisfies the requested tolerance.
     */
    template <typename Vector>
    double errorNorm(
        const Vector &error,
        const Vector &old_state,
        const Vector &new_state) const
    {

      double sum = 0.0;

      for (std::size_t i = 0;
           i < error.size();
           ++i)
      {

        const double scale =
            options_.absolute_tolerance +
            options_.relative_tolerance *
                std::max(
                    std::abs(old_state[i]),
                    std::abs(new_state[i]));

        const double value =
            error[i] / scale;

        sum += value * value;
      }

      return std::sqrt(
          sum / error.size());
    }

    /**
     * @brief Computes the next integration step.
     */
    double nextStep(
        double dt,
        double error,
        int order) const
    {

      if (error <= 0.0)
        return std::min(
            dt * options_.max_factor,
            options_.max_step);

      double factor =
          options_.safety_factor *
          std::pow(
              1.0 / error,
              1.0 / order);

      factor =
          std::clamp(
              factor,
              options_.min_factor,
              options_.max_factor);

      double next =
          dt * factor;

      return std::clamp(
          next,
          options_.min_step,
          options_.max_step);
    }

    const Options &options() const
    {
      return options_;
    }

  private:
    Options options_;
  };

}

#include "RK5.hpp"
#include "RK45.hpp"