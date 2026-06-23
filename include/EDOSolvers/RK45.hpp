#pragma once

#include "../Algebra/Vector.hpp"
#include "EdOSolvers.hpp"

#include <functional>
#include <cstddef>

namespace EDOSolvers
{

  class RK45 : public AdaptiveSolver
  {

  public:
    using Vector = Algebra::Vector;

    /**
     * @brief System dynamic function mapping dx/dt = f(t, x, u).
     */
    using SystemFunction = std::function<Vector(double, const Vector &, const Vector &)>;

    /**
     * @brief Output tracking metadata for an adaptive time-step execution.
     */
    struct Result
    {
      Vector state;
      Vector error;
      double error_norm;
      double dt_used;
      double next_step;
      bool accepted;
    };

  public:
    explicit RK45(SystemFunction system);
    RK45(SystemFunction system, const Options &options);

    /**
     * @brief Computes the next suggested time step based on error threshold.
     */
    double computeNextStep(double dt, double error_norm) const;

    /**
     * @brief Executes a single step execution of the RK45 integration method.
     */
    Result step(double t, const Vector &x, const Vector &u, double dt);

    std::size_t acceptedSteps() const noexcept;
    std::size_t rejectedSteps() const noexcept;

  private:
    SystemFunction system_;

    Options options_;

    Vector k1_;
    Vector k2_;
    Vector k3_;
    Vector k4_;
    Vector k5_;
    Vector k6_;
    Vector temp_;
    Vector x4_;
    Vector x5_;

    void resize(std::size_t n);

    double computeStep(double dt, double error);

  private:
    std::size_t accepted_ = 0;
    std::size_t rejected_ = 0;
  };

}