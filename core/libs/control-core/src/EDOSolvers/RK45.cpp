#include "../../include/EDOSolvers/EDOSolvers.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace EDOSolvers
{
  RK45::RK45(SystemFunction system)
      : system_(std::move(system))
  {
  }

  RK45::RK45(SystemFunction system, const AdaptiveSolver::Options &options)
      : AdaptiveSolver(options),
        system_(std::move(system))
  {
  }

  void RK45::resize(std::size_t n)
  {
    if (k1_.size() == n)
      return;

    k1_ = Vector(n);
    k2_ = Vector(n);
    k3_ = Vector(n);
    k4_ = Vector(n);
    k5_ = Vector(n);
    k6_ = Vector(n);

    temp_ = Vector(n);

    x4_ = Vector(n);
    x5_ = Vector(n);
  }

  RK45::Result RK45::step(
      double t,
      const Vector &x,
      const Vector &u,
      double dt)
  {
    resize(x.size());

    k1_ = system_(t, x, u);

    k2_ = system_(t + dt * 0.25, x + k1_ * (dt * 0.25), u);

    k3_ = system_(t + 3.0 * dt / 8.0,
                  x + k1_ * (3.0 * dt / 32.0) + k2_ * (9.0 * dt / 32.0), u);

    k4_ = system_(t + 12.0 * dt / 13.0,
                  x + k1_ * (1932.0 * dt / 2197.0) - k2_ * (7200.0 * dt / 2197.0) + k3_ * (7296.0 * dt / 2197.0), u);

    k5_ = system_(t + dt,
                  x + k1_ * (439.0 * dt / 216.0) - k2_ * (8.0 * dt) + k3_ * (3680.0 * dt / 513.0) - k4_ * (845.0 * dt / 4104.0), u);

    k6_ = system_(t + dt * 0.5,
                  x - k1_ * (8.0 * dt / 27.0) + k2_ * (2.0 * dt) - k3_ * (3544.0 * dt / 2565.0) + k4_ * (1859.0 * dt / 4104.0) - k5_ * (11.0 * dt / 40.0), u);

    x4_ = x + k1_ * (25.0 * dt / 216.0) + k3_ * (1408.0 * dt / 2565.0) + k4_ * (2197.0 * dt / 4104.0) - k5_ * (dt / 5.0);

    x5_ = x + k1_ * (16.0 * dt / 135.0) + k3_ * (6656.0 * dt / 12825.0) + k4_ * (28561.0 * dt / 56430.0) - k5_ * (9.0 * dt / 50.0) + k6_ * (2.0 * dt / 55.0);

    Vector error = x5_ - x4_;

    // Call to the inherited errorNorm template method from AdaptiveSolver
    double norm = errorNorm(error, x, x5_);

    const bool accepted = norm <= 1.0;

    if (accepted)
      ++accepted_;
    else
      ++rejected_;

    // Call to the inherited nextStep method from AdaptiveSolver (RK45 uses order 5)
    double new_dt = nextStep(dt, norm, 5);

    return {accepted ? x5_ : x, error, norm, dt, new_dt, accepted};
  }

  std::size_t RK45::acceptedSteps() const noexcept
  {
    return accepted_;
  }

  std::size_t RK45::rejectedSteps() const noexcept
  {
    return rejected_;
  }
}