#include "../../include/EDOSolvers/EDOSolvers.hpp"

namespace EDOSolvers
{

  RK5::RK5(
      SystemFunction system)
      : system_(std::move(system))
  {
  }

  void RK5::resize(
      std::size_t n)
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
    state_ = Vector(n);
  }

  RK5::Vector RK5::step(
      double t,
      const Vector &x,
      const Vector &u,
      double dt)
  {

    resize(
        x.size());

    k1_ =
        system_(
            t,
            x,
            u);

    state_ =
        x +
        k1_ * (dt * 0.25);

    k2_ =
        system_(
            t + dt * 0.25,
            state_,
            u);

    temp_ =
        k1_ * (3.0 / 32.0) +
        k2_ * (9.0 / 32.0);

    state_ =
        x +
        temp_ * dt;

    k3_ =
        system_(
            t + dt * 3.0 / 8.0,
            state_,
            u);

    temp_ =
        k1_ * (1932.0 / 2197.0) +
        k2_ * (-7200.0 / 2197.0) +
        k3_ * (7296.0 / 2197.0);

    state_ =
        x +
        temp_ * dt;

    k4_ =
        system_(
            t + dt * 12.0 / 13.0,
            state_,
            u);

    temp_ =
        k1_ * (439.0 / 216.0) +
        k2_ * (-8.0) +
        k3_ * (3680.0 / 513.0) +
        k4_ * (-845.0 / 4104.0);

    state_ =
        x +
        temp_ * dt;

    k5_ =
        system_(
            t + dt,
            state_,
            u);

    temp_ =
        k1_ * (-8.0 / 27.0) +
        k2_ * (2.0) +
        k3_ * (-3544.0 / 2565.0) +
        k4_ * (1859.0 / 4104.0) +
        k5_ * (-11.0 / 40.0);

    state_ =
        x +
        temp_ * dt;

    k6_ =
        system_(
            t + dt * 0.5,
            state_,
            u);

    temp_ =
        k1_ * (16.0 / 135.0) +
        k3_ * (6656.0 / 12825.0) +
        k4_ * (28561.0 / 56430.0) +
        k5_ * (-9.0 / 50.0) +
        k6_ * (2.0 / 55.0);

    return x + temp_ * dt;
  }

}