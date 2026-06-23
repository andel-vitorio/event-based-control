#pragma once

#include "../Algebra/Vector.hpp"

#include <functional>

namespace EDOSolvers
{

  /**
   * @brief Generic fifth-order Runge-Kutta solver.
   *
   * Solves generic first-order systems:
   *
   *      x_dot = f(t, x, u)
   *
   * where:
   *
   *      x : state vector
   *      u : input vector
   *
   */
  class RK5
  {

  public:
    using Vector = Algebra::Vector;

    /**
     * @brief Function describing the dynamic system.
     *
     * @param t Current simulation time.
     * @param x Current state.
     * @param u Current input.
     *
     * @return State derivative.
     */
    using SystemFunction =
        std::function<Vector(
            double,
            const Vector &,
            const Vector &)>;

    /**
     * @brief Creates an RK5 solver.
     *
     * @param system Dynamic system function.
     */
    explicit RK5(
        SystemFunction system);

    /**
     * @brief Performs one RK5 integration step.
     *
     * @param t Current time.
     * @param x Current state.
     * @param u Input.
     * @param dt Integration step.
     *
     * @return Estimated next state.
     */
    Vector step(
        double t,
        const Vector &x,
        const Vector &u,
        double dt);

  private:
    SystemFunction system_;

    Vector k1_;
    Vector k2_;
    Vector k3_;
    Vector k4_;
    Vector k5_;
    Vector k6_;

    Vector temp_;
    Vector state_;

    void resize(
        std::size_t n);
  };

}