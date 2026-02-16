#ifndef DYNAMICAL_SYSTEM_H
#define DYNAMICAL_SYSTEM_H

#include "Numeric.h"

/**
 * @brief Represents a linear time-invariant (LTI) state-space system.
 *
 * The system is defined by the equations:
 * \f[
 * \dot{x}(t) = A x(t) + B u(t) \\
 * y(t) = C x(t) + D u(t)
 * \f]
 * where x is the state vector, u is the input vector, and y is the output vector.
 */
struct StateSpace
{
  Numeric::Matrix A; ///< State matrix (nx x nx).
  Numeric::Matrix B; ///< Input matrix (nx x nu).
  Numeric::Matrix C; ///< Output matrix (ny x nx).
  Numeric::Matrix D; ///< Feedthrough matrix (ny x nu).

  /**
   * @brief Gets the number of states (nx).
   * @return The dimension of the state vector.
   */
  int nx() const { return A.size(); }

  /**
   * @brief Gets the number of inputs (nu).
   * @return The dimension of the input vector.
   */
  int nu() const { return B[0].size(); }

  /**
   * @brief Gets the number of outputs (ny).
   * @return The dimension of the output vector.
   */
  int ny() const { return C.size(); }
};

#endif