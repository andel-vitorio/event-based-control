#ifndef NUMERIC_H
#define NUMERIC_H

#include <vector>
#include <cmath>

namespace Numeric
{

  /**
   * @brief Alias for a vector of doubles.
   */
  using Vector = std::vector<double>;

  /**
   * @brief Alias for a matrix (vector of vectors) of doubles.
   */
  using Matrix = std::vector<std::vector<double>>;

  /**
   * @brief Computes the product of a matrix and a vector.
   *
   * @param A The matrix.
   * @param x The vector.
   * @return Vector The resulting vector Ax.
   */
  Vector mat_vec_mul(const Matrix &A, const Vector &x);

  /**
   * @brief Computes the linear combination of two vectors.
   *
   * Calculates \f$ a \cdot x + b \cdot y \f$.
   *
   * @param x First vector.
   * @param y Second vector.
   * @param a Scalar multiplier for x (default 1.0).
   * @param b Scalar multiplier for y (default 1.0).
   * @return Vector The resulting vector.
   */
  Vector vec_add(const Vector &x, const Vector &y, double a = 1.0, double b = 1.0);

  /**
   * @brief Performs a multiply-accumulate operation.
   *
   * Calculates \f$ x + \text{scale} \cdot k \f$.
   *
   * @param x Base vector.
   * @param k Vector to be scaled and added.
   * @param scale Scalar multiplier for k.
   * @return Vector The resulting vector.
   */
  Vector vec_mac(const Vector &x, const Vector &k, double scale);

  /**
   * @brief Performs a single step of Runge-Kutta 5th order integration.
   *
   * Solves for the next state of the linear system \f$ \dot{x} = Ax + Bu \f$.
   *
   * @param A State matrix.
   * @param B Input matrix.
   * @param x Current state vector.
   * @param u Input vector (assumed constant over the step).
   * @param dt Time step.
   * @return Vector The estimated state at t + dt.
   */
  Vector rk5_step(const Matrix &A, const Matrix &B, const Vector &x, const Vector &u, double dt);

  /**
   * Calcula o valor escalar da forma quadrática x' * M * x.
   * @param x Vetor de estado (n x 1)
   * @param M Matriz de peso (n x n)
   * @return Escalar resultante
   */
  double scalar_quadratic_form(const Vector &x, const Matrix &M);
}

#endif