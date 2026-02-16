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
   * @param A The matrix.
   * @param x The vector.
   * @return Vector The resulting vector Ax.
   */
  Vector mat_vec_mul(const Matrix &A, const Vector &x);

  /**
   * @brief Computes the product of two matrices.
   * @param A Left matrix.
   * @param B Right matrix.
   * @return Matrix Resulting matrix AB.
   */
  Matrix mat_mul(const Matrix &A, const Matrix &B);

  /**
   * @brief Computes the linear combination of two vectors.
   * Calculates a*x + b*y.
   */
  Vector vec_add(const Vector &x, const Vector &y, double a = 1.0, double b = 1.0);

  /**
   * @brief Computes the sum of two matrices with optional scaling.
   * Calculates a*A + b*B.
   */
  Matrix mat_add(const Matrix &A, const Matrix &B, double a = 1.0, double b = 1.0);

  /**
   * @brief Multiplies a matrix by a scalar.w
   */
  Matrix mat_scalar_mul(const Matrix &A, double scalar);

  /**
   * @brief Returns an identity matrix of size n.
   */
  Matrix identity(size_t n);

  /**
   * @brief Computes the inverse of a square matrix using Gauss-Jordan elimination.
   */
  Matrix inverse(const Matrix &A);

  /**
   * @brief Computes the matrix exponential exp(M) using Scaling and Squaring with Pade approximation.
   * Matches the behavior of scipy.linalg.expm.
   */
  Matrix expm(const Matrix &M);

  /**
   * @brief Performs exact Zero-Order Hold (ZOH) discretization.
   * Matches scipy.signal.cont2discrete exactly by using the augmented matrix exponential.
   */
  void discretize_zoh(const Matrix &A, const Matrix &B, double h, Matrix &Ad, Matrix &Bd);

  /**
   * @brief Performs a multiply-accumulate operation for vectors.
   */
  Vector vec_mac(const Vector &x, const Vector &k, double scale);

  /**
   * @brief Performs a single step of Runge-Kutta 5th order integration.
   * Solves for the next state of the linear system x_dot = Ax + Bu.
   */
  Vector rk5_step(const Matrix &A, const Matrix &B, const Vector &x, const Vector &u, double dt);

  /**
   * @brief Computes the scalar value of the quadratic form x' * M * x.
   * @param x Input vector (n x 1).
   * @param M Weighting matrix (n x n).
   * @return double Resulting scalar value.
   */
  double scalar_quadratic_form(const Vector &x, const Matrix &M);
}

#endif