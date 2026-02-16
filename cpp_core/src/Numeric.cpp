#include "Numeric.h"

namespace Numeric
{

  /** Computes the matrix-vector product y = Ax. */
  Vector mat_vec_mul(const Matrix &A, const Vector &x)
  {
    size_t rows = A.size();
    size_t cols = A[0].size();
    Vector y(rows, 0.0);

    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        y[i] += A[i][j] * x[j];
      }
    }
    return y;
  }

  /** Computes the linear combination res = a*x + b*y. */
  Vector vec_add(const Vector &x, const Vector &y, double a, double b)
  {
    size_t n = x.size();
    Vector res(n);
    for (size_t i = 0; i < n; ++i)
    {
      res[i] = a * x[i] + b * y[i];
    }
    return res;
  }

  /** Computes the multiply-accumulate operation res = x + scale*k. */
  Vector vec_mac(const Vector &x, const Vector &k, double scale)
  {
    size_t n = x.size();
    Vector res(n);
    for (size_t i = 0; i < n; ++i)
    {
      res[i] = x[i] + scale * k[i];
    }
    return res;
  }

  /** Performs a single integration step using the Cash-Karp Runge-Kutta method (5th order). */
  Vector rk5_step(const Matrix &A, const Matrix &B, const Vector &x, const Vector &u, double dt)
  {
    Vector Ax = mat_vec_mul(A, x);
    Vector Bu = mat_vec_mul(B, u);
    Vector k1 = vec_add(Ax, Bu);

    Vector x2 = vec_mac(x, k1, dt * 0.25);
    Vector k2 = vec_add(mat_vec_mul(A, x2), Bu);

    Vector temp = vec_add(k1, k2, 3.0 / 32.0, 9.0 / 32.0);
    Vector x3 = vec_mac(x, temp, dt);
    Vector k3 = vec_add(mat_vec_mul(A, x3), Bu);

    temp = vec_add(k1, k2, 1932.0 / 2197.0, -7200.0 / 2197.0);
    temp = vec_add(temp, k3, 1.0, 7296.0 / 2197.0);
    Vector x4 = vec_mac(x, temp, dt);
    Vector k4 = vec_add(mat_vec_mul(A, x4), Bu);

    temp = vec_add(k1, k2, 439.0 / 216.0, -8.0);
    temp = vec_add(temp, k3, 1.0, 3680.0 / 513.0);
    temp = vec_add(temp, k4, 1.0, -845.0 / 4104.0);
    Vector x5 = vec_mac(x, temp, dt);
    Vector k5 = vec_add(mat_vec_mul(A, x5), Bu);

    temp = vec_add(k1, k2, -8.0 / 27.0, 2.0);
    temp = vec_add(temp, k3, 1.0, -3544.0 / 2565.0);
    temp = vec_add(temp, k4, 1.0, 1859.0 / 4104.0);
    temp = vec_add(temp, k5, 1.0, -11.0 / 40.0);
    Vector x6 = vec_mac(x, temp, dt);
    Vector k6 = vec_add(mat_vec_mul(A, x6), Bu);

    Vector sum_k = vec_add(k1, k3, 16.0 / 135.0, 6656.0 / 12825.0);
    sum_k = vec_add(sum_k, k4, 1.0, 28561.0 / 56430.0);
    sum_k = vec_add(sum_k, k5, 1.0, -9.0 / 50.0);
    sum_k = vec_add(sum_k, k6, 1.0, 2.0 / 55.0);

    return vec_mac(x, sum_k, dt);
  }
}