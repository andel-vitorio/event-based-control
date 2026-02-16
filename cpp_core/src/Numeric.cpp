/**
 * @file Numeric.cpp
 * @brief High-performance linear algebra and integration utilities.
 */

#include "../include/Numeric.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace Numeric
{
  Vector mat_vec_mul(const Matrix &A, const Vector &x)
  {
    Vector y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
      for (size_t j = 0; j < x.size(); ++j)
        y[i] += A[i][j] * x[j];
    return y;
  }

  Matrix mat_mul(const Matrix &A, const Matrix &B)
  {
    Matrix res(A.size(), Vector(B[0].size(), 0.0));
    for (size_t i = 0; i < A.size(); ++i)
      for (size_t k = 0; k < A[0].size(); ++k)
        for (size_t j = 0; j < B[0].size(); ++j)
          res[i][j] += A[i][k] * B[k][j];
    return res;
  }

  Vector vec_add(const Vector &x, const Vector &y, double a, double b)
  {
    Vector res(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      res[i] = a * x[i] + b * y[i];
    return res;
  }

  Matrix mat_add(const Matrix &A, const Matrix &B, double a, double b)
  {
    Matrix res(A.size(), Vector(A[0].size()));
    for (size_t i = 0; i < A.size(); ++i)
      for (size_t j = 0; j < A[0].size(); ++j)
        res[i][j] = a * A[i][j] + b * B[i][j];
    return res;
  }

  Matrix mat_scalar_mul(const Matrix &A, double scalar)
  {
    Matrix res = A;
    for (auto &row : res)
      for (auto &val : row)
        val *= scalar;
    return res;
  }

  Matrix identity(size_t n)
  {
    Matrix res(n, Vector(n, 0.0));
    for (size_t i = 0; i < n; ++i)
      res[i][i] = 1.0;
    return res;
  }

  Matrix expm(const Matrix &M)
  {
    const size_t n = M.size();
    Matrix res = identity(n);
    Matrix term = identity(n);

    for (int i = 1; i <= 30; ++i)
    {
      term = mat_mul(term, M);
      term = mat_scalar_mul(term, 1.0 / static_cast<double>(i));
      res = mat_add(res, term);

      double norm = 0.0;
      for (const auto &row : term)
        for (double val : row)
          norm += std::abs(val);

      if (norm < 1e-18)
        break;
    }
    return res;
  }

  void discretize_zoh(const Matrix &A, const Matrix &B, double h, Matrix &Ad, Matrix &Bd)
  {
    const size_t nx = A.size();
    const size_t nu = B[0].size();
    const size_t dim = nx + nu;
    Matrix M(dim, Vector(dim, 0.0));

    for (size_t i = 0; i < nx; ++i)
    {
      for (size_t j = 0; j < nx; ++j)
        M[i][j] = A[i][j] * h;
      for (size_t j = 0; j < nu; ++j)
        M[i][nx + j] = B[i][j] * h;
    }

    Matrix expM = expm(M);
    Ad = Matrix(nx, Vector(nx));
    Bd = Matrix(nx, Vector(nu));

    for (size_t i = 0; i < nx; ++i)
    {
      for (size_t j = 0; j < nx; ++j)
        Ad[i][j] = expM[i][j];
      for (size_t j = 0; j < nu; ++j)
        Bd[i][j] = expM[i][nx + j];
    }
  }

  double scalar_quadratic_form(const Vector &x, const Matrix &M)
  {
    double result = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
      double tmp = 0.0;
      for (size_t j = 0; j < x.size(); ++j)
        tmp += M[i][j] * x[j];
      result += x[i] * tmp;
    }
    return result;
  }

  Vector vec_mac(const Vector &x, const Vector &k, double scale)
  {
    Vector res(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      res[i] = x[i] + scale * k[i];
    return res;
  }

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