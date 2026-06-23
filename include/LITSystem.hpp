#pragma once

#include "SystemModel.hpp"

#include "Algebra/Algebra.hpp"

#include <cstddef>
#include <optional>

class LITSystem
{

public:
  using Vector = Algebra::Vector;
  using Matrix = Algebra::Matrix;

public:
  explicit LITSystem(const SystemModel &model);

  /**
   * @brief Computes x_dot = A*x + B*u + E*w
   */
  Vector stateDerivative(
      const Vector &x,
      const Vector &u,
      const Vector &w) const;

  /**
   * @brief Computes system output.
   */
  Vector output(
      const Vector &x,
      const Vector &u,
      const Vector &w) const;

  /**
   * @brief Computes performance output.
   */
  Vector performance(
      const Vector &x,
      const Vector &u,
      const Vector &w) const;

  std::size_t states() const noexcept;
  std::size_t inputs() const noexcept;
  std::size_t outputs() const noexcept;

private:
  Matrix A_;
  Matrix B_;
  Matrix C_;
  Matrix D_;
  Matrix E_;
  Matrix F_;

  Matrix Cz_;
  Matrix Dz_;
  Matrix Fz_;

private:
  static Matrix extractMatrix(
      const std::optional<RawMatrix> &mat);

  void validate() const;
};