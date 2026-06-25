#include "../include/LITSystem.hpp"

#include <stdexcept>

LITSystem::LITSystem(
    const SystemModel &model)
    : A_(extractMatrix(model.A)),
      B_(extractMatrix(model.B)),
      C_(extractMatrix(model.C)),
      D_(extractMatrix(model.D)),
      E_(extractMatrix(model.E)),
      F_(extractMatrix(model.F)),
      Cz_(extractMatrix(model.Cz)),
      Dz_(extractMatrix(model.Dz)),
      Fz_(extractMatrix(model.Fz))
{
  validate();
}

Algebra::Matrix LITSystem::extractMatrix(
    const std::optional<RawMatrix> &mat)
{
  if (!mat.has_value())
  {
    return Matrix(0, 0);
  }

  const auto &source = *mat;

  std::size_t rows = source.size();

  if (rows == 0)
    return Matrix(0, 0);

  std::size_t cols = source[0].size();

  Matrix result(rows, cols);

  for (std::size_t i = 0; i < rows; ++i)
  {

    for (std::size_t j = 0; j < cols; ++j)
    {

      const auto &element = source[i][j];

      if (std::holds_alternative<double>(element))
      {

        result(i, j) =
            std::get<double>(element);
      }
      else
      {

        throw std::runtime_error(
            "Matrix contains non numeric value");
      }
    }
  }

  return result;
}

LITSystem::Vector LITSystem::stateDerivative(
    const Vector &x,
    const Vector &u,
    const Vector &w) const
{

  return (A_ * x) +
         (B_ * u) +
         (E_ * w);
}

LITSystem::Vector LITSystem::output(
    const Vector &x,
    const Vector &u,
    const Vector &w) const
{

  return (C_ * x) +
         (D_ * u) +
         (F_ * w);
}

LITSystem::Vector LITSystem::performance(
    const Vector &x,
    const Vector &u,
    const Vector &w) const
{

  return (Cz_ * x) +
         (Dz_ * u) +
         (Fz_ * w);
}

std::size_t LITSystem::states() const noexcept
{
  return A_.rows();
}

std::size_t LITSystem::inputs() const noexcept
{
  return B_.cols();
}

std::size_t LITSystem::outputs() const noexcept
{
  return C_.rows();
}

void LITSystem::validate() const
{

  if (A_.rows() != A_.cols())
    throw std::runtime_error(
        "Matrix A must be square.");

  if (A_.cols() != B_.rows())
    throw std::runtime_error(
        "Matrix dimensions A and B are incompatible.");

  if (A_.cols() != E_.rows())
    throw std::runtime_error(
        "Matrix dimensions A and E are incompatible.");
}