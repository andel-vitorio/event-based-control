#include "../../include/Algebra/Algebra.hpp"

#include <stdexcept>

namespace Algebra
{

  Matrix::Matrix(
      std::size_t rows,
      std::size_t cols)
      : rows_(rows),
        cols_(cols),
        data_(rows * cols, 0.0)
  {
  }

  double &Matrix::operator()(
      std::size_t row,
      std::size_t col)
  {
    return data_[row * cols_ + col];
  }

  double Matrix::operator()(
      std::size_t row,
      std::size_t col) const
  {
    return data_[row * cols_ + col];
  }

  std::size_t Matrix::rows() const noexcept
  {
    return rows_;
  }

  std::size_t Matrix::cols() const noexcept
  {
    return cols_;
  }

  Vector Matrix::operator*(const Vector &v) const
  {

    if (v.size() != cols_)
    {
      throw std::invalid_argument(
          "Matrix/vector dimension mismatch");
    }

    Vector result(rows_);

    for (std::size_t i = 0; i < rows_; ++i)
    {
      double sum = 0.0;

      for (std::size_t j = 0; j < cols_; ++j)
      {
        sum += (*this)(i, j) * v[j];
      }

      result[i] = sum;
    }

    return result;
  }

  std::ostream &operator<<(
      std::ostream &os,
      const Matrix &M)
  {

    for (std::size_t i = 0; i < M.rows(); ++i)
    {
      os << "[";

      for (std::size_t j = 0; j < M.cols(); ++j)
      {

        os << M(i, j);

        if (j + 1 < M.cols())
          os << ", ";
      }

      os << "]";

      if (i + 1 < M.rows())
        os << '\n';
    }

    return os;
  }
}