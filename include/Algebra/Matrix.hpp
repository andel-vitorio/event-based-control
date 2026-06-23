#pragma once

#include <vector>
#include <cstddef>
#include <ostream>

#include "Vector.hpp"

namespace Algebra
{

  class Matrix
  {

  public:
    Matrix(
        std::size_t rows,
        std::size_t cols);

    double &operator()(
        std::size_t row,
        std::size_t col);

    double operator()(
        std::size_t row,
        std::size_t col) const;

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;

    Vector operator*(const Vector &v) const;

    friend std::ostream &operator<<(
        std::ostream &os,
        const Matrix &M);

  private:
    std::size_t rows_;
    std::size_t cols_;

    std::vector<double> data_;
  };

}