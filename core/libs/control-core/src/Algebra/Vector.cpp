#include "../../include/Algebra/Algebra.hpp"

#include <stdexcept>

namespace Algebra
{

  Vector::Vector(std::size_t size)
      : values_(size, 0.0)
  {
  }

  double &Vector::operator[](std::size_t index)
  {
    return values_[index];
  }

  double Vector::operator[](std::size_t index) const
  {
    return values_[index];
  }

  std::size_t Vector::size() const noexcept
  {
    return values_.size();
  }

  std::vector<double> &Vector::data() noexcept
  {
    return values_;
  }

  const std::vector<double> &Vector::data() const noexcept
  {
    return values_;
  }

  Vector &Vector::operator+=(
      const Vector &other)
  {

    if (values_.size() != other.size())
    {
      throw std::invalid_argument(
          "Vector dimension mismatch");
    }

    for (std::size_t i = 0; i < values_.size(); ++i)
    {
      values_[i] += other[i];
    }

    return *this;
  }

  Vector operator+(
      const Vector &lhs,
      const Vector &rhs)
  {

    Vector result = lhs;

    result += rhs;

    return result;
  }

  std::ostream &operator<<(
      std::ostream &os,
      const Vector &v)
  {

    os << "[";

    for (std::size_t i = 0; i < v.size(); ++i)
    {

      os << v[i];

      if (i + 1 < v.size())
        os << ", ";
    }

    os << "]";

    return os;
  }

  Vector operator*(
      const Vector &v,
      double scalar)
  {

    Vector result(v.size());

    for (std::size_t i = 0; i < v.size(); ++i)
    {
      result[i] = v[i] * scalar;
    }

    return result;
  }

  Vector operator*(
      double scalar,
      const Vector &v)
  {
    return v * scalar;
  }

  Vector &Vector::operator-=(
      const Vector &other)
  {

    if (values_.size() != other.size())
    {
      throw std::invalid_argument(
          "Vector dimension mismatch");
    }

    for (std::size_t i = 0; i < values_.size(); ++i)
    {
      values_[i] -= other[i];
    }

    return *this;
  }

  Vector operator-(
      const Vector &lhs,
      const Vector &rhs)
  {

    Vector result = lhs;

    result -= rhs;

    return result;
  }
}