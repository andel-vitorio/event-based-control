#pragma once

#include <vector>
#include <cstddef>
#include <ostream>

namespace Algebra
{

  class Vector
  {

  public:
    explicit Vector(std::size_t size = 0);

    double &operator[](std::size_t index);

    double operator[](std::size_t index) const;

    std::size_t size() const noexcept;

    std::vector<double> &data() noexcept;

    const std::vector<double> &data() const noexcept;

    Vector &operator+=(
        const Vector &other);

    friend Vector operator+(
        const Vector &lhs,
        const Vector &rhs);

    friend std::ostream &operator<<(
        std::ostream &os,
        const Vector &v);

    friend Vector operator*(
        const Vector &v,
        double scalar);

    friend Vector operator*(
        double scalar,
        const Vector &v);

    Vector &operator-=(const Vector &other);

    friend Vector operator-(
        const Vector &lhs,
        const Vector &rhs);

  private:
    std::vector<double> values_;
  };

}