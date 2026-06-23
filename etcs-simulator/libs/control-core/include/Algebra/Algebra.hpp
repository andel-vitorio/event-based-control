#pragma once

#include "Matrix.hpp"
#include "Vector.hpp"
#include "Expression.hpp"
#include "Lexer.hpp"
#include "Parser.hpp"
#include "Token.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <type_traits>
#include <cstdint>

namespace Algebra
{

  /**
   * @brief Generates values in the interval [start, stop).
   * @tparam T Floating point type.
   */
  template <typename T>
  std::vector<T> arange(
      T start,
      T stop,
      T step = static_cast<T>(1))
  {
    static_assert(
        std::is_floating_point<T>::value,
        "Only floating point types are supported.");

    if (step == 0)
      throw std::invalid_argument(
          "Step size cannot be zero.");

    std::vector<T> result;

    for (T i = start; i < stop; i += step)
    {
      result.push_back(i);
    }

    return result;
  }

  /**
   * @brief Generates num evenly spaced samples.
   */
  template <typename T>
  std::vector<T> linspace(
      T start,
      T stop,
      size_t num)
  {
    static_assert(
        std::is_floating_point<T>::value,
        "Only floating point types are supported.");

    std::vector<T> result;

    if (num == 0)
      return result;

    result.reserve(num);

    if (num == 1)
    {
      result.push_back(start);
      return result;
    }

    T step =
        (stop - start) /
        static_cast<T>(num - 1);

    for (size_t i = 0; i < num; ++i)
    {
      result.push_back(
          start + step * static_cast<T>(i));
    }

    result.back() = stop;

    return result;
  }

  /**
   * @brief Generates values evenly spaced on log scale.
   */
  template <typename T>
  std::vector<T> logspace(
      T start,
      T stop,
      size_t num,
      T base = static_cast<T>(10))
  {
    std::vector<T> result =
        linspace(start, stop, num);

    for (auto &value : result)
    {
      value = std::pow(base, value);
    }

    return result;
  }

  /**
   * @brief Optimized TimeSeries container.
   */
  template <typename T, typename Time = uint64_t>
  class TimeSeries
  {
  public:
    void push_back(
        Time time,
        T value)
    {
      if (!timestamps.empty() &&
          time <= timestamps.back())
      {
        throw std::invalid_argument(
            "Timestamps must be strictly increasing.");
      }

      timestamps.push_back(time);
      values.push_back(value);
    }

    void reserve(size_t n)
    {
      timestamps.reserve(n);
      values.reserve(n);
    }

    const std::vector<Time> &get_timestamps() const
    {
      return timestamps;
    }

    const std::vector<T> &get_values() const
    {
      return values;
    }

    size_t size() const
    {
      return values.size();
    }

  private:
    std::vector<Time> timestamps;
    std::vector<T> values;
  };

}