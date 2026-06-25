#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <variant>
#include <ostream>
#include <iomanip>

/**
 * @brief Represents a raw matrix element.
 *
 * A value can be either a numeric constant or a symbolic parameter.
 * Symbolic values are useful for LPV models.
 */
using MatrixValue = std::variant<double, std::string>;

/**
 * @brief Raw matrix representation used during parsing.
 *
 * This representation is intentionally independent from the numerical
 * LinearAlgebra module. It is used as an intermediate representation
 * between JSON files and system models.
 */
using RawMatrix =
    std::vector<std::vector<MatrixValue>>;

/**
 * @brief Represents metadata associated with a system variable.
 */
struct Variable
{

  /**
   * @brief Variable name.
   */
  std::string name;

  /**
   * @brief Mathematical symbol.
   */
  std::string symbol;

  /**
   * @brief Physical unit.
   */
  std::string unit;

  /**
   * @brief Optional initial or nominal value.
   */
  std::optional<std::string> value;
};

/**
 * @brief Intermediate representation of a dynamic system.
 *
 * SystemModel stores the information extracted from configuration files.
 * Numerical conversion and validation are performed by specific system
 * classes such as LITSystem.
 */
struct SystemModel
{

  /**
   * @brief System identifier.
   */
  std::string name;

  /*
   * State-space matrices:
   *
   * x_dot = A*x + B*u + E*w
   *
   * y = C*x + D*u + F*w
   */

  std::optional<RawMatrix> A;

  std::optional<RawMatrix> B;

  std::optional<RawMatrix> C;

  std::optional<RawMatrix> D;

  std::optional<RawMatrix> E;

  std::optional<RawMatrix> F;

  /*
   * Performance output:
   *
   * z = Cz*x + Dz*u + Fz*w
   */

  std::optional<RawMatrix> Cz;

  std::optional<RawMatrix> Dz;

  std::optional<RawMatrix> Fz;

  /**
   * @brief State variables.
   */
  std::map<std::string, Variable> states;

  /**
   * @brief Input variables.
   */
  std::map<std::string, Variable> inputs;

  /**
   * @brief Output variables.
   */
  std::map<std::string, Variable> outputs;

  /**
   * @brief Disturbance variables.
   */
  std::map<std::string, Variable> disturbances;

  /**
   * @brief Model parameters.
   */
  std::map<std::string, Variable> parameters;
};

/**
 * @brief Outputs the matrix to an output stream with dynamic column alignment.
 * @param os The output stream.
 * @param matrix The matrix to be serialized.
 * @return std::ostream& Reference to the modified output stream.
 */
inline std::ostream &operator<<(std::ostream &os, const RawMatrix &matrix)
{
  if (matrix.empty())
  {
    return os << "[]";
  }

  const size_t rows = matrix.size();
  const size_t cols = matrix[0].size();
  std::vector<size_t> col_widths(cols, 0);

  // Calculate maximum width per column
  for (const auto &row : matrix)
  {
    for (size_t j = 0; j < cols; ++j)
    {
      std::ostringstream oss;
      std::visit(
          [&](const auto &value)
          {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, double>)
            {
              oss << std::fixed << std::setprecision(3) << value;
            }
            else
            {
              oss << value;
            }
          },
          row[j]);
      col_widths[j] = std::max(col_widths[j], oss.str().length());
    }
  }

  // Print matrix
  for (size_t i = 0; i < rows; ++i)
  {
    os << (i == 0 ? "[[" : " [");
    for (size_t j = 0; j < cols; ++j)
    {
      std::visit(
          [&](const auto &value)
          {
            using T = std::decay_t<decltype(value)>;
            os << std::right << std::setw(static_cast<int>(col_widths[j]));

            if constexpr (std::is_same_v<T, double>)
            {
              os << std::fixed << std::setprecision(3) << value;
            }
            else
            {
              os << value;
            }
          },
          matrix[i][j]);

      if (j < cols - 1)
      {
        os << "  "; // Inter-column spacing
      }
    }
    os << "]" << (i == rows - 1 ? "]" : "\n");
  }

  return os;
}