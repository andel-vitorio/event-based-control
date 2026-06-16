#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <variant>

/**
 * @brief Represents a single element in a system matrix.
 * Can be a fixed numeric value or a symbolic reference to a parameter/state.
 */
using MatrixValue = std::variant<double, std::string>;

/**
 * @brief A matrix representation using MatrixValue elements.
 */
using SystemMatrix = std::vector<std::vector<MatrixValue>>;

/**
 * @brief Metadata for system variables.
 */
struct Variable
{
  std::string name;
  std::string symbol;
  std::string unit;
  std::optional<std::string> value;
};

/**
 * @brief Data structure for system configuration, acting as an intermediate layer.
 */
struct SystemModel
{
  std::string name;

  std::optional<SystemMatrix> A;
  std::optional<SystemMatrix> B;
  std::optional<SystemMatrix> C;
  std::optional<SystemMatrix> D;
  std::optional<SystemMatrix> E;
  std::optional<SystemMatrix> F;
  std::optional<SystemMatrix> Cz;
  std::optional<SystemMatrix> Dz;
  std::optional<SystemMatrix> Fz;

  std::map<std::string, Variable> states;
  std::map<std::string, Variable> parameters;
  std::map<std::string, Variable> disturbances;
  std::map<std::string, Variable> inputs;
  std::map<std::string, Variable> outputs;
};