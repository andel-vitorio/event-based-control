#include "SystemParser.hpp"
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

SystemModel SystemParser::parseFromFile(const std::filesystem::path &filePath)
{
  std::ifstream file(filePath);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open system configuration file: " + filePath.string());
  }

  json data;
  file >> data;

  SystemModel model;
  model.name = data.value("name", "undefined");

  // Process system matrices
  if (data.contains("system_matrices"))
  {
    const auto &matrices = data["system_matrices"];
    auto processMatrix = [&](const std::string &key) -> std::optional<SystemMatrix>
    {
      if (matrices.contains(key) && !matrices[key].is_null())
      {
        return parseMatrix(matrices[key]);
      }
      return std::nullopt;
    };

    model.A = processMatrix("A");
    model.B = processMatrix("B");
    model.C = processMatrix("C");
    model.D = processMatrix("D");
    model.E = processMatrix("E");
    model.F = processMatrix("F");
    model.Cz = processMatrix("Cz");
    model.Dz = processMatrix("Dz");
    model.Fz = processMatrix("Fz");
  }

  // Process metadata maps
  auto parseMap = [&](const std::string &key)
  {
    std::map<std::string, Variable> map;
    if (data.contains(key) && !data[key].is_null())
    {
      for (auto it = data[key].begin(); it != data[key].end(); ++it)
      {
        map[it.key()] = parseVariable(it.value());
      }
    }
    return map;
  };

  model.states = parseMap("states");
  model.parameters = parseMap("parameters");
  model.disturbances = parseMap("disturbances");
  model.inputs = parseMap("inputs");
  model.outputs = parseMap("outputs");

  return model;
}

SystemMatrix SystemParser::parseMatrix(const json &matrixJson)
{
  SystemMatrix matrix;
  for (const auto &row : matrixJson)
  {
    std::vector<MatrixValue> matrixRow;
    for (const auto &val : row)
    {
      if (val.is_number())
      {
        matrixRow.push_back(val.get<double>());
      }
      else if (val.is_string())
      {
        matrixRow.push_back(val.get<std::string>());
      }
      else
      {
        throw std::runtime_error("Invalid matrix element type: expected number or string.");
      }
    }
    matrix.push_back(matrixRow);
  }
  return matrix;
}

Variable SystemParser::parseVariable(const json &varJson)
{
  Variable var;
  var.name = varJson.value("name", "");
  var.symbol = varJson.value("symbol", "");
  var.unit = varJson.value("unit", "");
  if (varJson.contains("value") && !varJson["value"].is_null())
  {
    var.value = varJson["value"].get<std::string>();
  }
  return var;
}