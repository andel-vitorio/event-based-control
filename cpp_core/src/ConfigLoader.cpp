#include "ConfigLoader.h"
#include "../lib/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

namespace
{
  // Helper to parse a JSON Array into a Matrix.
  // Handles values represented as both numbers and strings.
  Numeric::Matrix parse_matrix(const json &j_mat)
  {
    Numeric::Matrix mat;

    if (j_mat.is_null())
    {
      return mat; // Return empty if null
    }

    for (const auto &row_json : j_mat)
    {
      Numeric::Vector row;
      for (const auto &val_json : row_json)
      {
        double val = 0.0;
        // Handle values represented as strings (e.g., "3.75") or numbers
        if (val_json.is_string())
        {
          val = std::stod(val_json.get<std::string>());
        }
        else if (val_json.is_number())
        {
          val = val_json.get<double>();
        }
        row.push_back(val);
      }
      mat.push_back(row);
    }
    return mat;
  }

  // Helper to generate a zero matrix if D is null/empty.
  Numeric::Matrix ensure_matrix(const Numeric::Matrix &mat, int rows, int cols)
  {
    if (!mat.empty())
      return mat;
    // Create zero matrix
    return Numeric::Matrix(rows, Numeric::Vector(cols, 0.0));
  }
}

ExperimentConfig ConfigLoader::load(const std::string &json_path)
{
  std::ifstream file(json_path);
  if (!file.is_open())
  {
    throw std::runtime_error("Could not open JSON file: " + json_path);
  }

  json j;
  file >> j;

  ExperimentConfig config;
  config.name = j.value("name", "unnamed_exp");
  config.duration = j.value("duration", 10.0);
  config.dt = j.value("dt", 1e-4);

  // --- Plant Parsing ---
  const auto &sys = j["plant"]["system_matrices"];

  config.plant.A = parse_matrix(sys["A"]);
  config.plant.B = parse_matrix(sys["B"]);
  config.plant.C = parse_matrix(sys["C"]);

  // Special handling for D (can be null)
  Numeric::Matrix D_raw = parse_matrix(sys["D"]);

  // If D is null/empty, initialize with zeros based on C and B dimensions.
  // D dimensions: (ny x nu), where ny = rows of C, nu = cols of B.
  int ny = config.plant.C.size();
  int nu = config.plant.B[0].size();
  config.plant.D = ensure_matrix(D_raw, ny, nu);

  // --- Design Parameters ---
  // Access design_params -> dspetc -> h
  if (j.contains("design_params") && j["design_params"].contains("dspetc"))
  {
    config.design_h = j["design_params"]["dspetc"].value("h", 0.001);
  }
  else
  {
    config.design_h = 0.001; // Default
  }

  return config;
}