#pragma once

#include "SystemModel.hpp"
#include "../lib/json.hpp"
#include <string>
#include <filesystem>

/**
 * @brief Parser class responsible for converting JSON system descriptions
 * into internal SystemModel representations.
 */
class SystemParser
{
public:
  /**
   * @brief Parses a system configuration from a JSON file.
   * @param filePath Path to the target .json file.
   * @return A populated SystemModel structure.
   */
  static SystemModel parseFromFile(const std::filesystem::path &filePath);

private:
  /**
   * @brief Parses a JSON matrix into a SystemMatrix structure.
   * Handles both numeric values and string identifiers (LPV parameters).
   * * @param matrixJson The nlohmann::json object representing the matrix.
   * @return A SystemMatrix containing either doubles or strings.
   */
  static SystemMatrix parseMatrix(const nlohmann::json &matrixJson);

  /**
   * @brief Parses a single variable (state, parameter, etc.) from the JSON.
   * * @param varJson The nlohmann::json object for the variable.
   * @return A populated Variable structure.
   */
  static Variable parseVariable(const nlohmann::json &varJson);
};