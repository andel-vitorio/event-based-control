#include <iostream>
#include <variant>
#include <filesystem>
#include "include/SystemParser.hpp"

/**
 * @brief Helper to print MatrixValue elements.
 */
struct MatrixPrinter
{
  void operator()(double d) const { std::cout << d << " "; }
  void operator()(const std::string &s) const { std::cout << "[" << s << "] "; }
};

/**
 * @brief Prints the system model details for verification.
 */
void printSystemInfo(const SystemModel &model)
{
  std::cout << "System Name: " << model.name << std::endl;

  if (model.A.has_value())
  {
    std::cout << "Matrix A:" << std::endl;
    for (const auto &row : model.A.value())
    {
      for (const auto &element : row)
      {
        std::visit(MatrixPrinter{}, element);
      }
      std::cout << std::endl;
    }
  }
}

int main()
{
  try
  {
    // Ensure this path matches your project structure
    std::filesystem::path jsonPath = "../systems-datas/sys-02.json";

    SystemModel model = SystemParser::parseFromFile(jsonPath);

    std::cout << "System successfully parsed." << std::endl;
    printSystemInfo(model);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during parsing: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}