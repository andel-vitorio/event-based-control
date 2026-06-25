#pragma once
#include <memory>
#include <filesystem>
#include "../../libs/control-core/include/SystemParser.hpp"

// Namespace hierárquico: Projeto -> Domínio
namespace PeriodicETC::LIT
{
  class Engine // Apenas Engine, sem repetir o nome do namespace
  {
  private:
    std::unique_ptr<SystemModel> model_;

  public:
    Engine();
    ~Engine();

    void configure(const std::filesystem::path &json_path);
    const SystemModel &getModel() const { return *model_; }
  };
}