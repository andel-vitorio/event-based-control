#include "../../include/PeriodicETC/LITEngine.hpp"
#include <stdexcept>

namespace PeriodicETC::LIT
{
  Engine::Engine() = default;
  Engine::~Engine() = default;

  void Engine::configure(const std::filesystem::path &json_path)
  {
    auto model = std::make_unique<SystemModel>(SystemParser::parseFromFile(json_path));

    if (!model->A.has_value() || !model->B.has_value())
    {
      throw std::runtime_error("LIT Engine Error: System model requires matrices A and B.");
    }

    model_ = std::move(model);
  }
}