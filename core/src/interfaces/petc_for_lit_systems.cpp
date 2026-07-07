#include "../../include/PeriodicETC/LITEngine.hpp"

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#include <stdexcept>
#include <iostream>

using EnginePtr = PeriodicETC::LIT::Engine *;

DLLEXPORT EnginePtr create()
{
  return new PeriodicETC::LIT::Engine();
}

DLLEXPORT void load_system(EnginePtr e, const char *path)
{
  try
  {
    if (e)
    {
      e->loadSystem(std::string(path));
    }
  }
  catch (const std::exception &ex)
  {
    // Isso impede que o Python trave e imprime o erro no console do Jupyter
    std::cerr << "[C++ Error inside DLL]: " << ex.what() << std::endl;
  }
  catch (...)
  {
    std::cerr << "[C++ Error inside DLL]: Unknown error" << std::endl;
  }
}

DLLEXPORT void load_sim(EnginePtr e, double dt, double tf)
{
  if (e)
  {
    PeriodicETC::SimulationParams p;
    p.time_step = dt;
    p.final_time = tf;
    e->loadSimulation(p);
  }
}

DLLEXPORT void destroy(EnginePtr e)
{
  if (e)
    delete e;
}

/**
 * @brief Executes the open-loop simulation for the specified duration.
 * @param e Pointer to the engine instance.
 * @param duration Simulation duration in seconds.
 */
DLLEXPORT void open_loop(EnginePtr e, double duration)
{
  if (e)
  {
    e->openLoop(duration);
  }
}

DLLEXPORT double *get_history_data(EnginePtr e)
{
  return (double *)e->getHistoryData();
}

DLLEXPORT int get_history_size(EnginePtr e)
{
  return (int)e->getHistorySize();
}

DLLEXPORT int get_state_dim(EnginePtr e)
{
  return (int)e->getState().size();
}