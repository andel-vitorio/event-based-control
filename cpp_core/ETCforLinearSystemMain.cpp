#include <iostream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <io.h>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <cmath>
#include "../include/ConfigLoader.h"
#include "../include/ETCforLinearSystemSimulator.h"
#include "../include/Numeric.h"

/**
 * @brief Entry point for the C++ simulation backend.
 * Refactored for strict binary protocol alignment and timing precision.
 */
int main(int argc, char *argv[])
{
  // Usage: <executable> <json_path> <x0_csv> <u_constant> [--closed | --recurrence]
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <json_path> <x0_csv> <u_constant> [--closed | --recurrence]" << std::endl;
    return 1;
  }

  // Ensure stdout is in binary mode to prevent newline translation (CRLF vs LF)
  _setmode(_fileno(stdout), _O_BINARY);

  std::string json_path = argv[1];
  std::string x0_str = argv[2];
  double u_val = std::stod(argv[3]);

  bool closed_loop = false;
  bool recurrence = false;
  if (argc > 4)
  {
    std::string flag = argv[4];
    closed_loop = (flag == "--closed");
    recurrence = (flag == "--recurrence");
  }

  try
  {
    // 1. Load Configuration
    ExperimentConfig exp = ConfigLoader::load(json_path);

    // 2. Parse Initial State (x0) using robust stringstream
    Numeric::Vector x0;
    std::stringstream ss(x0_str);
    std::string item;
    while (std::getline(ss, item, ','))
    {
      if (!item.empty())
        x0.push_back(std::stod(item));
    }

    SimulationResult res;

    // 3. Simulation Execution
    if (recurrence)
    {
      // Only event instants are calculated here
      res.event_times = Simulator::run_recurrence_map_setm(exp.plant, exp.ctrl, x0, exp.duration);
    }
    else
    {
      // Use round + 1 to include the last sample at t = duration
      int n_steps = static_cast<int>(std::round(exp.duration / exp.dt)) + 1;

      if (closed_loop)
      {
        res = Simulator::run_closed_loop_setm(exp.plant, exp.ctrl, x0, exp.dt, n_steps);
      }
      else
      {
        res = Simulator::run_open_loop(exp.plant, x0, u_val, exp.dt, n_steps);
      }
    }

    // 4. Metadata Preparation
    // Extract dimensions from the actual plant model
    uint32_t ny_real = static_cast<uint32_t>(exp.plant.C.size());
    uint32_t nx_real = static_cast<uint32_t>(exp.plant.A.size());
    uint32_t nu_real = static_cast<uint32_t>(exp.plant.B.empty() ? 0 : exp.plant.B[0].size());

    uint32_t n_events = static_cast<uint32_t>(res.event_times.size());
    uint32_t total_steps = static_cast<uint32_t>(res.time.size());

    // Header for the Python interface: [ny, nx, nu, n_events, total_steps]
    // In recurrence mode, we send 0 for ny, nx, nu to signal no history matrices
    uint32_t header[5];
    header[0] = (recurrence) ? 0 : ny_real;
    header[1] = (recurrence) ? 0 : nx_real;
    header[2] = (recurrence) ? 0 : nu_real;
    header[3] = n_events;
    header[4] = total_steps;

    // Atomic write of the header
    std::cout.write(reinterpret_cast<const char *>(header), 5 * sizeof(uint32_t));

    // 5. Data Stream writing
    if (total_steps > 0)
    {
      // Time vector
      std::cout.write(reinterpret_cast<const char *>(res.time.data()), total_steps * sizeof(double));

      // Output History (Y)
      for (const auto &v : res.y_hist)
        std::cout.write(reinterpret_cast<const char *>(v.data()), header[0] * sizeof(double));

      // State History (X)
      for (const auto &v : res.x_hist)
        std::cout.write(reinterpret_cast<const char *>(v.data()), header[1] * sizeof(double));

      // Control History (U)
      for (const auto &v : res.u_hist)
        std::cout.write(reinterpret_cast<const char *>(v.data()), header[2] * sizeof(double));
    }

    // Always write event times at the end of the stream
    if (n_events > 0)
    {
      std::cout.write(reinterpret_cast<const char *>(res.event_times.data()), n_events * sizeof(double));
    }

    std::cout.flush();
  }
  catch (const std::exception &e)
  {
    // Use stderr for errors so Python can catch the RuntimeError with the message
    std::cerr << "CRITICAL_ERROR: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}