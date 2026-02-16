#include <iostream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <io.h>
#include <cstdint> // Required for fixed-width integer types (uint32_t)
#include "../include/ConfigLoader.h"
#include "../include/ETCforLinearSystemSimulator.h"
#include "../include/Numeric.h"

/**
 * @brief Entry point for the C++ simulation backend.
 * * Orchestrates configuration loading, simulation execution (Open or Closed Loop),
 * and binary data streaming to stdout for Python Inter-Process Communication (IPC).
 */
int main(int argc, char *argv[])
{
  // Usage: <executable> <json_path> <x0_csv> <u_constant> [--closed]
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <json_path> <x0_csv> <u_constant> [--closed]" << std::endl;
    return 1;
  }

  // Set stdout to binary mode to ensure raw double values are transmitted correctly
  // without line-ending translation (CRLF), which is critical for Windows.
  _setmode(_fileno(stdout), _O_BINARY);

  std::string json_path = argv[1];
  std::string x0_str = argv[2];
  double u_val = std::stod(argv[3]);
  bool closed_loop = (argc > 4 && std::string(argv[4]) == "--closed");

  try
  {
    // 1. Load Configuration
    // Parses plant matrices and control/ETM parameters from the JSON file.
    ExperimentConfig exp = ConfigLoader::load(json_path);

    // 2. Parse Initial State (x0)
    // Converts comma-separated string from CLI into a Numeric::Vector.
    Numeric::Vector x0;
    size_t pos = 0;
    std::string s = x0_str;
    while ((pos = s.find(",")) != std::string::npos)
    {
      x0.push_back(std::stod(s.substr(0, pos)));
      s.erase(0, pos + 1);
    }
    x0.push_back(std::stod(s));

    // 3. Execute Simulation
    // Dynamically chooses the simulation kernel based on the command line arguments.
    int n_steps = static_cast<int>(exp.duration / exp.dt);
    SimulationResult res;

    if (closed_loop)
    {
      res = Simulator::run_closed_loop_setm(exp.plant, exp.ctrl, x0, exp.dt, n_steps);
    }
    else
    {
      res = Simulator::run_open_loop(exp.plant, x0, u_val, exp.dt, n_steps);
    }

    // 4. Metadata Header (Binary)
    // Sends dimensions first (20 bytes) so the Python backend can correctly
    // reconstruct the flattened matrices.
    // Format: [ny, nx, nu, n_events, total_steps] (uint32_t)
    uint32_t ny = static_cast<uint32_t>(exp.plant.C.size());
    uint32_t nx = static_cast<uint32_t>(exp.plant.A.size());
    uint32_t nu = static_cast<uint32_t>(exp.plant.B.empty() ? 0 : exp.plant.B[0].size());
    uint32_t n_events = static_cast<uint32_t>(res.event_times.size());
    uint32_t total_steps = static_cast<uint32_t>(res.time.size());

    std::cout.write(reinterpret_cast<const char *>(&ny), sizeof(uint32_t));
    std::cout.write(reinterpret_cast<const char *>(&nx), sizeof(uint32_t));
    std::cout.write(reinterpret_cast<const char *>(&nu), sizeof(uint32_t));
    std::cout.write(reinterpret_cast<const char *>(&n_events), sizeof(uint32_t));
    std::cout.write(reinterpret_cast<const char *>(&total_steps), sizeof(uint32_t));

    // 5. Binary Data Stream
    // Transmits raw memory blocks for maximum efficiency.
    // Sequence: [Time_Vec] [Y_Matrix] [X_Matrix] [U_Matrix] [Event_Vec]

    // Time Vector
    std::cout.write(reinterpret_cast<const char *>(res.time.data()), res.time.size() * sizeof(double));

    // Output History (Y)
    for (const auto &v : res.y_hist)
      std::cout.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(double));

    // State History (X)
    for (const auto &v : res.x_hist)
      std::cout.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(double));

    // Control History (U)
    for (const auto &v : res.u_hist)
      std::cout.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(double));

    // Event Instants
    // Only streamed if n_events > 0, following the header information.
    if (n_events > 0)
      std::cout.write(reinterpret_cast<const char *>(res.event_times.data()), res.event_times.size() * sizeof(double));

    std::cout.flush();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}