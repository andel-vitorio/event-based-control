#include <iostream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <io.h>
#include "../include/ConfigLoader.h"
#include "../include/Simulator.h"
#include "../include/Numeric.h"

int main(int argc, char *argv[])
{
  // Usage: <executable> <json_path> <x0_csv> <u_constant>
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <json_path> <x0_csv> <u_constant>" << std::endl;
    return 1;
  }

  // Set stdout to binary mode to ensure raw double values are transmitted correctly
  // without line-ending translation (CRLF), which is critical for Windows.
  _setmode(_fileno(stdout), _O_BINARY);

  std::string json_path = argv[1];
  std::string x0_str = argv[2];
  double u_val = std::stod(argv[3]);

  try
  {
    // 1. Load Configuration
    ExperimentConfig exp = ConfigLoader::load(json_path);

    // 2. Parse Initial State (x0)
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
    int n_steps = static_cast<int>(exp.duration / exp.dt);
    SimulationResult res = Simulator::run_open_loop(exp.plant, x0, u_val, exp.dt, n_steps);

    // 4. Binary Output
    // Stream results directly to stdout.
    // Format per step: [time (double), y_1 (double), ..., y_ny (double)]
    for (size_t i = 0; i < res.time.size(); ++i)
    {
      // Write time step
      std::cout.write(reinterpret_cast<const char *>(&res.time[i]), sizeof(double));

      // Write output vector
      if (i < res.y_hist.size())
      {
        const auto &y_vec = res.y_hist[i];
        if (!y_vec.empty())
        {
          std::cout.write(reinterpret_cast<const char *>(y_vec.data()), y_vec.size() * sizeof(double));
        }
      }
    }

    std::cout.flush();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}