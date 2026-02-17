/**
 * @file ETCforLinearSystemMain.cpp
 * @brief High-performance entry point for the C++ simulation backend.
 * Supports single-run simulations and massive parallel recurrence mapping
 * using binary file input to bypass Windows CLI limits.
 */

#include <iostream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <io.h>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <fstream> // Added for binary file stream support
#include <omp.h>
#include "../include/ConfigLoader.h"
#include "../include/ETCforLinearSystemSimulator.h"
#include "../include/Numeric.h"

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <json_path> <x0_input> <u_constant> [--closed | --recurrence | --parallel-rec]" << std::endl;
    return 1;
  }

  // Set stdout to binary mode to prevent CRLF translation
  _setmode(_fileno(stdout), _O_BINARY);

  std::string json_path = argv[1];
  std::string x0_input = argv[2]; // Path to binary file in parallel mode, or CSV in single mode
  double u_val = std::stod(argv[3]);

  bool closed_loop = false;
  bool recurrence = false;
  bool parallel_rec = false;

  if (argc > 4)
  {
    std::string flag = argv[4];
    closed_loop = (flag == "--closed");
    recurrence = (flag == "--recurrence");
    parallel_rec = (flag == "--parallel-rec");
  }

  try
  {
    ExperimentConfig exp = ConfigLoader::load(json_path);

    if (parallel_rec)
    {
      /** * PARALLEL BATCH MODE
       * Reads states from a binary file: [uint32 num_states][uint32 dim][double data...]
       */
      std::ifstream bin_file(x0_input, std::ios::binary);
      if (!bin_file.is_open())
      {
        throw std::runtime_error("Could not open binary states file: " + x0_input);
      }

      uint32_t num_states = 0;
      uint32_t dim = 0;

      bin_file.read(reinterpret_cast<char *>(&num_states), sizeof(uint32_t));
      bin_file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));

      std::vector<Numeric::Vector> states(num_states, Numeric::Vector(dim));
      for (uint32_t i = 0; i < num_states; ++i)
      {
        bin_file.read(reinterpret_cast<char *>(states[i].data()), dim * sizeof(double));
      }
      bin_file.close();

      std::vector<std::vector<double>> all_event_times(num_states);

// Native OpenMP thread management
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < static_cast<int>(num_states); ++i)
      {
        all_event_times[i] = Simulator::run_recurrence_map_setm(
            exp.plant, exp.ctrl, states[i], exp.duration);
      }

      // Write results to stdout using the multi-sequence binary protocol
      std::cout.write(reinterpret_cast<const char *>(&num_states), sizeof(uint32_t));

      for (uint32_t i = 0; i < num_states; ++i)
      {
        uint32_t n_ev = static_cast<uint32_t>(all_event_times[i].size());
        std::cout.write(reinterpret_cast<const char *>(&n_ev), sizeof(uint32_t));
        if (n_ev > 0)
        {
          std::cout.write(reinterpret_cast<const char *>(all_event_times[i].data()), n_ev * sizeof(double));
        }
      }
    }
    else
    {
      /** * SINGLE SIMULATION MODE
       * Processes x0 from CSV argument
       */
      Numeric::Vector x0;
      std::stringstream ss(x0_input);
      std::string item;
      while (std::getline(ss, item, ','))
      {
        if (!item.empty())
          x0.push_back(std::stod(item));
      }

      SimulationResult res;
      if (recurrence)
      {
        res.event_times = Simulator::run_recurrence_map_setm(exp.plant, exp.ctrl, x0, exp.duration);
      }
      else
      {
        int n_steps = static_cast<int>(std::round(exp.duration / exp.dt)) + 1;
        if (closed_loop)
          res = Simulator::run_closed_loop_setm(exp.plant, exp.ctrl, x0, exp.dt, n_steps);
        else
          res = Simulator::run_open_loop(exp.plant, x0, u_val, exp.dt, n_steps);
      }

      uint32_t ny = (recurrence) ? 0 : static_cast<uint32_t>(exp.plant.C.size());
      uint32_t nx = (recurrence) ? 0 : static_cast<uint32_t>(exp.plant.A.size());
      uint32_t nu = (recurrence) ? 0 : static_cast<uint32_t>(exp.plant.B.empty() ? 0 : exp.plant.B[0].size());
      uint32_t n_events = static_cast<uint32_t>(res.event_times.size());
      uint32_t total_steps = static_cast<uint32_t>(res.time.size());

      uint32_t header[5] = {ny, nx, nu, n_events, total_steps};
      std::cout.write(reinterpret_cast<const char *>(header), 5 * sizeof(uint32_t));

      if (total_steps > 0)
      {
        std::cout.write(reinterpret_cast<const char *>(res.time.data()), total_steps * sizeof(double));
        for (const auto &v : res.y_hist)
          std::cout.write(reinterpret_cast<const char *>(v.data()), ny * sizeof(double));
        for (const auto &v : res.x_hist)
          std::cout.write(reinterpret_cast<const char *>(v.data()), nx * sizeof(double));
        for (const auto &v : res.u_hist)
          std::cout.write(reinterpret_cast<const char *>(v.data()), nu * sizeof(double));
      }

      if (n_events > 0)
        std::cout.write(reinterpret_cast<const char *>(res.event_times.data()), n_events * sizeof(double));
    }

    std::cout.flush();
  }
  catch (const std::exception &e)
  {
    std::cerr << "CRITICAL_ERROR: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}