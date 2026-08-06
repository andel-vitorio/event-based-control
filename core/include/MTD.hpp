#pragma once

#include <random>
#include <vector>
#include "Algebra/Algebra.hpp"

namespace ControlSystems
{
  class MTDManager
  {
  private:
    std::mt19937 gen;
    std::vector<double> pi_tensor;
    int n_modes;
    int n_regions;
    Algebra::Matrix W; // Agora a classe gerencia a matriz diretamente
    std::vector<double> c;

    int current_mode;
    int current_region;

  public:
    MTDManager() : n_modes(0), n_regions(0), W(0, 0), current_mode(0), current_region(0)
    {
      std::random_device rd;
      gen.seed(rd());
    }

    void setSeed(unsigned int seed) { gen.seed(seed); }

    // Métodos modulares de configuração
    void setTransitionProbabilities(const double *pi_data, int m, int r);
    void setEnergyWeightMatrix(const Algebra::Matrix &W_matrix);
    void setEnergyThresholds(const double *c_data, int n_regions);

    void update(const Algebra::Vector &);

    int getCurrentMode() const { return current_mode; }
    int getCurrentRegion() const { return current_region; }
  };
}