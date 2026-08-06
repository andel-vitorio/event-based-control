#include "MTD.hpp"

namespace ControlSystems
{
  void MTDManager::setTransitionProbabilities(const double *pi_data, int m, int r)
  {
    n_modes = m;
    n_regions = r;
    pi_tensor.assign(pi_data, pi_data + (m * m * r));
  }

  void MTDManager::setEnergyWeightMatrix(const Algebra::Matrix &W_matrix)
  {
    // A dimensão é capturada diretamente da matriz fornecida
    W = W_matrix;
  }

  void MTDManager::setEnergyThresholds(const double *c_data, int n_regions)
  {
    c.assign(c_data, c_data + n_regions - 1);
  }

  void MTDManager::update(const Algebra::Vector &x)
  {
    double energy = Algebra::Vector::dot(x.T(), W * x);

    // Classificação limpa e adaptativa para n_regions - 1 fronteiras
    current_region = (int)c.size(); // Assume por padrão a última região (index = n_regions - 1)
    for (size_t i = 0; i < c.size(); ++i)
    {
      if (energy <= c[i])
      {
        current_region = (int)i;
        break;
      }
    }

    std::vector<double> probs(n_modes);
    for (int next_mode = 0; next_mode < n_modes; ++next_mode)
    {
      probs[next_mode] = pi_tensor[current_region * (n_modes * n_modes) + current_mode * n_modes + next_mode];
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    current_mode = dist(gen);
  }
}