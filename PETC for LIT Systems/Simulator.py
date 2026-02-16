# -*- coding: utf-8 -*-
"""
Simulator.py
Implementação funcional para simulação utilizando backend C++.
"""

import numpy as np
from Utils.CppInterface import CppSimulator

# Instância única e interna para gerenciar o caminho do executável
# Isso evita recriar o objeto a cada chamada de função
_backend = CppSimulator(exe_name="LinearStateSpaceSimulator")


def open_loop(x0, config_path, u_constant=0.0):
  """
  Simulação em Malha Aberta (estilo funcional).

  Args:
      x0 (list/array): Condição inicial.
      config_path (str): Caminho para o .json de configuração.
      u_constant (float): Entrada constante.
  """
  # 1. Chamada ao backend C++ (I/O via memória/Pipe)
  data = _backend.run(
      json_config_path=config_path,
      x0=x0,
      u_constant=u_constant
  )

  if data is None or data.size == 0:
    return None, None, None

  # 2. Reorganização dos dados para manter compatibilidade com o código original
  time_history = data[:, 0]
  y_hist = data[:, 1:].T  # Shape: (ny, n_steps)

  # 3. Histórico de controle constante
  n_steps = len(time_history)
  u_hist = np.full((1, n_steps), u_constant, dtype=np.float64)

  return y_hist, time_history, u_hist
