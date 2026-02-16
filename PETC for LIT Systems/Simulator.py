# -*- coding: utf-8 -*-
"""
Simulator.py
Functional implementation for simulation using C++ backend.
"""

import numpy as np
import json
import os
import tempfile
from Utils.CppInterface import CppSimulator

# Single internal instance to manage the executable path
_backend = CppSimulator(exe_name="ETCforLinearSystemMain")


class NumpyEncoder(json.JSONEncoder):
  """
  Custom JSON Encoder to handle NumPy types during serialization.
  """

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
      return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
      return int(obj)
    return super(NumpyEncoder, self).default(obj)


def open_loop(x0, config_path, u_constant=0.0):
  """
  Performs an Open-Loop simulation.
  """
  results = _backend.run(
      json_config_path=config_path,
      x0=x0,
      u_constant=u_constant,
      closed_loop=False
  )
  if not results:
    return None, None, None
  return results['y'], results['t'], results['u']


def closed_loop_setm(x0, config_path, results_dict):
  """
  Performs a Closed-Loop SETM simulation.
  Maps optimization keys (Ξ, Ψ) to C++ expected keys (Xi, Psi).
  """
  if results_dict is None:
    raise ValueError("Optimization results are required for closed-loop.")

  with open(config_path, 'r') as f:
    full_data = json.load(f)

  # Normalização das chaves para o padrão esperado pelo ConfigLoader.cpp
  # O C++ não gosta de caracteres gregos no mapeamento de chaves do JSON
  full_data['results'] = {
      "controller": {
          "K": results_dict['controller']['K']
      },
      "etm": {
          "Xi": results_dict['etm']['Ξ'],
          "Psi": results_dict['etm']['Ψ']
      }
  }

  with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
    json.dump(full_data, tmp, cls=NumpyEncoder)
    temp_config_path = tmp.name

  try:
    sim_data = _backend.run(
        json_config_path=temp_config_path,
        x0=x0,
        u_constant=0.0,
        closed_loop=True
    )
    if not sim_data:
      return None, None, None, None

    return sim_data['y'], sim_data['t'], sim_data['u'], sim_data['event_times']
  finally:
    if os.path.exists(temp_config_path):
      os.remove(temp_config_path)
