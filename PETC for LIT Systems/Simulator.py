# -*- coding: utf-8 -*-
"""
Simulator.py
Functional implementation for simulation using C++ backend.
Provides strict I/O handling to ensure determinism between calls.
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
  Ensures all matrix types are converted to standard Python lists.
  """

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.longdouble)):
      return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16)):
      return int(obj)
    return super(NumpyEncoder, self).default(obj)


def _prepare_temp_config(config_path, results_dict):
  """
  Internal helper to merge base config with optimization results
  and write to a guaranteed closed temporary file.
  """
  with open(config_path, 'r') as f:
    full_data = json.load(f)

  # Map Greek symbols (Ξ, Ψ) to C++ expected keys (Xi, Psi)
  # This prevents key errors in the backend ConfigLoader
  full_data['results'] = {
      "controller": {
          "K": np.array(results_dict['controller']['K']).tolist()
      },
      "etm": {
          "Xi": np.array(results_dict['etm']['Ξ']).tolist(),
          "Psi": np.array(results_dict['etm']['Ψ']).tolist()
      }
  }

  # Use delete=False to manage lifecycle manually and avoid sharing violations
  tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
  try:
    json.dump(full_data, tmp, cls=NumpyEncoder)
    tmp.flush()
    os.fsync(tmp.fileno())  # Force write to disk
    return tmp.name
  finally:
    tmp.close()


def open_loop(x0, config_path, u_constant=0.0):
  """
  Performs an Open-Loop simulation using the C++ backend.
  """
  # Ensure x0 is a clean copy to prevent mutation
  x0_clean = np.array(x0, copy=True).flatten()

  results = _backend.run(
      json_config_path=config_path,
      x0=x0_clean,
      u_constant=u_constant,
      closed_loop=False
  )
  if not results:
    return None, None, None
  return results['y'], results['t'], results['u']


def closed_loop_setm(x0, config_path, results_dict):
  """
  Performs a Closed-Loop SETM simulation (RK5).
  """
  if results_dict is None:
    raise ValueError("Optimization results are required for closed-loop.")

  temp_path = _prepare_temp_config(config_path, results_dict)
  x0_clean = np.array(x0, copy=True).flatten()

  try:
    sim_data = _backend.run(
        json_config_path=temp_path,
        x0=x0_clean,
        u_constant=0.0,
        closed_loop=True
    )
    if not sim_data:
      return None, None, None, None

    return sim_data['y'], sim_data['t'], sim_data['u'], sim_data['event_times']
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def recurrence_model_setm(x0, config_path, results_dict):
  """
  Simulates the system using the Discrete-Time Recurrence Map.
  Optimized for high-speed event detection.
  """
  if results_dict is None:
    raise ValueError(
        "Optimization results are required for recurrence map.")

  temp_path = _prepare_temp_config(config_path, results_dict)
  x0_clean = np.array(x0, copy=True).flatten()

  try:
    sim_data = _backend.run(
        json_config_path=temp_path,
        x0=x0_clean,
        u_constant=0.0,
        closed_loop=False,
        recurrence=True
    )

    if not sim_data:
      return None

    # Return only the event timestamps
    return sim_data['event_times']
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)
