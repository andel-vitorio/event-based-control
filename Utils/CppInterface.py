# -*- coding: utf-8 -*-
"""
CppInterface.py
Metadata-aware binary streaming interface for C++ backend.
Supports single-run simulations and high-performance parallel symbolic sequences.
"""

import subprocess
import os
import numpy as np
import sys


class CppSimulator:
  def __init__(self, exe_name="ETCforLinearSystemSimulator"):
    """
    Initializes the C++ simulator interface.

    Args:
        exe_name (str): Name of the compiled C++ executable.
    """
    ext = ".exe" if sys.platform == "win32" else ""
    filename = f"{exe_name}{ext}"

    self.exe_path = None
    cwd = os.getcwd()

    possible_paths = [
        os.path.join(cwd, "bin", filename),
        os.path.join(os.path.dirname(cwd), "bin", filename),
        os.path.join(os.path.dirname(
            os.path.dirname(cwd)), "bin", filename),
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "bin", filename)
    ]

    for path in possible_paths:
      if os.path.exists(path):
        self.exe_path = os.path.abspath(path)
        break

    if self.exe_path is None:
      raise FileNotFoundError(f"Executable '{filename}' not found.")

  def run(self, json_config_path, x0, u_constant=0.0, closed_loop=False, recurrence=False):
    """
    Executes a single simulation run.

    Returns:
        dict: Dictionary containing 'y', 'x', 'u', 't', and 'event_times'.
    """
    x0_str = ",".join(map(str, np.array(x0).flatten()))
    cmd = [self.exe_path, os.path.abspath(
        json_config_path), x0_str, str(u_constant)]

    if recurrence:
      cmd.append("--recurrence")
    elif closed_loop:
      cmd.append("--closed")

    stdout_data, stderr_data = self._execute_process(cmd)

    # --- Decode Metadata Header (5 uint32 = 20 bytes) ---
    header = np.frombuffer(stdout_data[:20], dtype=np.uint32)
    ny, nx, nu, n_events, total_steps = header

    offset = 20

    def fetch_data(count, shape=None):
      nonlocal offset
      bytes_to_read = count * 8
      data = np.frombuffer(
          stdout_data[offset: offset + bytes_to_read], dtype=np.float64)
      offset += bytes_to_read
      return data.reshape(shape) if shape else data

    time_history = fetch_data(total_steps)
    y_hist = fetch_data(total_steps * ny, (total_steps, ny))
    x_hist = fetch_data(total_steps * nx, (total_steps, nx))
    u_hist = fetch_data(total_steps * nu, (total_steps, nu))
    event_times = fetch_data(n_events)

    return {
        't': time_history,
        'y': y_hist.T,
        'x': x_hist.T,
        'u': u_hist.T,
        'event_times': event_times
    }

  def run_parallel_recurrence(self, json_config_path, initial_states):
    """
      Uses a temporary binary file to pass states to C++, bypassing Windows CLI limits.
      """
    import tempfile

    # 1. Preparar dados binários: [NumStates (u32), Dim (u32), Data (doubles...)]
    num_states = np.uint32(len(initial_states))
    dim = np.uint32(initial_states.shape[1])

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
      f.write(num_states.tobytes())
      f.write(dim.tobytes())
      f.write(initial_states.astype(np.float64).tobytes())
      bin_path = f.name

    try:
      # Passamos o CAMINHO do binário, que é uma string curta
      cmd = [self.exe_path, os.path.abspath(
          json_config_path), bin_path, "0.0", "--parallel-rec"]
      stdout_data, stderr_data = self._execute_process(cmd)

      # --- Parsing do retorno (mesmo de antes) ---
      num_sequences = np.frombuffer(stdout_data[:4], dtype=np.uint32)[0]
      offset = 4
      all_event_times = []
      for _ in range(num_sequences):
        n_ev = np.frombuffer(
            stdout_data[offset: offset + 4], dtype=np.uint32)[0]
        offset += 4
        ev_data = np.frombuffer(
            stdout_data[offset: offset + n_ev*8], dtype=np.float64)
        all_event_times.append(ev_data.copy())
        offset += n_ev * 8
      return all_event_times
    finally:
      if os.path.exists(bin_path):
        os.remove(bin_path)

  def _execute_process(self, cmd):
    """Internal helper to handle subprocess execution and errors."""
    try:
      process = subprocess.Popen(
          cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE
      )
      stdout_data, stderr_data = process.communicate()

      if process.returncode != 0:
        raise RuntimeError(
            f"C++ Execution Error: {stderr_data.decode('utf-8')}")

      return stdout_data, stderr_data
    except Exception as e:
      raise RuntimeError(f"Backend communication failed: {e}")
