import subprocess
import os
import numpy as np
import json
import sys


class CppSimulator:
  def __init__(self, exe_name="LinearStateSpaceSimulator"):
    """
    Initializes the C++ simulator interface.

    Implements a search strategy to locate the compiled executable, prioritizing
    local build directories over system-wide installations.

    Args:
        exe_name (str): Name of the executable file (without extension).
    """
    ext = ".exe" if sys.platform == "win32" else ""
    filename = f"{exe_name}{ext}"

    self.exe_path = None

    cwd = os.getcwd()

    possible_paths = [
        os.path.join(cwd, "bin", filename),
        os.path.join(os.path.dirname(cwd), "bin", filename),
        # Level 2: bin/ in grandparent folder (e.g., running from Utils/ or subfolder)
        os.path.join(os.path.dirname(
            os.path.dirname(cwd)), "bin", filename),
        # Fallback: Relative to lib installation (editable mode)
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "bin", filename)
    ]

    # Validation loop (Stops on first match)
    for path in possible_paths:
      if os.path.exists(path):
        self.exe_path = os.path.abspath(path)
        break

    if self.exe_path is None:
      raise FileNotFoundError(
          f"Executable '{filename}' not found.\n"
          f"Search paths included 'bin/' directories near: {cwd}\n"
          f"Ensure the project is built and you are running from the project root."
      )

  def _get_ny_from_json(self, json_path):
    """
    Helper to extract the number of outputs (ny) from the JSON configuration.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        int or None: Number of rows in matrix C, or None if parsing fails.
    """
    try:
      with open(json_path, 'r') as f:
        config = json.load(f)
      return len(config['plant']['system_matrices']['C'])
    except:
      return None

  def run(self, json_config_path, x0, u_constant=0.0):
    """
    Executes the C++ simulation.

    Args:
        json_config_path (str): Path to the experiment configuration JSON.
        x0 (list or np.ndarray): Initial state vector.
        u_constant (float): Constant control input value.

    Returns:
        np.ndarray: Simulation results matrix [time, y1, ..., yny].

    Raises:
        RuntimeError: If the C++ process fails or returns an error code.
    """
    x0_str = ",".join(map(str, np.array(x0).flatten()))
    abs_json_path = os.path.abspath(json_config_path)

    cmd = [self.exe_path, abs_json_path, x0_str, str(u_constant)]

    try:
      process = subprocess.Popen(
          cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE
      )
      stdout_data, stderr_data = process.communicate()

      if process.returncode != 0:
        raise RuntimeError(
            f"C++ Error: {stderr_data.decode('utf-8', errors='replace')}")

      raw_data = np.frombuffer(stdout_data, dtype=np.float64)

      if raw_data.size == 0:
        return np.empty((0, 0))

      ny = self._get_ny_from_json(abs_json_path)
      if ny is not None:
        return raw_data.reshape(-1, 1 + ny)
      return raw_data

    except Exception as e:
      raise RuntimeError(f"Simulation failed: {e}")
