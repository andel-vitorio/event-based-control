import subprocess
import os
import numpy as np
import json
import sys


class CppSimulator:
  def __init__(self, exe_name="ETCforLinearSystemSimulator"):
    """
    Initializes the C++ simulator interface with metadata-aware binary streaming.

    Args:
        exe_name (str): Name of the compiled C++ executable.
    """
    ext = ".exe" if sys.platform == "win32" else ""
    filename = f"{exe_name}{ext}"

    self.exe_path = None
    cwd = os.getcwd()

    # Positional heuristics to locate the binary regardless of installation mode
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

  def run(self, json_config_path, x0, u_constant=0.0, closed_loop=False):
    """
    Executes the C++ simulation and decodes the multi-vector binary stream.
    Handles dynamic dimensions for Open-Loop and Closed-Loop modes.

    Args:
        json_config_path (str): Path to the experiment configuration.
        x0 (list/np.ndarray): Initial state vector.
        u_constant (float): Constant input for open-loop simulation.
        closed_loop (bool): Flag to trigger the SETM kernel in the backend.

    Returns:
        dict: Dictionary containing 'y', 'x', 'u', 't', and 'event_times'.
    """
    x0_str = ",".join(map(str, np.array(x0).flatten()))
    abs_json_path = os.path.abspath(json_config_path)

    # Execution command with optional --closed flag
    cmd = [self.exe_path, abs_json_path, x0_str, str(u_constant)]
    if closed_loop:
      cmd.append("--closed")

    try:
      process = subprocess.Popen(
          cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE
      )
      stdout_data, stderr_data = process.communicate()

      if process.returncode != 0:
        raise RuntimeError(f"C++ Error: {stderr_data.decode('utf-8')}")

      # --- 1. Decode Metadata Header (5 uint32 = 20 bytes) ---
      # Header provides [ny, nx, nu, n_events, total_steps]
      header = np.frombuffer(stdout_data[:20], dtype=np.uint32)
      ny, nx, nu, n_events, total_steps = header

      # --- 2. Dynamic Stream Slicing ---
      offset = 20

      def fetch_data(count, shape=None):
        """Helper to extract slices from the binary buffer based on header counts."""
        nonlocal offset
        bytes_to_read = count * 8  # Each double is 8 bytes
        chunk = stdout_data[offset: offset + bytes_to_read]
        data = np.frombuffer(chunk, dtype=np.float64)
        offset += bytes_to_read
        return data.reshape(shape) if shape else data

      # Extract vectors sequentially according to the binary protocol
      time_history = fetch_data(total_steps)
      y_hist = fetch_data(total_steps * ny, (total_steps, ny))
      x_hist = fetch_data(total_steps * nx, (total_steps, nx))
      u_hist = fetch_data(total_steps * nu, (total_steps, nu))

      # Remaining data in buffer represents event instants
      event_times = fetch_data(n_events)

      return {
          't': time_history,
          # Transpose to (ny, n_steps) for legacy compatibility
          'y': y_hist.T,
          'x': x_hist.T,
          'u': u_hist.T,
          'event_times': event_times
      }

    except Exception as e:
      raise RuntimeError(f"Simulation failed: {e}")
