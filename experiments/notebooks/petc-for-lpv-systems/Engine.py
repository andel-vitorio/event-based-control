import ctypes
import os
import sys
import json
import numpy as np
from ctypes import wintypes


class Engine:
  def __init__(self, dll_path="../../../core/dll/petc_for_lpv_systems.dll"):
    self.dll_path = os.path.abspath(dll_path)
    self.model_data = None
    self.lib = None
    self.engine_ptr = None

    if not os.path.exists(self.dll_path):
      raise FileNotFoundError(f"DLL não encontrada: {self.dll_path}")

    if sys.platform == 'win32':
      os.add_dll_directory(os.path.dirname(self.dll_path))

    # Carregamento da DLL ocorre automaticamente na criação do objeto
    self.lib = ctypes.CDLL(self.dll_path)
    self._initialize_dll_bindings()
    self.engine_ptr = self.lib.create()

  def _initialize_dll_bindings(self):
    """Define os tipos da API da DLL."""
    self.lib.create.restype = ctypes.c_void_p
    self.lib.destroy.argtypes = [ctypes.c_void_p]

    self.lib.load_system.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    self.lib.load_sim.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    self.lib.open_loop.argtypes = [ctypes.c_void_p, ctypes.c_double]

    self.lib.set_initial_state.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]

    self.lib.get_history_size.argtypes = [ctypes.c_void_p]
    self.lib.get_history_size.restype = ctypes.c_size_t
    self.lib.get_state_dim.argtypes = [ctypes.c_void_p]
    self.lib.get_state_dim.restype = ctypes.c_int
    self.lib.get_history_data.argtypes = [ctypes.c_void_p]
    self.lib.get_history_data.restype = ctypes.POINTER(ctypes.c_double)

    self.lib.get_time_size.argtypes = [ctypes.c_void_p]
    self.lib.get_time_size.restype = ctypes.c_int
    self.lib.get_time_data.argtypes = [ctypes.c_void_p]
    self.lib.get_time_data.restype = ctypes.POINTER(ctypes.c_double)

    self.lib.run_closed_loop.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,  # x0_matrix, n_states
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,  # K_matrix, nu
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(
            ctypes.c_double),  # Xi, Psi
        ctypes.c_double, ctypes.c_int,  # sampling_period, num_scenarios
        ctypes.c_double, ctypes.c_double  # sample_time, duration
    ]
    self.lib.get_num_scenarios_results.argtypes = [ctypes.c_void_p]
    self.lib.get_num_scenarios_results.restype = ctypes.c_int

    self.lib.get_scenario_data_sizes.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(
            ctypes.c_int)  # Adicionado u_sz
    ]
    self.lib.copy_scenario_results.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(
            ctypes.c_double)  # Adicionado u_out
    ]

    self.lib.get_matrix.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(
            ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
    ]

  def close(self):
    """Libera os recursos manualmente."""
    if self.engine_ptr and self.lib:
      self.lib.destroy(self.engine_ptr)
      self.engine_ptr = None

    if self.lib:
      dll_handle = self.lib._handle
      del self.lib
      self.lib = None

      if sys.platform == 'win32':
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
        kernel32.FreeLibrary(dll_handle)
        print("DLL descarregada com sucesso.")

  def load_lpv_model(self, json_path: str):
    """Carrega o modelo e sincroniza os metadados."""
    self.lib.load_system(self.engine_ptr, json_path.encode('utf-8'))
    try:
      with open(json_path, 'r', encoding='utf-8') as f:
        self.model_data = json.load(f)
    except Exception as e:
      print(f"Erro ao carregar metadados: {e}")
      self.model_data = None

  @property
  def lpv_model(self):
    if self.model_data is None:
      raise RuntimeError("Nenhum modelo carregado.")
    return self.model_data

  def set_initial_state(self, x0: np.ndarray):
    x0_contiguous = np.ascontiguousarray(x0, dtype=np.float64)
    dim = len(x0_contiguous)
    ptr = x0_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    self.lib.set_initial_state(self.engine_ptr, ptr, dim)

  def run_open_loop_simulation(self, sample_time: float, duration: float):
    """Executa a simulação e retorna os resultados unificados."""
    self.lib.load_sim(self.engine_ptr, sample_time, duration)
    self.lib.open_loop(self.engine_ptr, ctypes.c_double(duration))

    # Extração unificada de tempo e estados
    time_size = self.lib.get_time_size(self.engine_ptr)
    time_ptr = self.lib.get_time_data(self.engine_ptr)
    time_arr = np.ctypeslib.as_array(time_ptr, shape=(time_size,)).copy()

    history_size = self.lib.get_history_size(self.engine_ptr)
    state_dim = self.lib.get_state_dim(self.engine_ptr)
    history_ptr = self.lib.get_history_data(self.engine_ptr)

    total_points = history_size // state_dim
    raw_arr = np.ctypeslib.as_array(history_ptr, shape=(history_size,))
    states_arr = raw_arr.reshape(total_points, state_dim).copy()

    return time_arr, states_arr

  def get_matrix(self, name: str, params: dict):
    """Avalia uma matriz específica inferindo dimensões automaticamente."""
    if not self.engine_ptr:
      raise RuntimeError("Engine não foi inicializada.")

    matrix_data = self.lpv_model['system_matrices'].get(name)
    if matrix_data is None:
      raise ValueError(
          f"Matriz '{name}' não encontrada ou é nula no modelo.")

    rows = len(matrix_data)
    cols = len(matrix_data[0]) if rows > 0 else 0

    keys = [k.encode('utf-8') for k in params.keys()]
    c_keys = (ctypes.c_char_p * len(keys))(*keys)

    vals = [float(v) for v in params.values()]
    c_vals = (ctypes.c_double * len(vals))(*vals)

    buffer = (ctypes.c_double * (rows * cols))()

    self.lib.get_matrix(
        self.engine_ptr,
        name.encode('utf-8'),
        c_keys,
        c_vals,
        len(params),
        buffer,
        rows,
        cols
    )

    return np.array(buffer).reshape(rows, cols)

  def run_closed_loop_simulation(self, x0_list, K_list, Xi_list, Psi_list, sampling_period, sample_time, duration):
    num_scenarios = len(x0_list)
    n_states = len(x0_list[0])
    nu = K_list[0].shape[0] if hasattr(K_list[0], 'shape') else 1

    x0_flat = np.ascontiguousarray(
        np.array(x0_list), dtype=np.float64).flatten()
    K_flat = np.ascontiguousarray(
        np.array(K_list), dtype=np.float64).flatten()
    Xi_flat = np.ascontiguousarray(
        np.array(Xi_list), dtype=np.float64).flatten()
    Psi_flat = np.ascontiguousarray(
        np.array(Psi_list), dtype=np.float64).flatten()

    self.lib.run_closed_loop(
        self.engine_ptr,
        x0_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n_states,
        K_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nu,
        Xi_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Psi_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sampling_period, num_scenarios, sample_time, duration
    )

    results = []
    for i in range(num_scenarios):
      t_sz, x_sz, u_sz, ev_sz = ctypes.c_int(
      ), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
      self.lib.get_scenario_data_sizes(
          self.engine_ptr, i,
          ctypes.byref(t_sz), ctypes.byref(
              x_sz), ctypes.byref(u_sz), ctypes.byref(ev_sz)
      )

      t_buf = (ctypes.c_double * t_sz.value)()
      x_buf = (ctypes.c_double * x_sz.value)()
      u_buf = (ctypes.c_double * u_sz.value)()
      ev_buf = (ctypes.c_double * ev_sz.value)()

      self.lib.copy_scenario_results(
          self.engine_ptr, i, t_buf, x_buf, u_buf, ev_buf)

      t_arr = np.array(t_buf)
      x_arr = np.array(x_buf).reshape(t_sz.value, n_states)
      u_arr = np.array(u_buf).reshape(t_sz.value, nu)
      ev_arr = np.array(ev_buf)

      results.append({
          "time": t_arr,
          "states": x_arr,
          # Retorna o esforço de controle [N_Passos x Nu]
          "control": u_arr,
          "events": ev_arr
      })

    return results
