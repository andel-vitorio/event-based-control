import ctypes
from ctypes import wintypes
import json
import os
import sys
from typing import Any, Dict, Optional, Union
import numpy as np


class Engine:
  def __init__(self, dll_path: str = "dll/petc_for_lit_systems.dll"):
    self.dll_path = os.path.abspath(dll_path)
    self.lib: Optional[ctypes.CDLL] = None
    self.engine_ptr: Optional[ctypes.c_void_p] = None
    self.model_data: Optional[Dict[str, Any]] = None

    self.nx: int = 0
    self.nu: int = 0
    self.ny: int = 0

    if not os.path.exists(self.dll_path):
      raise FileNotFoundError(
          f"DLL não encontrada no caminho: {self.dll_path}")

    if sys.platform == "win32":
      os.add_dll_directory(os.path.dirname(self.dll_path))

    self.lib = ctypes.CDLL(self.dll_path)
    self._initialize_dll_bindings()

    self.engine_ptr = self.lib.create_dual_channel_engine()
    if not self.engine_ptr:
      raise RuntimeError(
          "Falha ao instanciar o motor interno DllEngine na DLL."
      )

  def _initialize_dll_bindings(self) -> None:
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    # 1. Ciclo de Vida
    self.lib.create_dual_channel_engine.restype = ctypes.c_void_p
    self.lib.create_dual_channel_engine.argtypes = []

    self.lib.destroy_dual_channel_engine.restype = None
    self.lib.destroy_dual_channel_engine.argtypes = [ctypes.c_void_p]

    # 2. Carregamento e Metadados do Sistema
    self.lib.load_system_dual_channel.restype = ctypes.c_int
    self.lib.load_system_dual_channel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]

    self.lib.get_system_dimensions.restype = ctypes.c_int
    self.lib.get_system_dimensions.argtypes = [
        ctypes.c_void_p,
        c_int_p,
        c_int_p,
        c_int_p,
    ]

    # 3. Execução da Simulação Dual-Channel Aumentada
    self.lib.run_dual_channel_simulation.restype = ctypes.c_int
    self.lib.run_dual_channel_simulation.argtypes = [
        ctypes.c_void_p,  # EnginePtr
        c_double_p,
        c_double_p,
        c_double_p,  # x0, x_hat0, x_hat_a0
        c_double_p,
        c_double_p,
        c_double_p,
        c_double_p,  # K, L0, L1, L2
        c_double_p,
        c_double_p,
        ctypes.c_double,
        ctypes.c_double,  # SC: Psi, Xi, sigma, threshold
        c_double_p,
        c_double_p,
        ctypes.c_double,
        ctypes.c_double,  # CA: Psi, Xi, sigma, threshold
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,  # sampling_period, duration, time_step
        ctypes.c_double,
        ctypes.c_double,  # max_iet_sc, max_iet_ca
    ]

    # 4. Extração de Tamanhos dos Vetores
    self.lib.get_simulation_result_sizes.restype = None
    self.lib.get_simulation_result_sizes.argtypes = [
        ctypes.c_void_p,  # EnginePtr
        c_int_p,  # num_steps
        c_int_p,  # states_sz
        c_int_p,  # est_states_sz
        c_int_p,  # error_sz
        c_int_p,  # control_sz
        c_int_p,  # sc_events_sz
        c_int_p,  # ca_events_sz
    ]

    # 5. Cópia Contígua de Buffers
    self.lib.copy_simulation_data.restype = None
    self.lib.copy_simulation_data.argtypes = [
        ctypes.c_void_p,  # EnginePtr
        c_double_p,  # t_out
        c_double_p,  # x_out
        c_double_p,  # x_est_out
        c_double_p,  # e_out
        c_double_p,  # u_out
        c_double_p,  # sc_events_out
        c_double_p,  # ca_events_out
    ]

  def __del__(self) -> None:
    self.close()

  def close(self) -> None:
    if self.engine_ptr and self.lib:
      self.lib.destroy_dual_channel_engine(self.engine_ptr)
      self.engine_ptr = None

    if self.lib:
      dll_handle = self.lib._handle
      del self.lib
      self.lib = None

      if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
        kernel32.FreeLibrary(dll_handle)

  def load_lit_model(self, json_path: str) -> None:
    if not self.engine_ptr:
      raise RuntimeError("Instância do motor não inicializada.")

    full_path = os.path.abspath(json_path)
    if not os.path.exists(full_path):
      raise FileNotFoundError(f"Arquivo de modelo não encontrado: {full_path}")

    res = self.lib.load_system_dual_channel(
        self.engine_ptr, full_path.encode("utf-8")
    )
    if res != 0:
      raise RuntimeError(
          f"Falha interna em C++ ao carregar o modelo JSON (código: {res})."
      )

    c_nx = ctypes.c_int()
    c_nu = ctypes.c_int()
    c_ny = ctypes.c_int()

    res_dim = self.lib.get_system_dimensions(
        self.engine_ptr,
        ctypes.byref(c_nx),
        ctypes.byref(c_nu),
        ctypes.byref(c_ny),
    )
    if res_dim != 0:
      raise RuntimeError(
          "Falha ao recuperar as dimensões do sistema carregado.")

    self.nx = c_nx.value
    self.nu = c_nu.value
    self.ny = c_ny.value

    try:
      with open(full_path, "r", encoding="utf-8") as f:
        self.model_data = json.load(f)
    except Exception as err:
      print(f"[Aviso] Falha ao carregar metadados brutos do JSON: {err}")
      self.model_data = None

  def run_dual_channel_simulation(
      self,
      x0: Union[np.ndarray, list],
      x_hat0: Union[np.ndarray, list],
      x_hat_a0: Union[np.ndarray, list],
      K: np.ndarray,
      L0: np.ndarray,
      L1: np.ndarray,
      L2: np.ndarray,
      Psi_sc: np.ndarray,
      Xi_sc: np.ndarray,
      Psi_ca: np.ndarray,
      Xi_ca: np.ndarray,
      sampling_period: float,
      duration: float,
      time_step: float = 1e-4,
      sigma_sc: float = 1.0,
      threshold_sc: float = 0.0,
      sigma_ca: float = 1.0,
      threshold_ca: float = 0.0,
      max_iet_sc: Optional[float] = None,
      max_iet_ca: Optional[float] = None,
  ) -> Dict[str, np.ndarray]:
    if not self.engine_ptr:
      raise RuntimeError("Instância do motor não inicializada.")
    if self.nx == 0:
      raise RuntimeError(
          "Carregue um modelo LIT via load_lit_model(...) antes de simular."
      )

    # 1. Normalização de Arrays para C-Contiguous float64
    x0_arr = np.ascontiguousarray(x0, dtype=np.float64).flatten()
    x_hat0_arr = np.ascontiguousarray(x_hat0, dtype=np.float64).flatten()
    x_hat_a0_arr = np.ascontiguousarray(x_hat_a0, dtype=np.float64).flatten()

    K_arr = np.ascontiguousarray(K, dtype=np.float64)
    L0_arr = np.ascontiguousarray(L0, dtype=np.float64)
    L1_arr = np.ascontiguousarray(L1, dtype=np.float64)
    L2_arr = np.ascontiguousarray(L2, dtype=np.float64)

    Psi_sc_arr = np.ascontiguousarray(Psi_sc, dtype=np.float64)
    Xi_sc_arr = np.ascontiguousarray(Xi_sc, dtype=np.float64)
    Psi_ca_arr = np.ascontiguousarray(Psi_ca, dtype=np.float64)
    Xi_ca_arr = np.ascontiguousarray(Xi_ca, dtype=np.float64)

    # 2. Validações Dimensionais Estritas
    if x0_arr.size != self.nx:
      raise ValueError(
          f"x0 deve ter dimensão {self.nx}. Fornecido: {x0_arr.size}."
      )
    if x_hat0_arr.size != self.nx:
      raise ValueError(
          f"x_hat0 deve ter dimensão {self.nx}. Fornecido: {x_hat0_arr.size}."
      )
    if x_hat_a0_arr.size != self.nx:
      raise ValueError(
          f"x_hat_a0 deve ter dimensão {self.nx}. Fornecido:"
          f" {x_hat_a0_arr.size}."
      )

    if K_arr.shape != (self.nu, self.nx):
      raise ValueError(
          f"K deve ser ({self.nu}, {self.nx}). Fornecido: {K_arr.shape}."
      )
    if L0_arr.shape != (self.nx, self.nx):
      raise ValueError(
          f"L0 deve ser ({self.nx}, {self.nx}). Fornecido: {L0_arr.shape}."
      )
    if L1_arr.shape != (self.nx, self.ny):
      raise ValueError(
          f"L1 deve ser ({self.nx}, {self.ny}). Fornecido: {L1_arr.shape}."
      )
    if L2_arr.shape != (self.nx, self.ny):
      raise ValueError(
          f"L2 deve ser ({self.nx}, {self.ny}). Fornecido: {L2_arr.shape}."
      )

    if Psi_sc_arr.shape != (self.ny, self.ny) or Xi_sc_arr.shape != (
        self.ny,
        self.ny,
    ):
      raise ValueError(
          f"Matrizes SC devem ter dimensão ({self.ny}, {self.ny})."
      )
    if Psi_ca_arr.shape != (self.nx, self.nx) or Xi_ca_arr.shape != (
        self.nx,
        self.nx,
    ):
      raise ValueError(
          f"Matrizes CA devem ter dimensão ({self.nx}, {self.nx})."
      )

    # 3. Resolução dos Limites Máximos
    actual_max_iet_sc = (
        float(max_iet_sc) if max_iet_sc is not None else float(sampling_period)
    )
    actual_max_iet_ca = (
        float(max_iet_ca) if max_iet_ca is not None else float(sampling_period)
    )

    # Função auxiliar de ponteiro de dados
    def ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
      return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # 4. Disparo da Simulação em C++
    status = self.lib.run_dual_channel_simulation(
        self.engine_ptr,
        ptr(x0_arr),
        ptr(x_hat0_arr),
        ptr(x_hat_a0_arr),
        ptr(K_arr),
        ptr(L0_arr),
        ptr(L1_arr),
        ptr(L2_arr),
        ptr(Psi_sc_arr),
        ptr(Xi_sc_arr),
        ctypes.c_double(sigma_sc),
        ctypes.c_double(threshold_sc),
        ptr(Psi_ca_arr),
        ptr(Xi_ca_arr),
        ctypes.c_double(sigma_ca),
        ctypes.c_double(threshold_ca),
        ctypes.c_double(sampling_period),
        ctypes.c_double(duration),
        ctypes.c_double(time_step),
        ctypes.c_double(actual_max_iet_sc),
        ctypes.c_double(actual_max_iet_ca),
    )

    if status != 0:
      raise RuntimeError(
          f"Falha na integração numérica do integrador RK5 (código: {status})."
      )

    # 5. Recuperação Estruturada das Dimensões dos Resultados
    c_n_steps = ctypes.c_int()
    c_sz_x = ctypes.c_int()
    c_sz_x_est = ctypes.c_int()
    c_sz_e = ctypes.c_int()
    c_sz_u = ctypes.c_int()
    c_sz_sc = ctypes.c_int()
    c_sz_ca = ctypes.c_int()

    self.lib.get_simulation_result_sizes(
        self.engine_ptr,
        ctypes.byref(c_n_steps),
        ctypes.byref(c_sz_x),
        ctypes.byref(c_sz_x_est),
        ctypes.byref(c_sz_e),
        ctypes.byref(c_sz_u),
        ctypes.byref(c_sz_sc),
        ctypes.byref(c_sz_ca),
    )

    n_steps = c_n_steps.value
    sz_x = c_sz_x.value
    sz_x_est = c_sz_x_est.value
    sz_e = c_sz_e.value
    sz_u = c_sz_u.value
    sz_sc = c_sz_sc.value
    sz_ca = c_sz_ca.value

    # 6. Alocação Contígua e Cópia em Memória
    t_out = np.empty(n_steps, dtype=np.float64)
    x_out = np.empty(sz_x, dtype=np.float64)
    x_est_out = np.empty(sz_x_est, dtype=np.float64)
    e_out = np.empty(sz_e, dtype=np.float64)
    u_out = np.empty(sz_u, dtype=np.float64)
    sc_out = np.empty(sz_sc, dtype=np.float64)
    ca_out = np.empty(sz_ca, dtype=np.float64)

    self.lib.copy_simulation_data(
        self.engine_ptr,
        ptr(t_out),
        ptr(x_out),
        ptr(x_est_out),
        ptr(e_out),
        ptr(u_out),
        ptr(sc_out),
        ptr(ca_out),
    )

    # 7. Reconstrução Matricial
    return {
        "time": t_out,
        "states": x_out.reshape((n_steps, self.nx)),
        "estimated_states": x_est_out.reshape((n_steps, self.nx)),
        "estimation_error": e_out.reshape((n_steps, self.nx)),
        "control": u_out.reshape((n_steps, self.nu)),
        "sc_trigger_times": sc_out,
        "ca_trigger_times": ca_out,
    }
