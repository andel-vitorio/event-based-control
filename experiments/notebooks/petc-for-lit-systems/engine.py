import ctypes
import os
import sys
import atexit
import numpy as np


class Engine:
  def __init__(self, dll_path="../../../core/dll/petc_for_lit_systems.dll"):
    self.dll_path = os.path.abspath(dll_path)
    if not os.path.exists(self.dll_path):
      raise FileNotFoundError(f"DLL não encontrada: {self.dll_path}")

    if sys.platform == 'win32':
      os.add_dll_directory(os.path.dirname(self.dll_path))

    self.lib = ctypes.CDLL(self.dll_path)

    self.lib.create.restype = ctypes.c_void_p
    self.lib.load_system.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    self.lib.load_sim.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    self.lib.destroy.argtypes = [ctypes.c_void_p]
    self.lib.open_loop.argtypes = [ctypes.c_void_p, ctypes.c_double]
    self.lib.open_loop.restype = None

    self.lib.get_history_size.argtypes = [ctypes.c_void_p]
    self.lib.get_history_size.restype = ctypes.c_size_t

    self.lib.get_state_dim.argtypes = [ctypes.c_void_p]
    self.lib.get_state_dim.restype = ctypes.c_int

    self.lib.get_history_data.argtypes = [ctypes.c_void_p]
    self.lib.get_history_data.restype = ctypes.POINTER(ctypes.c_double)

    self.engine_ptr = None

    # O segredo: registra a destruição automática no encerramento do Kernel
    atexit.register(self.close)

  def open(self):
    """Abre a conexão com a engine C++."""
    if not self.engine_ptr:
      self.engine_ptr = self.lib.create()

  def close(self):
    """Libera a memória C++. Chamado automaticamente pelo Python ao sair."""
    if self.engine_ptr:
      self.lib.destroy(self.engine_ptr)
      self.engine_ptr = None
      print("Engine C++ liberada automaticamente pelo kernel.")

  def load_system(self, path: str):
    if not self.engine_ptr:
      self.open()
    self.lib.load_system(self.engine_ptr, path.encode('utf-8'))

  def load_sim(self, dt: float, tf: float):
    if not self.engine_ptr:
      self.open()
    self.lib.load_sim(self.engine_ptr, dt, tf)

  def open_loop(self, duration: float):
    """
    Executes the open-loop simulation via the C++ core.
    """
    if not self.engine_ptr:
      raise RuntimeError("Engine not initialized.")

    self.lib.open_loop(self.engine_ptr, ctypes.c_double(duration))

  def get_results(self):
    if not self.engine_ptr:
      raise RuntimeError("Engine não foi inicializada.")

    size = self.lib.get_history_size(self.engine_ptr)
    dim = self.lib.get_state_dim(self.engine_ptr)

    # Validação defensiva
    if dim == 0:
      raise RuntimeError(
          "A dimensão do estado é 0. O sistema foi carregado corretamente com loadSystem?")

    if size == 0:
      print(
          "Aviso: O histórico de simulação está vazio. Verifique se o open_loop foi executado.")
      return np.array([])

    data_ptr = self.lib.get_history_data(self.engine_ptr)
    total_points = size // dim

    arr = np.ctypeslib.as_array(data_ptr, shape=(size,))
    return arr.reshape(total_points, dim)
