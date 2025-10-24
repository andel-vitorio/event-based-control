from typing import List, Dict, Optional
from typing import Dict, Any, Callable, Optional
import numpy as np
import math
from scipy.interpolate import interp1d
from IPython.display import display, Math
from numba import njit
from Numeric import *


class System:
  """Generic system node for NCS simulation."""

  def __init__(self, name: str, x0: Optional[np.ndarray] = None,
               dynamics: Optional[Callable] = None,
               output_func: Optional[Callable] = None):
    self.name = name
    self.x = x0.copy() if x0 is not None else None
    self.dynamics = dynamics
    self.output_func = output_func
    self.outputs: Dict[str, Any] = {}

  def update(self, t: float, inputs: Dict[str, Any], dt: float):
    if self.dynamics is not None and self.x is not None:
      self.x = rk5_step(self.dynamics, t, self.x, inputs, dt)

  def output(self, t: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if self.output_func is not None:
      self.outputs = self.output_func(t, self.x, inputs)
    return self.outputs


class StateSpace(System):
  """State-space system with symbolic parameters and disturbances."""

  def __init__(self, data: dict, name: str = "StateSpace", dtype=np.float32):
    super().__init__(name=name, dynamics=None, output_func=None)

    if not isinstance(data, dict):
      raise ValueError(
          "O argumento 'data' deve ser um dicionário válido.")

    self.system_data = data
    self.dtype = dtype
    self.matrices_exprs = self.system_data.get("system_matrices", {})

    labels = self.get_labels()
    self.nu = len(labels.get("inputs", []))
    self.nρ = len(labels.get("parameters", []))
    self.nw = len(labels.get("disturbances", []))
    self.nx = len(labels.get("states", []))

    # Compila expressões
    self.compiled_matrices = {
        key: [[compile(expr, "<string>", "eval") for expr in row]
              for row in matrix]
        for key, matrix in self.matrices_exprs.items()
    }

    self.compiled_params = {
        k: compile(v, "<string>", "eval")
        for k, v in self.system_data.get("parameters", {}).items()
        if isinstance(v, str)
    }

    self.compiled_disturbs = {
        k: compile(v, "<string>", "eval")
        for k, v in self.system_data.get("disturbances", {}).items()
        if isinstance(v, str)
    }

    self.safe_globals = {
        "__builtins__": None,
        "math": math,
        "np": np,
        "abs": abs,
        "pow": pow
    }

    self._x0 = None

    # Atribui os métodos internos
    self.dynamics = self._dynamics
    self.output_func = self._output_func

  # ===========================================================
  # Labels
  # ===========================================================

  def get_labels(self) -> dict:
    return self.system_data.get("labels", {})

  # ===========================================================
  # Simulação
  # ===========================================================

  def set_initial_state(self, x0: np.ndarray):
    if x0.shape[0] != self.nx or x0.shape[1] != 1:
      raise ValueError(f"x0 deve ter dimensão ({self.nx}, 1)")
    self._x0 = x0.astype(self.dtype)
    self.x = self._x0.copy()

  def reset(self):
    if self._x0 is None:
      raise ValueError(
          "Estado inicial não definido. Use setup_simulation(x0).")
    self.x = self._x0.copy()

  # ===========================================================
  # Avaliação interna
  # ===========================================================

  def _dynamics(self, t: float, x: np.ndarray, u_dict: dict, params=None):
    # Converte inputs para vetor NumPy
    u_vec = np.hstack(list(u_dict.values())).astype(
        self.dtype) if u_dict else np.zeros((self.nu,), dtype=self.dtype)

    state_labels = list(self.get_labels().get("states", {}).keys())
    state_dict = dict(zip(state_labels, x.flatten()))

    ρ_vec = self.evaluate_parameters(t, state_dict)
    w_vec = self.evaluate_disturbances(t, state_dict)

    # Cria dicionários apenas para avaliar as matrizes
    param_dict = dict(
        zip(self.get_labels().get("parameters", {}).keys(), ρ_vec))
    disturb_dict = dict(
        zip(self.get_labels().get("disturbances", {}).keys(), w_vec))

    matrices = self.evaluate_matrices(state_dict, param_dict, disturb_dict)

    A = matrices.get("A", np.zeros((self.nx, self.nx), dtype=self.dtype))
    B = matrices.get("B", np.zeros((self.nx, self.nu), dtype=self.dtype))
    E = matrices.get("E", np.zeros((self.nx, self.nw), dtype=self.dtype))

    u_actual = u_vec.reshape(-1,
                             1) if self.nu > 0 else np.zeros((0, 1), dtype=self.dtype)
    w_actual = w_vec.reshape(-1,
                             1) if self.nw > 0 else np.zeros((0, 1), dtype=self.dtype)

    dx = A @ x + B @ u_actual + E @ w_actual
    return dx

  def _output_func(self, t: float, x: np.ndarray, u_dict: dict):
    # Retorna sempre vetor nx x 1
    return x.copy()

  # ===========================================================
  # Avaliação de expressões
  # ===========================================================

  def evaluate_expr(self, code, context: dict) -> float:
    return eval(code, self.safe_globals, context)

  def evaluate_parameters(self, t: float, state: dict = None) -> np.ndarray:
    context = {**(state or {}), "t": t}
    param_labels = list(self.get_labels().get("parameters", {}).keys())
    return np.array([self.evaluate_expr(self.compiled_params[k], context) for k in param_labels],
                    dtype=self.dtype)

  def evaluate_disturbances(self, t: float, state: dict = None) -> np.ndarray:
    context = {**(state or {}), "t": t}
    disturb_labels = list(self.get_labels().get("disturbances", {}).keys())
    return np.array([self.evaluate_expr(self.compiled_disturbs[k], context) for k in disturb_labels],
                    dtype=self.dtype)

  def evaluate_matrices(self, state_dict: dict = None,
                        param_vals: dict = None, disturb_vals: dict = None) -> dict:
    env = {}
    if state_dict:
      env.update(state_dict)
    if param_vals:
      env.update({k: float(v) for k, v in param_vals.items()})
    if disturb_vals:
      env.update({k: float(v) for k, v in disturb_vals.items()})

    evaluated = {}
    for key, matrix in self.compiled_matrices.items():
      evaluated[key] = np.array(
          [[self.evaluate_expr(expr, env) for expr in row]
           for row in matrix],
          dtype=self.dtype
      )
    return evaluated

  def matrices_func(self, ρi):
    """
    Recebe ρi: lista de valores dos parâmetros no vértice
    Retorna dicionário com as matrizes A, B, C, E no dtype da planta
    """
    # Pega os nomes dos parâmetros
    labels = self.get_labels()
    param_names = labels.get("parameters", [])
    param_dict = dict(zip(param_names, ρi))

    # Avalia as matrizes usando a planta
    matrices = self.evaluate_matrices(
        state_dict=None, param_vals=param_dict, disturb_vals={})

    # Garante o tipo dtype da planta
    return {k: np.array(v, dtype=self.dtype) for k, v in matrices.items()}


class NetworkedControlSystem:
  def __init__(self, dtype=np.float32):
    self.systems = {}
    self.dtype = dtype       # Tipo numérico padrão
    self.t = self.dtype(0.0)
    self.dt = self.dtype(1e-4)
    self.duration = self.dtype(0.0)
    self.n_steps = 0
    self.current_step = 0
    self.output_history = {}  # histórico de saídas
    self.time_history = None

  def add_system(self, system: System, name: str):
    self.systems[name] = system

  def setup_clock(self, duration: float, dt: float = 1e-4):
    self.duration = self.dtype(duration)
    self.dt = self.dtype(dt)
    self.t = self.dtype(0.0)
    self.n_steps = int(np.ceil(duration / dt))
    self.current_step = 0

    # Inicializa histórico de saídas e tempo
    self.output_history = {}
    self.time_history = np.linspace(
        0, duration, self.n_steps, dtype=self.dtype)

    for name, sys in self.systems.items():
      ny = getattr(sys, 'ny', getattr(sys, 'nx', 1))
      self.output_history[name] = np.zeros(
          (ny, self.n_steps), dtype=self.dtype)

  def advance_clock(self) -> bool:
    if self.current_step < self.n_steps:
      self.t = self.dtype(self.current_step) * self.dt
      decimals = int(abs(np.floor(np.log10(float(self.dt))))) + 1
      self.t = np.round(self.t, decimals).astype(self.dtype)
      self.current_step += 1
      return True
    return False

  def update_systems(self, inputs: dict):
    for name, sys in self.systems.items():
      u = inputs.get(name, None)
      sys.update(self.t, u, self.dt)
      y_vec = sys.output(self.t, u).flatten()  # vetor nx,1
      self.output_history[name][:,
                                self.current_step-1] = y_vec.astype(self.dtype)

    return self.t, self.output_history

  def get_system_output(self, system_name: str) -> np.ndarray:
    sys = self.get_system(system_name)
    y = sys.output(self.t, {})
    # Se a saída for vetor NumPy, apenas converte dtype
    return np.array(y, dtype=self.dtype).reshape(-1, 1)

  def get_system(self, system_name: str):
    if system_name not in self.systems:
      raise ValueError(f"Sistema '{system_name}' não está na rede.")
    return self.systems[system_name]

  def reset_clock(self):
    self.t = self.dtype(0.0)
    self.current_step = 0
    for sys in self.systems.values():
      sys.reset()


class Sampler:
  """
  Sampler clock for NCS. Returns True at sampling instants.
  """

  def __init__(self, Ts: float, dtype=np.float32):
    self.Ts = dtype(Ts)
    self.dtype = dtype
    self.last_sample_time = self.dtype(-Ts)

  def check(self, t: float) -> bool:
    """
    Call at each simulation step.
    Returns True if a new sample occurs at time t.
    """
    if t - self.last_sample_time >= self.Ts - np.finfo(self.dtype).eps:
      self.last_sample_time = t
      return True
    return False

  def reset(self):
    self.last_sample_time = -self.Ts
