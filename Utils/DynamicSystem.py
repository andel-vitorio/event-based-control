import traceback
import itertools
from typing import List, Dict, Optional
from typing import Dict, Any, Callable, Optional
import numpy as np
import math
from scipy.interpolate import interp1d
from IPython.display import display, Math
from .Numeric import *
import re


import hashlib
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
  """
  Codificador JSON customizado para lidar com tipos de dados do NumPy
  que não são serializáveis por padrão.
  """

  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()  # Converte arrays para listas
    return super(NpEncoder, self).default(obj)


def create_simulation_id(plant_object, design_params_dict, scheme_name: str = "DETM") -> str:
  """
  Gera um ID de hash MD5 único baseado nos dados de definição da planta,
  nos parâmetros de projeto e no esquema de controle.

  Args:
      plant_object: A instância da sua classe StateSpace (que contém .system_data).
      design_params_dict: O dicionário com os parâmetros de projeto (h, v, etc.).
      scheme_name: Nome do esquema (ex: 'DETM', 'SETM', 'SETM_STAR', 'AETM').

  Returns:
      Uma string de hash MD5 (ex: "sim_a1b2c3d4...")
  """

  try:
    plant_definition = plant_object.system_data
  except AttributeError:
    raise ValueError(
        "O objeto 'plant' não possui o atributo 'system_data'.")

  combined_data = {
      "plant_definition": plant_definition,
      "design_parameters": design_params_dict,
      "scheme_name": scheme_name
  }

  data_string = json.dumps(
      combined_data,
      sort_keys=True,
      cls=NpEncoder
  )

  hash_object = hashlib.md5(data_string.encode('utf-8'))
  hash_id = hash_object.hexdigest()

  return f"sim_{hash_id}"


class System:
  """Generic system node for simulation."""

  def __init__(self, name: str, x0: Optional[np.ndarray] = None,
               dynamics: Optional[Callable] = None,
               output_func: Optional[Callable] = None):
    self.name = name
    self.states = x0.copy() if x0 is not None else None
    self.dynamics = dynamics
    self.output_func = output_func
    self.outputs: Dict[str, Any] = {}

  def update(self, t: float, inputs: Dict[str, Any], dt: float):
    if self.dynamics is not None:
      if self.states is None:
        raise ValueError(
            f"Invalid state in '{self.name}': 'dynamics' is defined but 'states' is None."
        )
      self.states = rk5_step(self.dynamics, t, self.states, inputs, dt)

  def output(self, t: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if self.output_func is not None:
      self.outputs = self.output_func(t, self.states, inputs)
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
    self.matrices_exprs = self.system_data.get("system_matrices") or {}

    labels = self.get_labels()

    def _safe_len(obj):
      return len(obj) if obj is not None else 0

    self.nu = _safe_len(labels.get("inputs"))
    self.n_rho = _safe_len(labels.get("parameters"))
    self.nw = _safe_len(labels.get("disturbances"))
    self.nx = _safe_len(labels.get("states"))
    self.ny = _safe_len(labels.get("outputs"))
    self.nz = _safe_len(labels.get("performance_outputs"))

    self.bounds = self.system_data.get("bounds") or {}

    # Compila expressões
    self.compiled_matrices = {}
    for key, matrix in self.matrices_exprs.items():
      if matrix is not None:
        self.compiled_matrices[key] = [[compile(expr, "<string>", "eval") for expr in row]
                                       for row in matrix]

    params_data = self.system_data.get("parameters") or {}
    self.compiled_params = {
        k: compile(v, "<string>", "eval")
        for k, v in params_data.items()
        if isinstance(v, str)
    }

    disturbs_data = self.system_data.get("disturbances") or {}
    self.compiled_disturbs = {
        k: compile(v, "<string>", "eval")
        for k, v in disturbs_data.items()
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
    self.dynamics = self._dynamics
    self.output_func = self._output_func

  # ===========================================================
  # Labels
  # ===========================================================
  def get_labels(self) -> dict:
    return self.system_data.get("labels") or {}

  # ===========================================================
  # Bounds
  # ===========================================================
  def get_bounds(self) -> dict:
    return self.bounds

  def get_input_bounds(self) -> np.ndarray:
    u_b = self.bounds.get("u")
    if u_b is None:
      if self.nu > 0:
        return np.full((self.nu, 1), np.inf, dtype=self.dtype)
      return np.zeros((0, 1), dtype=self.dtype)
    return np.array(u_b, dtype=self.dtype).reshape(-1, 1)

  def get_parameter_bounds(self) -> np.ndarray:
    p_b = self.bounds.get("ρ")
    if p_b is None:
      return np.zeros((self.n_rho, 2), dtype=self.dtype)
    return np.array(p_b, dtype=self.dtype)

  def get_disturbance_l2_norm_bound(self) -> np.ndarray:
    w_l2 = self.bounds.get("w_l2_norm")
    if w_l2 is None:
      return np.array([], dtype=self.dtype)
    return np.array(w_l2, dtype=self.dtype)

  # ===========================================================
  # Simulação
  # ===========================================================
  def set_initial_state(self, x0: np.ndarray):
    if x0.shape[0] != self.nx or x0.shape[1] != 1:
      raise ValueError(f"x0 deve ter dimensão ({self.nx}, 1)")
    self._x0 = x0.astype(self.dtype)
    self.states = self._x0.copy()

  def reset(self):
    if self._x0 is None:
      raise ValueError(
          "Estado inicial não definido. Use setup_simulation(x0).")
    self.states = self._x0.copy()

  # ===========================================================
  # Avaliação interna
  # ===========================================================

  def _get_simulation_context(self, t: float, x: np.ndarray, u_dict: dict) -> tuple:
    u_vec = np.hstack(list(u_dict.values())).astype(
        self.dtype) if u_dict else np.zeros((self.nu,), dtype=self.dtype)

    state_labels = list((self.get_labels().get("states") or {}).keys())
    state_dict = dict(zip(state_labels, x.flatten()))

    rho_vec = self.evaluate_parameters(t, state_dict)
    w_vec = self.evaluate_disturbances(t, state_dict)

    param_dict = dict(
        zip((self.get_labels().get("parameters") or {}).keys(), rho_vec))
    disturb_dict = dict(
        zip((self.get_labels().get("disturbances") or {}).keys(), w_vec))

    matrices = self.evaluate_matrices(state_dict, param_dict, disturb_dict)

    u_actual = u_vec.reshape(-1,
                             1) if self.nu > 0 else np.zeros((0, 1), dtype=self.dtype)
    w_actual = w_vec.reshape(-1,
                             1) if self.nw > 0 else np.zeros((0, 1), dtype=self.dtype)

    return matrices, u_actual, w_actual

  def _dynamics(self, t: float, x: np.ndarray, u_dict: dict, params=None):
    matrices, u_actual, w_actual = self._get_simulation_context(
        t, self.states, u_dict)

    A = matrices.get("A", np.zeros((self.nx, self.nx), dtype=self.dtype))
    B = matrices.get("B", np.zeros((self.nx, self.nu), dtype=self.dtype))
    E = matrices.get("E", np.zeros((self.nx, self.nw), dtype=self.dtype))

    dx = A @ self.states + B @ u_actual + E @ w_actual
    return dx

  def _output_func(self, t: float, x: np.ndarray, u_dict: dict):
    matrices, u_actual, _ = self._get_simulation_context(t, x, u_dict)
    C = matrices.get("C", np.zeros((self.ny, self.nx), dtype=self.dtype))
    D = matrices.get("D", np.zeros((self.ny, self.nu), dtype=self.dtype))
    y = C @ x + D @ u_actual
    return y

  def get_performance_output(self, t: float, x: np.ndarray, u_dict: dict) -> np.ndarray:
    """Calcula a saída de desempenho z = Cz*x + Dz*u."""
    matrices, u_actual, _ = self._get_simulation_context(t, x, u_dict)
    Cz = matrices.get("Cz", np.zeros((self.nz, self.nx), dtype=self.dtype))
    Dz = matrices.get("Dz", np.zeros((self.nz, self.nu), dtype=self.dtype))
    z = Cz @ x + Dz @ u_actual
    return z

  # <<< MUDANÇA: Nova seção para gerar LaTeX >>>
  # ===========================================================
  # Representação LaTeX
  # ===========================================================

  def _format_latex_symbol(self, s: str) -> str:
    """Formata uma string simbólica para LaTeX."""
    # Converte 'p1' em 'p_{1}', 'x10' em 'x_{10}', etc.
    s_tex = re.sub(r"([a-zA-Z])(\d+)", r"\1_{\2}", s)

    # Converte funções matemáticas
    s_tex = s_tex.replace("*", r" \cdot ")
    s_tex = s_tex.replace("math.cos", r"\cos")
    s_tex = s_tex.replace("math.sin", r"\sin")
    s_tex = s_tex.replace("math.exp", r"\exp")
    s_tex = s_tex.replace("math.pi", r"\pi")
    return s_tex

  def _build_latex_vector(self, label_keys: list, is_derivative=False) -> str:
    """Cria um vetor em LaTeX (ex: bmatrix) a partir de uma lista de labels."""
    if not label_keys:
      return ""

    prefix = r"\dot{" if is_derivative else ""
    suffix = "}" if is_derivative else ""

    elements = []
    for key in label_keys:
      # Formata o label (ex: 'x1' -> 'x_{1}')
      formatted_key = re.sub(r"([a-zA-Z])(\d+)", r"\1_{\2}", key)
      elements.append(f"{prefix}{formatted_key}{suffix}")

    body = r" \\ ".join(elements)
    return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"

  def _build_latex_matrix(self, matrix_key: str) -> str:
    """Cria uma matriz em LaTeX (ex: bmatrix) a partir de uma chave de matriz."""
    matrix_expr = self.matrices_exprs.get(matrix_key)

    # Se a matriz não estiver definida (ex: D=0), retorna a letra em negrito
    if not matrix_expr:
      return rf"\mathbf{{{matrix_key}}}"

    rows_tex = []
    for row in matrix_expr:
      # Formata cada elemento da linha
      elements_tex = [self._format_latex_symbol(el) for el in row]
      rows_tex.append(" & ".join(elements_tex))

    body = r" \\ ".join(rows_tex)
    return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"

  def get_latex_equations(self) -> str:
    """
    Retorna as equações de estado simbólicas formatadas em LaTeX.
    """
    # 1. Obter listas de labels
    labels = self.get_labels()
    state_labels = list((labels.get("states") or {}).keys())
    input_labels = list((labels.get("inputs") or {}).keys())
    disturb_labels = list((labels.get("disturbances") or {}).keys())
    output_labels = list((labels.get("outputs") or {}).keys())
    perf_labels = list((labels.get("performance_outputs") or {}).keys())

    # 2. Construir todos os vetores LaTeX
    x_dot_vec = self._build_latex_vector(state_labels, is_derivative=True)
    x_vec = self._build_latex_vector(state_labels)
    u_vec = self._build_latex_vector(input_labels)
    w_vec = self._build_latex_vector(disturb_labels)
    y_vec = self._build_latex_vector(output_labels)
    z_vec = self._build_latex_vector(perf_labels)

    # 3. Construir todas as matrizes LaTeX
    A_mat = self._build_latex_matrix("A")
    B_mat = self._build_latex_matrix("B")
    E_mat = self._build_latex_matrix("E")
    C_mat = self._build_latex_matrix("C")
    D_mat = self._build_latex_matrix("D")
    Cz_mat = self._build_latex_matrix("Cz")
    Dz_mat = self._build_latex_matrix("Dz")

    # 4. Montar as equações
    eqs = []

    # Equação de estado
    dyn_eq = [rf"{A_mat} {x_vec}"]
    if self.nu > 0:
      dyn_eq.append(rf"+ {B_mat} {u_vec}")
    if self.nw > 0:
      dyn_eq.append(rf"+ {E_mat} {w_vec}")
    eqs.append(rf"{x_dot_vec} &= {' '.join(dyn_eq)}")

    # Equação de saída (medição)
    if self.ny > 0:
      y_eq = [rf"{C_mat} {x_vec}"]
      if self.nu > 0 and self.matrices_exprs.get("D"):
        y_eq.append(rf"+ {D_mat} {u_vec}")
      eqs.append(rf"{y_vec} &= {' '.join(y_eq)}")

    # Equação de saída (desempenho)
    if self.nz > 0:
      z_eq = [rf"{Cz_mat} {x_vec}"]
      if self.nu > 0 and self.matrices_exprs.get("Dz"):
        z_eq.append(rf"+ {Dz_mat} {u_vec}")
      eqs.append(rf"{z_vec} &= {' '.join(z_eq)}")

    # 5. Juntar tudo em um ambiente 'align*'
    body = r" \\ ".join(eqs)
    return rf"\begin{{align*}} {body} \end{{align*}}"

  # ===========================================================
  # Avaliação de expressões
  # ===========================================================
  # ... (resto da classe permanece igual) ...
  def evaluate_expr(self, code, context: dict) -> float:
    return eval(code, self.safe_globals, context)

  def evaluate_parameters(self, t: float, state: dict = None) -> np.ndarray:
    context = {**(state or {}), "t": t}
    param_labels = list((self.get_labels().get("parameters") or {}).keys())
    return np.array([self.evaluate_expr(self.compiled_params[k], context) for k in param_labels],
                    dtype=self.dtype)

  def evaluate_disturbances(self, t: float, state: dict = None) -> np.ndarray:
    context = {**(state or {}), "t": t}
    disturb_labels = list((self.get_labels().get("disturbances") or {}).keys())
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

  def matrices_func(self, rho_i):
    """
    Recebe rho_i: lista de valores dos parâmetros no vértice
    Retorna dicionário com todas as matrizes (A, B, C, D, E, Cz, Dz...)
    """
    labels = self.get_labels()
    param_names = list((labels.get("parameters") or {}).keys())
    param_dict = dict(zip(param_names, rho_i))

    matrices = self.evaluate_matrices(
        state_dict=None, param_vals=param_dict, disturb_vals={})

    return {k: np.array(v, dtype=self.dtype) for k, v in matrices.items()}

  def export_matrices_jit(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exporta as matrizes A, B, C, D como arrays contíguos em memória (C-order),
    formatados especificamente para consumo por kernels Numba.

    Returns:
        A, B, C, D (np.float64, C-contiguous)
    """
    # Avalia no contexto vazio (t=0, x=0) para resolver strings numéricas
    mats = self.evaluate_matrices()

    # ascontiguousarray é CRUCIAL para vetorização no Numba
    A = np.ascontiguousarray(
        mats.get('A', np.zeros((self.nx, self.nx))), dtype=np.float64)
    B = np.ascontiguousarray(
        mats.get('B', np.zeros((self.nx, self.nu))), dtype=np.float64)
    C = np.ascontiguousarray(
        mats.get('C', np.zeros((self.ny, self.nx))), dtype=np.float64)

    if mats.get('D') is not None:
      D = np.ascontiguousarray(mats['D'], dtype=np.float64)
    else:
      D = np.zeros((self.ny, self.nu), dtype=np.float64)

    return A, B, C, D

  def get_input_bounds_jit(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna os limites de saturação (min, max) formatados para Numba.
    Se não houver bounds no JSON, retorna +/- infinito.
    """
    bounds_data = self.system_data.get("bounds", {}).get("u")

    if bounds_data is None:
      u_max = np.full(self.nu, np.inf, dtype=np.float64)
      u_min = np.full(self.nu, -np.inf, dtype=np.float64)
    else:
      # Assume que o JSON fornece o valor absoluto max (saturação simétrica)
      # Se o JSON tiver [min, max], ajustar lógica aqui.
      vals = np.array(bounds_data, dtype=np.float64).flatten()
      u_max = vals
      u_min = -vals

    return np.ascontiguousarray(u_min), np.ascontiguousarray(u_max)


class _SimulationStep:
  """
  Classe auxiliar interna (Gerenciador de Contexto) para gerenciar
  um único passo de simulação.
  """

  def __init__(self, sim_instance, default_inputs=None):
    self.sim = sim_instance
    if default_inputs is None:
      self.inputs = {}
    else:
      self.inputs = default_inputs

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_type is None:
      self.sim.update_systems(self.inputs)
    return False


class SimulationEngineOld:
  def __init__(self, dtype=np.float32):
    self.systems = {}
    self.dtype = dtype
    self.t = self.dtype(0.0)
    self.dt = self.dtype(1e-4)
    self.duration = self.dtype(0.0)
    self.n_steps = 0
    self.current_step = 0
    self.output_history = {}
    self.time_history = None

  def add_system(self, system):
    if not getattr(system, "name", None):
      raise ValueError("Cannot add a system without a valid name.")
    self.systems[system.name] = system

  def setup_clock(self, duration: float, dt: float = 1e-4):
    self.duration = self.dtype(duration)
    self.dt = self.dtype(dt)
    self.t = self.dtype(0.0)
    self.n_steps = int(np.ceil(duration / dt))
    self.current_step = 0

    self.time_history = np.linspace(
        0, duration, self.n_steps, dtype=self.dtype)
    self.output_history = {}
    for name in self.systems:
      self.output_history[name] = [None] * self.n_steps

  def advance_clock(self) -> bool:
    if self.current_step < self.n_steps:
      self.t = self.dtype(self.current_step) * self.dt
      decimals = int(abs(np.floor(np.log10(float(self.dt))))) + 1
      self.t = np.round(self.t, decimals).astype(self.dtype)
      self.current_step += 1
      return True
    return False

  def step(self, default_inputs=None):
    """Context manager para um passo de simulação."""
    return _SimulationStep(self, default_inputs)

  def update_systems(self, inputs: dict):
    for name, sys in self.systems.items():
      u = inputs.get(name, None)
      sys.update(self.t, u, self.dt)
      y = sys.output(self.t, u)

      # Converte qualquer saída para array 2D
      y_array = np.atleast_1d(y).reshape(-1, 1)
      self.output_history[name][self.current_step - 1] = y_array

  def finalize_history(self):
    """Converte listas temporárias em arrays NumPy 2D (ny, n_steps)."""
    for name, hist in self.output_history.items():
      # hist é uma lista de arrays (ny,1) ou (1,1)
      self.output_history[name] = np.concatenate(
          hist, axis=1)  # concatena colunas

  def get_system_output(self, system_name: str) -> np.ndarray:
    sys = self.get_system(system_name)
    y = sys.output(self.t, {})
    return np.atleast_1d(y).reshape(-1, 1).astype(self.dtype)

  def get_system(self, system_name: str):
    if system_name not in self.systems:
      raise ValueError(f"Sistema '{system_name}' não está na simulação.")
    return self.systems[system_name]

  def reset_clock(self):
    self.t = self.dtype(0.0)
    self.current_step = 0
    for sys in self.systems.values():
      sys.reset()


class SimulationEngine:
  """
  Infraestrutura de execução agnóstica.
  Atua como um Container de Sistemas e um 'Runner' para Kernels JIT.
  """

  def __init__(self, dtype=np.float64):
    self.systems: Dict[str, StateSpace] = {}
    self.dtype = dtype

    # Parâmetros de Tempo
    self.t = 0.0
    self.dt = 0.0
    self.duration = 0.0
    self.n_steps = 0

    # Históricos
    self.time_history: Optional[np.ndarray] = None
    self.output_history: Dict[str, np.ndarray] = {}

  def add_system(self, system: StateSpace):
    """Registra um sistema no motor."""
    if not getattr(system, "name", None):
      raise ValueError("Sistema sem nome inválido.")
    self.systems[system.name] = system

  def setup_clock(self, duration: float, dt: float = 1e-4):
    """
    Configura a base de tempo da simulação.
    """
    self.duration = float(duration)
    self.dt = float(dt)
    # Cálculo seguro para n_steps
    self.n_steps = int(np.ceil(self.duration / self.dt)) + 1

    # Pré-aloca vetor de tempo para referência futura
    self.time_history = np.linspace(
        0.0, self.duration, self.n_steps, dtype=self.dtype)

  def run_kernel(self, kernel_func, *args) -> Any:
    """c
    Método Genérico de Despacho JIT.
    Executa uma função kernel (definida em simulator.py) passando os argumentos.

    Args:
        kernel_func: Função decorada com @njit contendo a lógica do loop.
        *args: Argumentos posicionais esperados pelo kernel (arrays, floats).

    Returns:
        O retorno bruto do kernel (geralmente tuplas de arrays).
    """
    # Aqui é possível adicionar instrumentação (tempo de execução, logs)
    return kernel_func(*args)

  def get_system(self, name: str) -> StateSpace:
    return self.systems[name]

  def finalize(self):
    """Método opcional para limpeza ou pós-processamento de dados."""
    pass


class Sampler:
  """
  Sampler clock for simulation. Returns True at sampling instants.
  """

  def __init__(self, Ts: float, time_source: Optional[Callable[[], float]] = None, dtype=np.float32):
    self.Ts = dtype(Ts)
    self.dtype = dtype
    self.last_sample_time = self.dtype(-Ts)
    self.time_source = time_source

  def check(self, t: Optional[float] = None) -> bool:
    """
    Call at each simulation step.
    Returns True if a new sample occurs at time t.
    """
    if t is None:
      if self.time_source is None:
        raise ValueError(
            "Time 't' must be provided if 'time_source' is not set.")
      t = self.time_source()

    if t - self.last_sample_time >= self.Ts - np.finfo(self.dtype).eps:
      self.last_sample_time = t
      return True
    return False

  def reset(self):
    self.last_sample_time = -self.Ts


# -----------------------------------------------
# Controllers
# -----------------------------------------------


class GainScheduledController:
  """
  Gain-Scheduled Controller for LPV systems.

  Attributes
  ----------
  K : dict
      Dictionary mapping vertex tuples to gain matrices.
      Example: {(0, 0): K00, (0, 1): K01, (1, 0): K10, (1, 1): K11}
  rho_bounds : list of tuples
      Bounds for each scheduling parameter [(rho_min_1, rho_max_1), ..., (rho_min_n, rho_max_n)].
  """

  def __init__(self, K, rho_bounds):
    """
    Initialize the controller.

    Parameters
    ----------
    K : dict
        Gain matrices for each vertex of the LPV polytope.
    rho_bounds : list of tuples
        Lower and upper bounds for each scheduling parameter.
    """
    self.K = K
    self.rho_bounds = rho_bounds

  def compute_alphas(self, rho_hat):
    """
    Compute interpolation weights (alphas) for each scheduling parameter.

    Parameters
    ----------
    rho_hat : array-like
        Current estimated scheduling parameters.

    Returns
    -------
    alphas : list of float
        Normalized interpolation coefficients in [0, 1].
    """
    alphas = []
    for i, rho in enumerate(rho_hat):
      rho_min, rho_max = self.rho_bounds[i]
      alpha = (rho_max - rho) / (rho_max - rho_min)
      alpha = np.clip(alpha, 0.0, 1.0)
      alphas.append(alpha)
    return alphas

  def compute(self, x_hat, rho_hat):
    """
    Compute the control signal using convex combination of vertex gains.

    Parameters
    ----------
    x_hat : ndarray
        Estimated state vector (n_x, 1).
    rho_hat : array-like
        Current scheduling parameters.

    Returns
    -------
    u_total : ndarray
        Control signal (n_u, 1).
    """
    alphas = self.compute_alphas(rho_hat)
    vertex_keys = list(self.K.keys())
    n_u = self.K[vertex_keys[0]].shape[0]

    u_total = np.zeros((n_u, 1))

    # Convex combination over all vertices
    for vertex in itertools.product([0, 1], repeat=len(alphas)):
      weight = 1.0
      for i, v in enumerate(vertex):
        weight *= alphas[i] if v == 0 else (1.0 - alphas[i])

      u_total += weight * (self.K[vertex] @ x_hat)

    return u_total
