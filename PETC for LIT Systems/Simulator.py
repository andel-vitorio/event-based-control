# -*- coding: utf-8 -*-

from scipy.signal import cont2discrete
import numpy as np
import traceback
from numba import njit
from Utils import Numeric as nm
from Utils import DynamicSystem as ds


def _setup_simulation(config):
  """
  Prepara o ambiente de simulação: carrega planta e calcula tempo.
  Retorna objetos e parâmetros prontos para uso.
  """
  plant = ds.StateSpace(data=config["plant"], name='plant')
  duration = float(config['duration'])
  dt = float(config.get('dt', 1e-4))
  n_steps = int(np.ceil(duration / dt)) + 1
  time_history = np.linspace(0.0, duration, n_steps, dtype=np.float64)

  return plant, n_steps, dt, time_history

# ==============================================================================
# OPEN LOOP SIMULATOR
# ==============================================================================


@njit(fastmath=True, cache=True)
def _kernel_open_loop(n_steps, dt, x0, u_constant, A, B, C, D):
  """
  Loop de simulação compilado (Sem dependências de objetos Python).
  """
  nx = x0.shape[0]
  ny = C.shape[0]
  nu = B.shape[1]

  # 1. Alocação
  y_hist = np.zeros((ny, n_steps), dtype=np.float64)
  x_curr = x0.astype(np.float64).flatten()
  u_vec = np.full(nu, u_constant, dtype=np.float64)

  # 2. Loop Crítico
  for k in range(n_steps):
    # Saída
    y_hist[:, k] = C @ x_curr + D @ u_vec

    # Evolução (RK5 do Numeric.py)
    x_curr = nm.rk5_numba(A, B, x_curr, u_vec, dt)

  return y_hist


def open_loop(x0, config, u_constant=0.0):
  """
  Simulação Open Loop Otimizada.
  """
  try:
    # 1. Setup (Direto no simulator, sem classes extras)
    plant, n_steps, dt, time_history = _setup_simulation(config)

    # 2. Extração de Dados JIT (Da planta para arrays brutos)
    A, B, C, D = plant.export_matrices_jit()

    # Prepara estado inicial
    x0_arr = np.array(x0, dtype=np.float64).flatten()

    # 3. Execução do Kernel
    y_hist = _kernel_open_loop(
        n_steps,
        dt,
        x0_arr,
        float(u_constant),
        A, B, C, D
    )

    # 4. Retorno
    # Cria histórico de controle (constante) apenas para consistência de formato
    u_hist = np.full((1, n_steps), u_constant)

    return (y_hist, time_history, u_hist)

  except Exception as e:
    tb_str = traceback.format_exc()
    return (f"Erro Crítico: {e}\n{tb_str}", None, None)


# ==============================================================================
# CLOSED LOOP SIMULATOR - STATIC ETM - STATE SPACE MODEL
# ==============================================================================

@njit(fastmath=True, cache=True)
def _kernel_closed_loop_setm(
    n_steps, dt, x0,
    A, B, C, D,
    K, Ξ, Ψ, h
):
  """
  Kernel SETM sem saturação.
  """
  nx = x0.shape[0]
  ny = C.shape[0]
  nu = B.shape[1]

  y_hist = np.zeros((ny, n_steps), dtype=np.float64)
  x_hist = np.zeros((nx, n_steps), dtype=np.float64)
  u_hist = np.zeros((nu, n_steps), dtype=np.float64)
  event_idx = np.zeros(n_steps, dtype=np.int8)

  x_curr = x0.astype(np.float64).flatten()
  xm = x_curr.copy()
  x_hat = x_curr.copy()
  u_applied = np.zeros(nu, dtype=np.float64)

  steps_per_sample = int(round(h / dt))

  for k in range(n_steps):
    if k % steps_per_sample == 0:
      xm = x_curr.copy()

      ε = x_hat - xm
      # Ψ = 0.1 * Ξ
      should_trigger = (xm.T @ Ψ @ xm - ε.T @ Ξ @ ε < 0) or (k == 0)

      if should_trigger:
        event_idx[k] = 1
        x_hat = xm.copy()
        u_applied = K @ x_hat

    x_hist[:, k] = x_curr
    u_hist[:, k] = u_applied
    y_hist[:, k] = C @ x_curr + D @ u_applied

    x_curr = nm.rk5_numba(A, B, x_curr, u_applied, dt)

  return y_hist, x_hist, u_hist, event_idx


def closed_loop_setm(x0, config, results):
  if results is None:
    raise ValueError("Dicionário 'results' está Nulo.")

  plant, n_steps, dt, time_history = _setup_simulation(config)

  A, B, C, D = plant.export_matrices_jit()

  K = np.ascontiguousarray(results['controller']['K'], dtype=np.float64)
  Ξ = np.ascontiguousarray(results['etm']['Ξ'], dtype=np.float64)
  Ψ = np.ascontiguousarray(results['etm']['Ψ'], dtype=np.float64)
  h_param = float(config["design_params"]['h'])

  x0_arr = np.array(x0, dtype=np.float64).flatten()

  y_hist, _, u_hist, event_idx = _kernel_closed_loop_setm(
      n_steps, dt, x0_arr,
      A, B, C, D,
      K, Ξ, Ψ, h_param
  )

  event_times = time_history[event_idx == 1]

  return (y_hist, time_history, u_hist, event_times)


# ==============================================================================
# 3. MODELO DE RECORRÊNCIA - SETM
# ==============================================================================

@njit(fastmath=True, cache=True)
def _search_next_event(x_tk, Ad, Bd, K, Xi, Psi, max_lookahead):
  """
  Resolve o problema de minimização:
  nu = min { v in {h, 2h...} | x'Q(v)x < 0 }

  Ao invés de montar as matrizes Q(v) e A(v) explicitamente (caro),
  propagamos um estado virtual x_virt para avaliar a condição.
  """
  x_mh = x_tk.copy()
  u_hold = K @ x_tk

  for m in range(1, max_lookahead):
    x_mh = Ad @ x_mh + Bd @ u_hold
    e = x_tk - x_mh
    # Psi = 0.1 * Xi
    term_x = x_mh @ (Psi @ x_mh)
    term_e = e @ (Xi @ e)
    if (term_x - term_e) < 0:
      return m, x_mh

  return max_lookahead, x_mh


@njit(fastmath=True, cache=True)
def _kernel_recurrence_map(
    total_duration_steps,  # Horizonte total em passos h
    x0,
    Ad, Bd,
    K, Xi, Psi
):
  """
  Simula o sistema saltando de evento em evento (Mapa de Recorrência).
  x_{k+1} = A(nu(x_k)) * x_k
  """
  event_indices = np.zeros(total_duration_steps, dtype=np.int32)
  event_count = 0

  current_step_idx = 0
  event_indices[event_count] = 0
  event_count += 1

  x_tk = x0.astype(np.float64).flatten()

  while current_step_idx < total_duration_steps:
    remaining_steps = total_duration_steps - current_step_idx

    if remaining_steps <= 0:
      break

    m_star, x_next_event = _search_next_event(
        x_tk, Ad, Bd, K, Xi, Psi,
        max_lookahead=remaining_steps
    )

    current_step_idx += m_star

    if current_step_idx < total_duration_steps:
      event_indices[event_count] = current_step_idx
      event_count += 1

    x_tk = x_next_event

  return event_indices[:event_count]


def recurrence_model(x0, config, results):
  if results is None:
    raise ValueError("Resultados de otimização necessários.")

  plant = ds.StateSpace(data=config["plant"], name='plant')
  A_c, B_c, _, _ = plant.export_matrices_jit()

  h = float(config["design_params"]['h'])
  duration = float(config['duration'])

  sys_d = cont2discrete(
      (A_c, B_c, np.zeros((1, plant.nx)), np.zeros((1, plant.nu))),
      h, method='zoh')
  Ad = np.ascontiguousarray(sys_d[0], dtype=np.float64)
  Bd = np.ascontiguousarray(sys_d[1], dtype=np.float64)

  K = np.ascontiguousarray(results['controller']['K'], dtype=np.float64)
  Xi = np.ascontiguousarray(results['etm']['Ξ'], dtype=np.float64)
  Psi = np.ascontiguousarray(results['etm']['Ψ'], dtype=np.float64)

  x0_arr = np.array(x0, dtype=np.float64).flatten()

  total_steps = int(np.ceil(duration / h)) + 1

  event_indices = _kernel_recurrence_map(
      total_steps, x0_arr,
      Ad, Bd,
      K, Xi, Psi
  )

  return event_indices.astype(np.float64) * h
