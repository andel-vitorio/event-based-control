# -*- coding: utf-8 -*-

import numpy as np
import traceback
from numba import njit
from Utils import Numeric as nm
from Utils import DynamicSystem as ds

# ==============================================================================
# KERNEL JIT (Lógica de Execução)
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

# ==============================================================================
# HELPER DE SETUP (Substitui o SimulationContext/Engine)
# ==============================================================================


def _setup_simulation(config):
  """
  Prepara o ambiente de simulação: carrega planta e calcula tempo.
  Retorna objetos e parâmetros prontos para uso.
  """
  # 1. Carrega Planta
  plant = ds.StateSpace(data=config["plant"], name='plant')

  # 2. Configura Tempo
  duration = float(config['duration'])
  # Default seguro se não houver no config
  dt = float(config.get('dt', 1e-4))

  # Cálculo seguro de passos (arredondamento para cima)
  n_steps = int(np.ceil(duration / dt)) + 1

  # Cria vetor de tempo para retorno
  time_history = np.linspace(0.0, duration, n_steps, dtype=np.float64)

  return plant, n_steps, dt, time_history


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

  # Alocação
  y_hist = np.zeros((ny, n_steps), dtype=np.float64)
  x_hist = np.zeros((nx, n_steps), dtype=np.float64)
  u_hist = np.zeros((nu, n_steps), dtype=np.float64)
  event_idx = np.zeros(n_steps, dtype=np.int8)

  # Inicialização
  x_curr = x0.astype(np.float64).flatten()
  xm = x_curr.copy()
  x_hat = x_curr.copy()
  u_applied = np.zeros(nu, dtype=np.float64)

  steps_per_sample = int(round(h / dt))

  # Loop Principal
  for k in range(n_steps):

    # A. Amostrador
    if k % steps_per_sample == 0:
      xm = x_curr.copy()
      # print('t = ', k * dt)

      ε = x_hat - xm
      should_trigger = (xm.T @ Ψ @ xm - ε.T @ Ξ @ ε < 0) or (k == 0)
      # print(xm.T @ Ψ @ xm - ε.T @ Ξ @ ε)

      if should_trigger:
        event_idx[k] = 1
        x_hat = xm.copy()
        u_applied = K @ x_hat

    # Armazenamento
    x_hist[:, k] = x_curr
    u_hist[:, k] = u_applied
    y_hist[:, k] = C @ x_curr + D @ u_applied

    # Evolução
    x_curr = nm.rk5_numba(A, B, x_curr, u_applied, dt)

  return y_hist, x_hist, u_hist, event_idx


def closed_loop_setm(x0, config, results):
  if results is None:
    raise ValueError("Dicionário 'results' está Nulo.")

  # 1. Setup
  plant, n_steps, dt, time_history = _setup_simulation(config)

  # 2. Extração JIT (Sem bounds u_min/u_max)
  A, B, C, D = plant.export_matrices_jit()

  # 3. Preparação SETM
  K = np.ascontiguousarray(results['controller']['K'], dtype=np.float64)
  Ξ = np.ascontiguousarray(results['etm']['Ξ'], dtype=np.float64)
  Ψ = np.ascontiguousarray(results['etm']['Ψ'], dtype=np.float64)
  h_param = float(config["design_params"]['h'])

  x0_arr = np.array(x0, dtype=np.float64).flatten()

  # 4. Execução
  y_hist, x_hist, u_hist, event_idx = _kernel_closed_loop_setm(
      n_steps, dt, x0_arr,
      A, B, C, D,
      K, Ξ, Ψ, h_param
  )

  # 5. Retorno
  event_times = time_history[event_idx == 1]

  return (y_hist, time_history, u_hist, event_times)
