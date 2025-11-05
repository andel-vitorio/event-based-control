import DisturbedSaturatedPETC as DSPETC
import Signal as sgn
import Numeric as nm
import DynamicSystem as ds
import os
import sys
import json
import pickle
import numpy as np
import traceback

# --- 1. Configuração de Path (da célula [2]) ---
base_path = os.path.abspath("..")
dirs_to_add = ["Utils", "Optimization Problems", "DisturbedSaturatedPETC"]
for d in dirs_to_add:
  path = os.path.join(base_path, d)
  if path not in sys.path:
    sys.path.append(path)

# --- 2. Imports Essenciais (da célula [2]) ---

# --- A função ainda recebe config e results ---


def run_simulation(idx, config, results):
  """
  Executa uma simulação completa para o 'idx', usando os dados de 
  'config' e 'results' pré-carregados pelo processo pai.
  """

  try:
    # --- 3. Setup da Planta (Usa 'config' do argumento) ---
    plant = ds.StateSpace(data=config["plant"], name='plant')

    # --- 4. Setup do NCS ---
    ncs = ds.NetworkedControlSystem()
    ncs.add_system(plant)

    # --- 5. Carregamento dos Parâmetros (Usa 'results' do argumento) ---
    if results is None:
      return (idx, f"Erro: Dicionário 'results' está Nulo.", None, None, None)

    Ξ, Ψ = results['etm']['Ξ'], results['etm']['Ψ']
    θ, λ = results['etm']['θ'], results['etm']['λ']
    K = results['controller']['K']
    P = results['lyapunov'][0]

    design_params = config["design_params"]["dspetc"]

    # --- CORREÇÃO (Linha 55) ---
    # u_bar não vem de config["plant"], mas do objeto plant
    u_bar = plant.get_input_bounds()[0]
    if not isinstance(u_bar, np.ndarray):
      u_bar = np.array(u_bar)  # Garante que é um array numpy

    # --- 6. Setup do Sampler ---
    sampler = ds.Sampler(Ts=design_params['h'])

    # --- 7. Setup do DETM ---
    detm = DSPETC.DETM(Ξ=Ξ, Ψ=Ψ, λ=λ, θ=θ)
    ncs.add_system(detm)

    # --- 8. Setup do Controlador ---
    # --- CORREÇÃO (Linha 65) ---
    # ρ_bounds também vem do objeto plant
    ρ_bounds = plant.get_parameter_bounds()
    if not isinstance(ρ_bounds, np.ndarray):
      ρ_bounds = np.array(ρ_bounds)  # Garante que é um array numpy

    controller = ds.GainScheduledController(K, ρ_bounds)

    # --- 9. Setup do Estado Inicial (usando 'P' e 'idx') ---
    X0_list = nm.ellipsoid_boundary_points(P, 1, 20)
    x0 = []
    for i in range(plant.nx):
      x0 += [[X0_list[i][idx]]]
    x0 = np.array(x0, dtype=np.float32)

    # --- 10. Lógica da Simulação ---
    ncs.get_system('plant').set_initial_state(x0)
    detm.set_η0(0.0)
    detm.x_hat = x0.copy()
    detm.xm = x0.copy()
    ncs.setup_clock(duration=20.0, dt=1e-4)
    ncs.reset_clock()
    sampler.reset()

    event_time = []
    uc = [[0.0]]
    signal_control = []

    while ncs.advance_clock():
      u_sat = sgn.sat(uc, u_bar)
      signal_control += [u_sat]
      default_inputs = {'plant': {'u1': u_sat}}

      with ncs.step(default_inputs=default_inputs) as step:
        transmit = False
        if sampler.check(ncs.t):
          detm.xm = plant.states.copy()
          transmit = detm.triggering_condition() or (ncs.t == 0.0)
        if not transmit:
          continue
        ρ_hat = plant.evaluate_parameters(ncs.t)
        detm.x_hat = detm.xm.copy()
        event_time += [ncs.t]
        uc = controller.compute(detm.x_hat, ρ_hat)

    # --- 11. Pós-processamento ---
    signal_control = np.squeeze(np.array(signal_control))
    if signal_control.ndim == 1:
      signal_control = signal_control[np.newaxis, :]
    else:
      signal_control = signal_control.T
    event_time = np.array(event_time)
    ncs.finalize_history()
    plant_output = ncs.output_history['plant']
    time_history = ncs.time_history

    return (idx, plant_output, time_history, signal_control, event_time)

  except Exception as e:
    tb_str = traceback.format_exc()
    return (idx, f"Erro INESPERADO no idx {idx}: {e}\n{tb_str}", None, None, None)
