from optimization import DisturbedSaturatedPETC as DSPETC
from Utils import Signal as sgn
from Utils import Numeric as nm
from Utils import DynamicSystem as ds
import numpy as np
import traceback
import functools


def closed_loop(x0, config, results):
  """
  Executa uma simulação completa para o estado inicial 'x0'.
  """

  try:
    plant = ds.StateSpace(data=config["plant"], name='plant')
    sim = ds.SimulationEngine()
    sim.add_system(plant)

    if results is None:
      return (f"Erro: Dicionário 'results' está Nulo.", None, None, None, None)

    Xi, Psi = results['etm']['\u039e'], results['etm']['\u03a8']
    theta, lambda_ = results['etm']['\u03b8'], results['etm']['\u03bb']
    K = results['controller']['K']
    P = results['lyapunov'][0]

    design_params = config["design_params"]["dspetc"]
    u_bar = plant.get_input_bounds()
    sampler = ds.Sampler(Ts=design_params['h'], time_source=lambda: sim.t)

    detm_kwargs = {'\u039e': Xi, '\u03a8': Psi,
                   '\u03bb': lambda_, '\u03b8': theta}
    detm = DSPETC.DETM(**detm_kwargs)
    sim.add_system(detm)
    rho_bounds = plant.get_parameter_bounds()
    controller = ds.GainScheduledController(K, rho_bounds)

    x0 = np.array(x0, dtype=np.float32).reshape(-1, 1)

    sim.get_system('plant').set_initial_state(x0)
    detm.set_η0(0.0)
    detm.x_hat = x0.copy()
    detm.xm = x0.copy()
    sim.setup_clock(duration=20.0, dt=1e-4)
    sim.reset_clock()
    sampler.reset()

    event_time = []
    uc = [[0.0]]
    signal_control = []

    while sim.advance_clock():
      u_sat = sgn.sat(uc, u_bar)
      signal_control += [u_sat]
      default_inputs = {'plant': {'u1': u_sat}}

      with sim.step(default_inputs=default_inputs) as step:
        transmit = False
        if sampler.check():
          detm.xm = plant.states.copy()
          transmit = detm.triggering_condition() or (sim.t == 0.0)
        if not transmit:
          continue
        rho_hat = plant.evaluate_parameters(sim.t)
        detm.x_hat = detm.xm.copy()
        event_time += [sim.t]
        uc = controller.compute(detm.x_hat, rho_hat)

    signal_control = np.squeeze(np.array(signal_control))
    if signal_control.ndim == 1:
      signal_control = signal_control[np.newaxis, :]
    else:
      signal_control = signal_control.T
    event_time = np.array(event_time)
    sim.finalize_history()
    plant_output = sim.output_history['plant']
    eta = sim.output_history['DETM']
    time_history = sim.time_history

    return (plant_output, time_history, signal_control, eta, event_time)

  except Exception as e:
    tb_str = traceback.format_exc()
    return (f"Erro INESPERADO: {e}\n{tb_str}", None, None, None, None)


def open_loop(x0, config, u_constant=0.0):
  """
  Executa uma simulação em malha aberta (Open Loop) para o estado inicial 'x0'.

  Args:
      x0: Estado inicial.
      config: Dicionário de configuração contendo os dados da planta.
      u_constant: Valor constante de entrada (padrão 0.0 para resposta livre).
  """

  try:
    # 1. Configuração da Planta e Motor de Simulação
    plant = ds.StateSpace(data=config["plant"], name='plant')
    sim = ds.SimulationEngine()
    sim.add_system(plant)

    # 2. Configuração do Estado Inicial
    x0 = np.array(x0, dtype=np.float32).reshape(-1, 1)
    sim.get_system('plant').set_initial_state(x0)

    # 3. Configuração do Relógio (Mesma duração da malha fechada)
    sim.setup_clock(duration=20.0, dt=1e-4)
    sim.reset_clock()

    signal_control = []

    # 4. Loop de Simulação
    while sim.advance_clock():
      # Em malha aberta, a entrada é fixa ou pré-determinada,
      # não dependendo do estado atual (sem feedback).
      u_applied = u_constant

      # Armazena para histórico
      signal_control.append([u_applied])

      default_inputs = {'plant': {'u1': u_applied}}

      with sim.step(default_inputs=default_inputs) as step:
        pass  # Nenhuma lógica de gatilho ou cálculo de controle aqui

    # 5. Processamento dos Resultados
    sim.finalize_history()
    plant_output = sim.output_history['plant']
    time_history = sim.time_history

    # Formatação do sinal de controle para manter consistência com numpy
    signal_control = np.array(signal_control).T

    # Retorna a tupla no mesmo formato da run_simulation original.
    # Eta e event_time são retornados como None pois não existem em malha aberta.
    return (plant_output, time_history, signal_control, None, None)

  except Exception as e:
    tb_str = traceback.format_exc()
    return (f"Erro INESPERADO: {e}\n{tb_str}", None, None, None, None)
