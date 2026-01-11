from optimization import DisturbedSaturatedPETC as DSPETC
from Utils import Signal as sgn
from Utils import Numeric as nm
from Utils import DynamicSystem as ds
import numpy as np
import traceback
import functools


def run_simulation(x0, config, results):
  """
  Executa uma simulação completa para o estado inicial 'x0'.
  """

  try:
    plant = ds.StateSpace(data=config["plant"], name='plant')
    ncs = ds.NetworkedControlSystem()
    ncs.add_system(plant)

    if results is None:
      return (f"Erro: Dicionário 'results' está Nulo.", None, None, None, None)

    Ξ, Ψ = results['etm']['Ξ'], results['etm']['Ψ']
    θ, λ = results['etm']['θ'], results['etm']['λ']
    K = results['controller']['K']
    P = results['lyapunov'][0]

    design_params = config["design_params"]["dspetc"]
    u_bar = plant.get_input_bounds()
    sampler = ds.Sampler(Ts=design_params['h'])
    detm = DSPETC.DETM(Ξ=Ξ, Ψ=Ψ, λ=λ, θ=θ)
    ncs.add_system(detm)
    ρ_bounds = plant.get_parameter_bounds()
    controller = ds.GainScheduledController(K, ρ_bounds)

    x0 = np.array(x0, dtype=np.float32).reshape(-1, 1)

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

    signal_control = np.squeeze(np.array(signal_control))
    if signal_control.ndim == 1:
      signal_control = signal_control[np.newaxis, :]
    else:
      signal_control = signal_control.T
    event_time = np.array(event_time)
    ncs.finalize_history()
    plant_output = ncs.output_history['plant']
    eta = ncs.output_history['DETM']
    time_history = ncs.time_history

    return (plant_output, time_history, signal_control, eta, event_time)

  except Exception as e:
    tb_str = traceback.format_exc()
    return (f"Erro INESPERADO: {e}\n{tb_str}", None, None, None, None)
