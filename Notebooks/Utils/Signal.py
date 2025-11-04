import numpy as np
from typing import Sequence
from numpy.typing import NDArray


def step_signal(timepts: NDArray[np.float64],
                instants: Sequence[float],
                amplitudes: Sequence[float]) -> NDArray[np.float64]:
  """
  Generates a step signal based on specified transition times and amplitudes.

  The signal starts at zero and changes value at each specified time instant, 
  taking on the corresponding amplitude.

  Parameters
  ----------
  timepts : NDArray[np.float64]
      One-dimensional array containing the time points at which the signal
      is to be evaluated.
  instants : Sequence[float]
      A list or sequence of time instants at which step changes occur.
  amplitudes : Sequence[float]
      A list or sequence of amplitudes that the signal should take on after
      each corresponding instant. Must have the same length as `instants`.

  Returns
  -------
  signal : NDArray[np.float64]
      An array of the same shape as `timepts`, representing the step signal 
      with transitions defined by `instants` and `amplitudes`.

  Notes
  -----
  - If a step time `T` is greater than or equal to the last element in `timepts`,
    it will be ignored.
  - The function uses `np.heaviside` with value 1 at `x = 0`.
  """
  signal = np.zeros(timepts.shape, dtype=np.float64)
  last_amplitude = 0.
  for T, A in zip(instants, amplitudes):
    if T >= timepts[-1]:
      continue
    signal += (A - last_amplitude) * np.heaviside(timepts - T, 1)
    last_amplitude = A
  return signal


def sat(u, u_bar):
  """
  Applies saturation to a control signal, supporting scalar, vector, or time-varying signals.

  Parameters
  ----------
  u : float or ndarray
      Control signal to be saturated. Can be:
      - float: single control value (nu = 1)
      - ndarray of shape (nu, 1): control vector
      - ndarray of shape (nu, timepts): time-varying control signal
  u_bar : float or list or ndarray
      Saturation limits. Can be:
      - float: single limit applied to all elements
      - list or ndarray of shape (nu,): per-element limits

  Returns
  -------
  ndarray or float
      Saturated control signal with the same shape as input `u`.
  """
  u = np.asarray(u, dtype=float)

  # --- Case 1: scalar input ---
  if np.isscalar(u) or u.ndim == 0:
    return np.sign(u) * min(abs(u), u_bar)

  # --- Case 2: vector input (nu, 1) ---
  elif u.ndim == 2 and u.shape[1] == 1:
    u_bar = np.asarray(u_bar, dtype=float).flatten()
    if u_bar.size == 1:
      u_bar = np.full(u.shape[0], u_bar.item())

    u_sat = np.sign(u) * np.minimum(np.abs(u.flatten()), u_bar)
    return u_sat.reshape(u.shape)

  # --- Case 3: time-varying signal (nu, timepts) ---
  elif u.ndim == 2:
    u_bar = np.asarray(u_bar, dtype=float).flatten()
    if u_bar.size == 1:
      u_bar = np.full(u.shape[0], u_bar.item())

    # Broadcasting to apply per-channel saturation across time
    u_sat = np.sign(u) * np.minimum(np.abs(u), u_bar[:, np.newaxis])
    return u_sat

  else:
    raise ValueError(
        "Unsupported input shape for 'u'. Expected scalar, (nu, 1), or (nu, timepts).")
