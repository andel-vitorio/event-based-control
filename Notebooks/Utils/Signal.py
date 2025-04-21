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


def sat(u: float, u_bar: float) -> float:
  """
  Applies saturation to a value u, limiting its absolute value to u_bar.

  Parameters
  ----------
  u : float
      The value to be saturated.
  u_bar : float
      The saturation threshold, limiting the absolute value of u.

  Returns
  -------
  float
      The saturated value, which is the sign of u multiplied by the minimum
      of u_bar and the absolute value of u.
  """
  return np.sign(u) * min(u_bar, abs(u))
