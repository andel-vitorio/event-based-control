from typing import Optional, Dict, Any
import itertools
import cvxpy as cp
import numpy as np
import Numeric as nm
from DynamicSystem import *


def get_iet(event_time):
  if len(event_time) == 0:
    inter_event_time = np.array([])
  else:
    inter_event_time = np.empty_like(event_time)
    inter_event_time[0] = event_time[0]
    inter_event_time[1:] = event_time[1:] - event_time[:-1]

  return inter_event_time


def detm_synthesis(
    plant_data: Dict[str, Any],
    design_params: Dict[str, float],
    eps: float = 1e-6,
    dtype=np.float32
):
  """
  Solve the LMI-based networked control problem (full formulation).

  Parameters
  ----------
  plant_data : dict
      Plant information:
      - nx, nu, nρ, nw
      - ρ_bounds : list of tuples
      - u_bar : list
      - matrices_func : callable(ρ_values) -> dict with 'A','B','E'
      - get_C : callable(ρ_values) -> np.ndarray
  design_params : dict
      Design parameters: h, υ, δ, ε_γ, optionally λ, θ
  eps : float
      Small positive regularization
  dtype : numeric type
      Dtype for all arrays (default float32)

  Returns
  -------
  design_results : dict or None
  """
  nx = plant_data['nx']
  nu = plant_data['nu']
  nρ = plant_data['nρ']
  nw = plant_data['nw']
  ρ_bounds = plant_data['ρ_bounds']
  u_bar = np.array(plant_data['u_bar'], dtype=dtype)
  matrices_func = plant_data['matrices_func']
  nz = plant_data['nz']

  h = design_params['h']
  υ = design_params['υ']
  δ_val = design_params['δ']
  ε_γ = design_params['ε_γ']
  λ_val = design_params.get('λ', 0.0)
  θ_val = design_params.get('θ', 1.0)
  θtil_val = 1.0 / θ_val

  Bnp = list(itertools.product([0, 1], repeat=nρ))
  Onx = np.zeros((nx, nx), dtype=dtype)

  # Plant matrices
  A, B, C, E = {}, {}, {}, {}
  for i in Bnp:
    ρi = [ρ_bounds[idx][i[idx]] for idx in range(nρ)]
    mats = matrices_func(ρi)
    A[i] = cp.Parameter((nx, nx), value=mats['A'].astype(dtype))
    B[i] = cp.Parameter((nx, nu), value=mats['B'].astype(dtype))
    C[i] = cp.Parameter((nz, nx), value=mats['Cz'].astype(dtype))
    E[i] = cp.Parameter((nx, nw), value=mats['E'].astype(dtype))

  # e vectors
  e = nm.get_e(5*[nx] + [1, nu, nw])
  for i in range(1, len(e)):
    e[i] = cp.Parameter(e[i].shape, value=e[i].astype(dtype))

  # CVXPY variables
  Ptil = cp.Variable((nx, nx), PSD=True)
  Mtil = cp.Variable((2*nx + nu, 2*nx + nu), PSD=True)
  Q1til = cp.Variable((nx, nx), symmetric=True)
  Q2til = cp.Variable((nx, nx))
  Q3til = cp.Variable((nx, nu))
  Q4til = cp.Variable((nx, nx))
  S1til = cp.Variable((nx, nx), symmetric=True)
  S2til = cp.Variable((nx, nx))
  S3til = cp.Variable((nx, nu))
  S4til = cp.Variable((nx, nx))
  Rtil = cp.Variable((nx, nx), PSD=True)
  Ξtil = cp.Variable((nx, nx), PSD=True)
  Ψtil = cp.Variable((nx, nx), PSD=True)
  λ = cp.Parameter(value=λ_val)
  γ = cp.Variable(pos=True)
  δ = cp.Parameter(value=δ_val)
  β = cp.Variable(pos=True)
  θtil = cp.Parameter(value=θtil_val)
  X = cp.Variable((nx, nx))
  Ytil = cp.Variable((2*nx, e[1].shape[1]))

  Ktil, L1til, L2til, ℵ = {}, {}, {}, {}
  for i in Bnp:
    Ktil[i] = cp.Variable((nu, nx))
    L1til[i] = cp.Variable((nu, nx))
    L2til[i] = cp.Variable((nu, nx))
    ℵ[i] = cp.Variable((nu, nu), diag=True)

  Rcal = cp.bmat([[Rtil, Onx], [Onx, 3*Rtil]])
  Fscr = e[1].T + υ*e[2].T + υ*e[4].T
  κ1 = cp.bmat([[e[2]], [e[7]], [e[5]]])
  κ2 = cp.bmat([[e[1]-e[2]], [e[1]+e[2]-2*e[3]]])

  # Constraints
  constraints = []
  for i in Bnp:
    constraints += [ℵ[i] >> eps*np.eye(nu, dtype=dtype)]

  def get_Λ(i, j):
    Bscr = A[i] @ X @ e[1] + B[i] @ Ktil[j] @ e[2] - \
        X @ e[4] + B[i] @ Ktil[j] @ e[5] - B[i] @ ℵ[j] @ e[7] + γ * E[i] @ e[8]

    Θ1 = (e[1] - e[2]).T @ S1til @ (e[1] - e[2]) + \
        nm.He((e[1] - e[2]).T @ (S2til @ e[2] + S3til @ e[7] + S4til @ e[5]))

    Θ2 = nm.He(e[3].T @ (Q2til @ e[2] + Q3til @ e[7] + Q4til @ e[5]))

    Θ3 = nm.He(e[1].T @ (Q1til @ e[3] + Q2til @ e[2] +
                         Q3til @ e[7] + Q4til @ e[5]))

    Θ4 = nm.He(e[4].T @ (S1til @ (e[1] - e[2]) + S2til @
               e[2] + S3til @ e[7] + S4til @ e[5])) + \
        e[4].T @ Rtil @ e[4]

    Θ5 = e[7].T @ (ℵ[j] @ e[7] - L1til[j] @ e[2] - L2til[j] @ e[5])

    Θtil = {}
    Θtil['0'] = -λ * e[6].T @ e[6] - e[5].T @ Ξtil @ e[5] - Θ1 - \
        h * e[3].T @ Q1til @ e[3] + h * Θ3 - γ * e[8].T @ e[8] + \
        h * κ1.T @ Mtil @ κ1 + h * e[4].T @ Rtil @ e[4] - Θ5 + \
        nm.He(e[1].T @ Ptil @ e[4] + Fscr @ Bscr - κ2.T @ Ytil) + \
        h * Θ4

    Θtil['h'] = - λ * e[6].T @ e[6] - e[5].T @ Ξtil @ e[5] - Θ1 - \
        h * Θ2 - h * e[3].T @ Q1til @ e[3] - γ * e[8].T @ e[8] - \
        h * κ1.T @ Mtil @ κ1 - Θ5 + \
        nm.He(e[1].T @ Ptil @ e[4] + Fscr @ Bscr - κ2.T @ Ytil)

    Γ1_11 = Θtil['0']
    Γ1_12 = e[2].T @ X.T
    Γ1_13 = e[1].T @ X.T @ C[i].T

    Γ1_21 = Γ1_12.T
    Γ1_22 = - Ψtil
    Γ1_23 = np.zeros((nx, nz))

    Γ1_31 = Γ1_13.T
    Γ1_32 = Γ1_23.T
    Γ1_33 = - np.eye(nz)

    Γ1 = cp.bmat([[Γ1_11, Γ1_12, Γ1_13],
                  [Γ1_21, Γ1_22, Γ1_23],
                  [Γ1_31, Γ1_32, Γ1_33]])

    Γ2_11 = Θtil['h']
    Γ2_12 = Ytil.T
    Γ2_13 = e[2].T @ X.T
    Γ2_14 = e[1].T @ X.T @ C[i].T

    Γ2_21 = Γ2_12.T
    Γ2_22 = - (1. / h) * Rcal
    Γ2_23 = np.zeros((2 * nx, nx))
    Γ2_24 = np.zeros((2 * nx, nz))

    Γ2_31 = Γ2_13.T
    Γ2_32 = Γ2_23.T
    Γ2_33 = - Ψtil
    Γ2_34 = np.zeros((nx, nz))

    Γ2_41 = Γ2_14.T
    Γ2_42 = Γ2_24.T
    Γ2_43 = Γ2_34.T
    Γ2_44 = - np.eye(nz)

    Γ2 = cp.bmat([[Γ2_11, Γ2_12, Γ2_13, Γ2_14],
                  [Γ2_21, Γ2_22, Γ2_23, Γ2_24],
                  [Γ2_31, Γ2_32, Γ2_33, Γ2_34],
                  [Γ2_41, Γ2_42, Γ2_43, Γ2_44]])

    return Γ1, Γ2

  # Binary pair LMIs
  for pairs in nm.binary_pairs(nρ):
    LMI_SUM = {'0': 0., 'h': 0.}
    for pair in pairs:
      Λ0, Λh = get_Λ(pair[0], pair[1])
      LMI_SUM['0'] += Λ0
      LMI_SUM['h'] += Λh
    constraints += [LMI_SUM['0'] << -eps *
                    np.eye(LMI_SUM['0'].shape[0], dtype=dtype)]
    constraints += [LMI_SUM['h'] << -eps *
                    np.eye(LMI_SUM['h'].shape[0], dtype=dtype)]

  # Saturation LMIs
  for ell in range(nu):
    for j in Bnp:
      LMISAT11 = Ptil
      LMISAT12 = np.zeros((nx, nx), dtype=dtype)
      LMISAT13 = np.zeros((nx, 1), dtype=dtype)
      LMISAT14 = (Ktil[j][ell:ell+1]-L1til[j][ell:ell+1]).T
      LMISAT15 = (Ktil[j][ell:ell+1]-L1til[j][ell:ell+1]).T
      LMISAT16 = X.T
      LMISAT21 = LMISAT12.T
      LMISAT22 = Ξtil
      LMISAT23 = np.zeros((nx, 1), dtype=dtype)
      LMISAT24 = (Ktil[j][ell:ell+1]-L2til[j][ell:ell+1]).T
      LMISAT25 = (Ktil[j][ell:ell+1]-L2til[j][ell:ell+1]).T
      LMISAT26 = np.zeros((nx, nx), dtype=dtype)
      LMISAT31 = LMISAT13.T
      LMISAT32 = LMISAT23.T
      LMISAT33 = (1-θtil)*np.eye(1, dtype=dtype)
      LMISAT34 = np.zeros((1, 1), dtype=dtype)
      LMISAT35 = np.zeros((1, 1), dtype=dtype)
      LMISAT36 = np.zeros((1, nx), dtype=dtype)
      LMISAT41 = LMISAT14.T
      LMISAT42 = LMISAT24.T
      LMISAT43 = LMISAT34.T
      LMISAT44 = (u_bar[ell]**2)*np.eye(1, dtype=dtype)
      LMISAT45 = np.zeros((1, 1), dtype=dtype)
      LMISAT46 = np.zeros((1, nx), dtype=dtype)
      LMISAT51 = LMISAT15.T
      LMISAT52 = LMISAT25.T
      LMISAT53 = LMISAT35.T
      LMISAT54 = np.zeros((1, 1), dtype=dtype)
      LMISAT55 = γ*δ*(u_bar[ell]**2)*np.eye(1, dtype=dtype)
      LMISAT56 = np.zeros((1, nx), dtype=dtype)
      LMISAT61 = LMISAT16.T
      LMISAT62 = LMISAT26.T
      LMISAT63 = LMISAT36.T
      LMISAT64 = LMISAT46.T
      LMISAT65 = LMISAT56.T
      LMISAT66 = Ψtil

      LMISAT = cp.bmat(
          [[LMISAT11, LMISAT12, LMISAT13, LMISAT14, LMISAT15, LMISAT16],
           [LMISAT21, LMISAT22, LMISAT23, LMISAT24, LMISAT25, LMISAT26],
           [LMISAT31, LMISAT32, LMISAT33, LMISAT34, LMISAT35, LMISAT36],
           [LMISAT41, LMISAT42, LMISAT43, LMISAT44, LMISAT45, LMISAT46],
           [LMISAT51, LMISAT52, LMISAT53, LMISAT54, LMISAT55, LMISAT56],
           [LMISAT61, LMISAT62, LMISAT63, LMISAT64, LMISAT65, LMISAT66]])

      constraints += [LMISAT >> 0]

  # Incremental LMID0
  LMID0_11 = β * np.eye(nx)
  LMID0_12 = np.eye(nx)
  LMID0_21 = np.eye(nx)
  LMID0_22 = X + X.T - Ptil

  LMID0 = cp.bmat([[LMID0_11, LMID0_12],
                   [LMID0_21, LMID0_22]])

  constraints += [LMID0 >> 0]

  # Positive definite constraints
  constraints += [Ψtil >> eps*np.eye(nx, dtype=dtype)]
  constraints += [Ξtil >> eps*np.eye(nx, dtype=dtype)]
  constraints += [γ >= ε_γ]

  # Objective
  obj = cp.Minimize(cp.trace(Ξtil+Ψtil)+β)
  prob = cp.Problem(obj, constraints)
  prob.solve(solver=cp.MOSEK, verbose=False, ignore_dpp=True)

  # Results
  design_results = None
  if prob.status not in ["infeasible", "unbounded"]:
    Xinv = np.linalg.inv(X.value.astype(dtype))
    Ξ = Xinv.T @ Ξtil.value.astype(dtype) @ Xinv
    Ψ = np.linalg.inv(Ψtil.value.astype(dtype))
    P = Xinv.T @ Ptil.value.astype(dtype) @ Xinv
    S2 = Xinv.T @ S2til.value.astype(dtype) @ Xinv

    K = {i: Ktil[i].value.astype(dtype)@Xinv for i in Bnp}
    L1 = {i: L1til[i].value.astype(dtype)@Xinv for i in Bnp}
    L2 = {i: L2til[i].value.astype(dtype)@Xinv for i in Bnp}

    etm_results = {'Ξ': Ξ, 'Ψ': Ψ, 'θ': θ_val, 'λ': λ_val}
    design_results = {
        'optimal_value': prob.value,
        'etm': etm_results,
        'controller': {'K': K, 'L1': L1, 'L2': L2},
        'lyapunov': [P, S2],
        'bounds': [γ.value, β.value]
    }
  return design_results


class DETM(System):
  def __init__(self, Ξ: np.ndarray, Ψ: np.ndarray, λ: float, θ: float, name: str = 'DETM'):
    super().__init__(name=name, dynamics=None, output_func=None)
    self.Ξ = Ξ
    self.Ψ = Ψ
    self.λ = λ
    self.θ = θ

    self.η0 = None
    self.x_hat = None
    self.xm = None

    self.dynamics = self._dynamics
    self.output_func = self._output_func

  # --- Property: η é apenas uma interface para states ---
  @property
  def η(self):
    return self.states

  @η.setter
  def η(self, value):
    self.states = value

  # ------------------------------------------------------

  def set_η0(self, η0: float):
    self.η0 = np.array([[η0]])
    self.η = self.η0.copy()  # seta states também

  def reset(self):
    self.η = self.η0.copy()  # seta states também

  def triggering_condition(self) -> bool:
    return self.η[0][0] + self.θ * self._triggering_func() < 0

  def _triggering_func(self) -> float:
    x = self.xm
    ε = self.x_hat - x
    return (x.T @ self.Ψ @ x - ε.T @ self.Ξ @ ε)[0][0]

  def _dynamics(self, t: float, η: np.ndarray, inputs, params=None):
    dη = - self.λ * η + self._triggering_func()
    return dη

  def _output_func(self, t: float, η: np.ndarray, inputs: dict):
    pass
