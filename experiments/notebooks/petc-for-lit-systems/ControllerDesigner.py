from IPython.display import display, Markdown, Math
from scipy.linalg import expm
from typing import Dict, Any, Optional
import scipy.linalg as la
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import cvxpy as cp
import numpy as np
from Utils import Numeric as nm
from scipy.linalg import svd


# def synthesize_setm(
#     design_params: Dict[str, Any],
#     eps: float = 1e-6,
#     solver: str = cp.MOSEK,
#     verbose: bool = False,
#     dtype: np.dtype = np.float64,
# ) -> Optional[Dict[str, Any]]:
#   """Projeta o ganho do controlador K e as matrizes do SETM (Ξ, Ψ) via LMIs para um sistema Linear Invariante no Tempo (LIT).
#   """
#   # -------------------------------------------------------------------------
#   # 1. Extração de Parâmetros e Estruturas de Projeção
#   # -------------------------------------------------------------------------
#   A = np.asarray(design_params["A"], dtype=dtype)
#   B = np.asarray(design_params["B"], dtype=dtype)
#   h = float(design_params["h"])
#   upsilon = float(design_params["υ"])
#   lam = float(design_params["λ"])

#   nx = A.shape[0]
#   nu = B.shape[1]

#   # Projeções em bloco
#   e = nm.get_e(5 * [nx])
#   e_shape_1 = e[1].shape[1]

#   # -------------------------------------------------------------------------
#   # 2. Variáveis de Decisão
#   # -------------------------------------------------------------------------
#   Ptil = cp.Variable((nx, nx), PSD=True)
#   Mtil = cp.Variable((2 * nx, 2 * nx), PSD=True)
#   Q1til = cp.Variable((nx, nx), symmetric=True)
#   Q2til = cp.Variable((nx, nx))
#   Q3til = cp.Variable((nx, nx))
#   S1til = cp.Variable((nx, nx), symmetric=True)
#   S2til = cp.Variable((nx, nx))
#   S3til = cp.Variable((nx, nx))
#   Ktil = cp.Variable((nu, nx))
#   Rtil = cp.Variable((nx, nx), PSD=True)
#   Ξtil = cp.Variable((nx, nx), PSD=True)
#   Ψtil = cp.Variable((nx, nx), PSD=True)
#   X = cp.Variable((nx, nx))
#   Ytil = cp.Variable((2 * nx, e_shape_1))

#   # -------------------------------------------------------------------------
#   # 3. Expressões e Blocos Intermediários
#   # -------------------------------------------------------------------------
#   exp_lam_h = np.exp(-2.0 * lam * h)
#   exp_pos_lam_h = np.exp(2.0 * lam * h)
#   Onx = np.zeros((nx, nx), dtype=dtype)

#   Rcal = cp.bmat([[Rtil, Onx], [Onx, 3.0 * Rtil]])
#   Fscr = e[1].T + upsilon * e[2].T + upsilon * e[4].T
#   κ1 = cp.bmat([[e[2]], [e[5]]])
#   κ2 = cp.bmat([[e[1] - e[2]], [e[1] + e[2] - 2.0 * e[3]]])

#   Λ1 = (e[1] - e[2]).T @ S1til @ (e[1] - e[2]) + nm.He(
#       (e[1] - e[2]).T @ (S2til @ e[2] + S3til @ e[5])
#   )

#   Λ2 = (
#       κ1.T @ Mtil @ κ1
#       + e[4].T @ Rtil @ e[4]
#       - e[3].T @ Q1til @ e[3]
#       + nm.He(
#           (e[1] - e[2]).T @ S1til @ e[4]
#           + e[4].T @ S2til @ e[2]
#           + e[4].T @ S3til @ e[5]
#           + e[1].T @ Q1til @ e[3]
#           + e[1].T @ Q2til @ e[2]
#           + e[1].T @ Q3til @ e[5]
#       )
#   )

#   Λ3 = (
#       e[3].T @ Q1til @ e[3]
#       + κ1.T @ Mtil @ κ1
#       + nm.He(e[3].T @ (Q2til @ e[2] + Q3til @ e[5]))
#   )

#   Bscr = A @ X @ e[1] + B @ Ktil @ e[2] + B @ Ktil @ e[5] - X @ e[4]

#   theta0 = (
#       -e[5].T @ Ξtil @ e[5]
#       - 2.0 * lam * exp_lam_h * Λ1
#       + (1.0 - exp_lam_h) * Λ2
#       + nm.He(
#           Fscr @ Bscr
#           + e[1].T @ Ptil @ e[4]
#           + lam * e[1].T @ Ptil @ e[1]
#           - 2.0 * lam * exp_lam_h * κ2.T @ Ytil
#       )
#   )

#   thetah = (
#       -e[5].T @ Ξtil @ e[5]
#       - 2.0 * lam * exp_lam_h * (Λ1 + h * Λ3 + nm.He(κ2.T @ Ytil))
#       + nm.He(Fscr @ Bscr + e[1].T @ Ptil @
#               e[4] + lam * e[1].T @ Ptil @ e[1])
#   )

#   # -------------------------------------------------------------------------
#   # 4. Construção das LMIs
#   # -------------------------------------------------------------------------
#   Γ1 = cp.bmat([[theta0, e[2].T @ X.T],
#                 [X @ e[2], -Ψtil]])

#   Γ2 = cp.bmat(
#       [
#           [thetah, Ytil.T, e[2].T @ X.T],
#           [
#               Ytil,
#               -(exp_pos_lam_h / (2.0 * lam * h)) * Rcal,
#               np.zeros((2 * nx, nx), dtype=dtype),
#           ],
#           [X @ e[2], np.zeros((nx, 2 * nx), dtype=dtype), -Ψtil],
#       ]
#   )

#   constraints = [
#       Γ1 << -eps * np.eye(Γ1.shape[0], dtype=dtype),
#       Γ2 << -eps * np.eye(Γ2.shape[0], dtype=dtype),
#       Ψtil >> eps * np.eye(nx, dtype=dtype),
#       Ξtil >> eps * np.eye(nx, dtype=dtype),
#   ]

#   # -------------------------------------------------------------------------
#   # 5. Resolução do Problema de Otimização
#   # -------------------------------------------------------------------------
#   objective = cp.Minimize(cp.trace(Ξtil + Ψtil))
#   problem = cp.Problem(objective, constraints)

#   try:
#     problem.solve(solver=solver, verbose=verbose)
#   except (cp.SolverError, cp.DCPError, Exception) as e:
#     if verbose:
#       print(f"[Solver Error] Falha na execução do solver: {e}")
#     return None

#   if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
#     if verbose:
#       print(
#           f"[Solver Status] Problema não resolvido. Status: {problem.status}")
#     return None

#   # -------------------------------------------------------------------------
#   # 6. Reconstrução das Variáveis Originais (Congruência com X)
#   # -------------------------------------------------------------------------
#   X_val = np.asarray(X.value, dtype=dtype)
#   Xinv = np.linalg.inv(X_val)
#   XinvT = Xinv.T

#   Xinv_2x = np.block([[Xinv, Onx], [Onx, Xinv]])
#   XinvT_2x = Xinv_2x.T

#   def _rec_sym(V: cp.Variable) -> np.ndarray:
#     return XinvT @ np.asarray(V.value, dtype=dtype) @ Xinv

#   return {
#       "optimal_value": problem.value,
#       "solver_status": problem.status,
#       "condition_number_X": float(np.linalg.cond(X_val)),
#       "etm": {
#           "Ξ": _rec_sym(Ξtil),
#           "Ψ": np.linalg.inv(np.asarray(Ψtil.value, dtype=dtype)),
#       },
#       "controller": {"K": np.asarray(Ktil.value, dtype=dtype) @ Xinv},
#       "functional": {
#           "P": _rec_sym(Ptil),
#           "R": _rec_sym(Rtil),
#           "M": XinvT_2x @ np.asarray(Mtil.value, dtype=dtype) @ Xinv_2x,
#           "S1": _rec_sym(S1til),
#           "S2": _rec_sym(S2til),
#           "S3": _rec_sym(S3til),
#           "Q1": _rec_sym(Q1til),
#           "Q2": _rec_sym(Q2til),
#           "Q3": _rec_sym(Q3til),
#           "Y": XinvT_2x @ np.asarray(Ytil.value, dtype=dtype),
#       },
#   }

def synthesize_output_based_setm(
    design_params: Dict[str, Any],
    eps: float = 1e-5,
    solver: str = cp.MOSEK,
    verbose: bool = False,
    dtype: np.dtype = np.float64,
) -> Optional[Dict[str, Any]]:
  """Sintetiza o ganho do controlador K e as matrizes do SETM (Ξ, Ψ)

  via relaxação politópica em simplex triangular (Teorema 2).
  """
  # -------------------------------------------------------------------------
  # 1. Extração de Parâmetros
  # -------------------------------------------------------------------------
  A = np.asarray(design_params["A"], dtype=dtype)
  B = np.asarray(design_params["B"], dtype=dtype)
  C = np.asarray(design_params["C"], dtype=dtype)
  h = float(design_params["h"])
  lam = float(design_params.get("λ", design_params.get("lambda", 0.1)))

  upsilon1 = design_params.get("υ1")
  upsilon2 = design_params.get("υ2")

  nx = A.shape[0]
  nu = B.shape[1]
  ny = C.shape[0]

  # Dimensão total do vetor aumentado: 5*nx + ny
  dim_zeta = 5 * nx + ny

  # Projeções em bloco: e[1]..e[4] (nx), e[5] (ny), e[6] (nx)
  # e[i] possui dimensão (n_i, dim_zeta)
  e = {}
  offsets = [0, nx, 2 * nx, 3 * nx, 4 * nx, 4 * nx + ny, dim_zeta]
  dims = [nx, nx, nx, nx, ny, nx]
  for i in range(1, 7):
    mat = np.zeros((dims[i - 1], dim_zeta), dtype=dtype)
    mat[:, offsets[i - 1]: offsets[i]] = np.eye(dims[i - 1], dtype=dtype)
    e[i] = mat

  # -------------------------------------------------------------------------
  # 2. Variáveis de Decisão
  # -------------------------------------------------------------------------
  # Matrizes de Lyapunov e ETM (Restritas ao cone PSD e Simétricas)
  Ptil = cp.Variable((nx, nx), PSD=True)
  Rtil = cp.Variable((nx, nx), PSD=True)
  S1til = cp.Variable((nx, nx), symmetric=True)
  Q1til = cp.Variable((nx, nx), symmetric=True)
  Mtil = cp.Variable((2 * nx + ny, 2 * nx + ny), symmetric=True)
  Ξtil = cp.Variable((ny, ny), PSD=True)
  Ψtil = cp.Variable((ny, ny), PSD=True)
  Sδtil = cp.Variable((nx, nx), PSD=True)

  # Multiplicadores de folga e laço (Genéricos)
  S2til = cp.Variable((nx, nx))
  S3til = cp.Variable((nx, ny))
  S4til = cp.Variable((nx, nx))
  Q2til = cp.Variable((nx, nx))
  Q3til = cp.Variable((nx, ny))
  Q4til = cp.Variable((nx, nx))
  Ytil = cp.Variable((2 * nx, dim_zeta))
  Ktil = cp.Variable((nu, nx))

  # REMOVIDA A FLAG PSD=True. X1 e X2 são matrizes genéricas!
  X1 = cp.Variable((nx, nx))
  X2 = cp.Variable((ny, ny))

  # -------------------------------------------------------------------------
  # 3. Construção dos Operadores Algébricos
  # -------------------------------------------------------------------------
  exp_lam_h = float(np.exp(lam * h))
  Onx = np.zeros((nx, nx), dtype=dtype)
  O_ny_2nx = np.zeros((ny, 2 * nx), dtype=dtype)

  Rcal = cp.bmat([[Rtil, Onx], [Onx, 3.0 * Rtil]])
  κ1 = np.vstack([e[2], e[5], e[6]])
  κ2 = np.vstack([e[1] - e[2], e[1] + e[2] - 2.0 * e[3]])

  def He(expr):
    return expr + expr.T

  # Componentes Λ1, Λ2, Λ3
  Λ1 = (e[1] - e[2]).T @ S1til @ (e[1] - e[2]) + He(
      (e[1] - e[2]).T @ (S2til @ e[2] + S3til @ e[5] + S4til @ e[6])
  )

  Λ2 = (
      κ1.T @ Mtil @ κ1
      + e[4].T @ Rtil @ e[4]
      - e[3].T @ Q1til @ e[3]
      + He(
          (e[1] - e[2]).T @ S1til @ e[4]
          + e[4].T @ (S2til @ e[2] + S3til @ e[5] + S4til @ e[6])
          + e[1].T
          @ (Q1til @ e[3] + Q2til @ e[2] + Q3til @ e[5] + Q4til @ e[6])
      )
  )

  Λ3 = (
      e[3].T @ Q1til @ e[3]
      + κ1.T @ Mtil @ κ1
      + He(e[3].T @ (Q2til @ e[2] + Q3til @ e[5] + Q4til @ e[6]))
  )

  A_lam = A + lam * np.eye(nx, dtype=dtype)

  Fx = e[1].T + upsilon1 * e[2].T + upsilon2 * e[4].T
  Fy = e[5].T
  Bzy = X2 @ e[5] - C @ X1 @ e[6]

  # Decomposição Afim: Θ_0_bar, Θ_1_bar e Θ_2
  theta0_bar = (
      -e[5].T @ Ξtil @ e[5]
      + h * Λ2
      - (Λ1 + He(κ2.T @ Ytil))
      + He(e[1].T @ Ptil @ e[4] +
           Fx @ (A_lam @ X1 @ e[1] - X1 @ e[4]) +
           Fy @ Bzy)
  )
  theta1_bar = -Λ2 - Λ3 - (1. / h) * e[6].T @ Sδtil @ e[6]
  theta2 = He(Fx @ B @ Ktil @ (e[2] + e[6]))

  Θbar_1 = theta0_bar + theta2
  Θbar_2 = theta0_bar + h * theta1_bar + theta2
  Θbar_3 = theta0_bar + h * theta1_bar + exp_lam_h * theta2

  # -------------------------------------------------------------------------
  # 4. Acoplamento de Schur nos 3 Vértices
  # -------------------------------------------------------------------------
  ΓeXC = e[2].T @ X1.T @ C.T  # shape: (dim_zeta, ny)

  Γ1 = cp.bmat([[Θbar_1, ΓeXC],
                [ΓeXC.T, -Ψtil]])

  Γ2 = cp.bmat(
      [
          [Θbar_2, h * Ytil.T, ΓeXC],
          [h * Ytil, - h * Rcal, np.zeros((2 * nx, ny))],
          [ΓeXC.T, np.zeros((ny, 2 * nx)), -Ψtil],
      ]
  )

  Γ3 = cp.bmat(
      [
          [Θbar_3, h * Ytil.T, ΓeXC],
          [h * Ytil, - h * Rcal, np.zeros((2 * nx, ny))],
          [ΓeXC.T, np.zeros((ny, 2 * nx)), -Ψtil],
      ]
  )

  # -------------------------------------------------------------------------
  # 5. Restrições e Normalização de Escala (Gauge Fixing)
  # -------------------------------------------------------------------------
  I_nx = np.eye(nx, dtype=dtype)
  I_ny = np.eye(ny, dtype=dtype)

  constraints = [
      Γ1 << -eps * np.eye(Γ1.shape[0], dtype=dtype),
      Γ2 << -eps * np.eye(Γ2.shape[0], dtype=dtype),
      Γ3 << -eps * np.eye(Γ3.shape[0], dtype=dtype),

      # Limitações operacionais do ETM
      Ξtil >> eps * I_ny,
      Ψtil >> eps * I_ny,
      Sδtil >> eps * I_ny
  ]

  # -------------------------------------------------------------------------
  # 6. Função Objetivo Focada em Desempenho
  # -------------------------------------------------------------------------
  # objective = cp.Minimize(cp.trace(Ξtil + Ψtil))
  objective = cp.Minimize(0.0)

  problem = cp.Problem(objective, constraints)

  try:
    problem.solve(solver=cp.MOSEK, verbose=verbose)
  except Exception as err:
    if verbose:
      print(f"[Solver Error] Falha na execução: {err}")
    return None

  if problem.status not in (cp.OPTIMAL):
    if verbose:
      print(f"[Solver Status] Não ótimo. Status: {problem.status}")
    return None

  # -------------------------------------------------------------------------
  # 7. Reconstrução das Variáveis Físicas
  # -------------------------------------------------------------------------
  X1_val = np.asarray(X1.value, dtype=dtype)
  X2_val = np.asarray(X2.value, dtype=dtype)

  X1inv = np.linalg.inv(X1_val)
  X2inv = np.linalg.inv(X2_val)
  X1invT = X1inv.T
  X2invT = X2inv.T

  Ψtil_val = np.asarray(Ψtil.value, dtype=dtype)
  Ξtil_val = np.asarray(Ξtil.value, dtype=dtype)
  Ktil_val = np.asarray(Ktil.value, dtype=dtype)

  K_rec = Ktil_val @ X1inv
  Xi_rec = X2invT @ Ξtil_val @ X2inv
  Psi_rec = np.linalg.inv(Ψtil_val)

  return {
      "solver_status": problem.status,
      "optimal_value": problem.value,
      "controller": {"K": K_rec},
      "etm": {
          "Ξ": Xi_rec,
          "Ψ": Psi_rec,
      },
      "functional": {
          "P": X1invT @ np.asarray(Ptil.value, dtype=dtype) @ X1inv,
          "R": X1invT @ np.asarray(Rtil.value, dtype=dtype) @ X1inv,
      },
      "slacks": {
          "X1": X1_val,
          "X2": X2_val,
      },
  }


# =============================================================================
# 1. Definições das Funções de Síntese
# =============================================================================

def synthesize_observer_clock_calibrated(
    design_params: Dict[str, Any],
    eps: float = 1e-6,
    solver: str = cp.MOSEK,
    verbose: bool = False,
    dtype: np.dtype = np.float64,
) -> Optional[Dict[str, Any]]:
  """Síntese via Funcional com Dependência do Relógio (Clock-Dependent)."""
  A = np.asarray(design_params["A"], dtype=dtype)
  C = np.asarray(design_params["C"], dtype=dtype)
  h = float(design_params["h"])
  nu_max = float(design_params.get(
      "nu_max", design_params.get("nu_bar", 1.0)))

  eig_max = np.max(np.real(np.linalg.eigvals(A)))
  alpha = float(design_params.get("alpha", max(0.0, eig_max + 0.1)))
  rho = float(design_params.get("rho", 0.85))

  nx, ny = A.shape[0], C.shape[0]
  I_nx, I_ny = np.eye(nx, dtype=dtype), np.eye(ny, dtype=dtype)

  P0 = cp.Variable((nx, nx), symmetric=True)
  P1 = cp.Variable((nx, nx), symmetric=True)
  X = cp.Variable((nx, nx))
  Y = cp.Variable((nx, ny))
  gamma_L = cp.Variable(nonneg=True)

  def build_psi(tau_val: float):
    P_tau = P0 + tau_val * P1
    return cp.bmat([
        [-rho * P_tau, (X - Y @ C).T],
        [X - Y @ C, P0 - (X + X.T)]
    ])

  P_numax = P0 + nu_max * P1
  flow_0 = P1 + A.T @ P0 + P0 @ A - 2.0 * alpha * P0
  flow_numax = P1 + A.T @ P_numax + P_numax @ A - 2.0 * alpha * P_numax

  schur_gamma = cp.bmat([
      [gamma_L * I_ny, Y.T],
      [Y, I_nx]
  ])

  constraints = [
      P0 - I_nx >> eps * I_nx,
      P_numax >> eps * I_nx,
      flow_0 << -eps * I_nx,
      flow_numax << -eps * I_nx,
      build_psi(h) << -eps * np.eye(2 * nx, dtype=dtype),
      build_psi(nu_max) << -eps * np.eye(2 * nx, dtype=dtype),
      schur_gamma >> 0,
      X + X.T >> eps * I_nx
  ]

  problem = cp.Problem(cp.Minimize(gamma_L), constraints)
  try:
    problem.solve(solver=solver, verbose=verbose)
  except Exception:
    return None

  if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    return None

  X_val = np.asarray(X.value, dtype=dtype)
  Y_val = np.asarray(Y.value, dtype=dtype)
  L_rec = np.linalg.inv(X_val) @ Y_val

  return {
      "L": L_rec,
      "gamma_L": float(gamma_L.value),
      "P0": np.asarray(P0.value, dtype=dtype),
      "P1": np.asarray(P1.value, dtype=dtype),
      "alpha": alpha,
      "rho": rho,
      "decay_factor": rho * np.exp(2.0 * alpha * nu_max)
  }


def synthesize_impulsive_observer_discrete_exact(
    design_params: Dict[str, Any],
    eps: float = 1e-6,
    solver: str = cp.MOSEK,
    verbose: bool = False,
    dtype: np.dtype = np.float64,
) -> Optional[Dict[str, Any]]:
  """Síntese Discreta Exata Multimodo via Operador de Monodromia."""
  A = np.asarray(design_params["A"], dtype=dtype)
  C = np.asarray(design_params["C"], dtype=dtype)
  h = float(design_params["h"])
  nu_bar_steps = int(round(float(design_params.get("nu_bar", 1.0)) / h))
  lambda_obs = float(design_params.get("lambda_obs", 0.1))

  nx, ny = A.shape[0], C.shape[0]
  P = cp.Variable((nx, nx), PSD=True)
  Y = cp.Variable((nx, ny))
  gamma_L = cp.Variable(nonneg=True)

  constraints = [P >> eps * np.eye(nx, dtype=dtype)]

  # Avaliação em todos os modos discretos tau in {h, 2h, ..., nu_bar*h}
  for l in range(1, nu_bar_steps + 1):
    tau = l * h
    Phi_tau = expm(A * tau)
    decay = np.exp(-2.0 * lambda_obs * tau)
    M_tau = P @ Phi_tau - Y @ (C @ Phi_tau)

    LMI_l = cp.bmat([
        [decay * P, M_tau.T],
        [M_tau, P]
    ])
    constraints.append(LMI_l >> eps * np.eye(2 * nx, dtype=dtype))

  schur_gamma = cp.bmat([
      [gamma_L * np.eye(ny, dtype=dtype), Y.T],
      [Y, P]
  ])
  constraints.append(schur_gamma >> 0)

  problem = cp.Problem(cp.Minimize(gamma_L), constraints)
  try:
    problem.solve(solver=solver, verbose=verbose)
  except Exception:
    return None

  if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    return None

  P_val = np.asarray(P.value, dtype=dtype)
  Y_val = np.asarray(Y.value, dtype=dtype)
  L_rec = np.linalg.inv(P_val) @ Y_val

  return {
      "L": L_rec,
      "gamma_L": float(gamma_L.value),
      "P": P_val,
      "Y": Y_val,
      "nu_bar_steps": nu_bar_steps
  }
