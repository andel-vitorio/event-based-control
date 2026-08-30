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
    eps: float = 1e-6,
    solver: str = cp.MOSEK,
    verbose: bool = False,
    dtype: np.dtype = np.float64,
) -> Optional[Dict[str, Any]]:
  """Synthesize controller gain K and decoupled ETM matrices (SC and CA).

  Formulation based on looped-functional, Finsler's lemma, and triangular simplex
  relaxation in the transformed coordinates z(t) = exp(lambda*t) * T * x(t).
  """
  # -------------------------------------------------------------------------
  # 1. Parameter Extraction & System Setup
  # -------------------------------------------------------------------------
  A = np.asarray(design_params["A"], dtype=dtype)
  B = np.asarray(design_params["B"], dtype=dtype)

  nx = A.shape[0]
  nu = B.shape[1]

  h = float(design_params["h"])
  lam = float(design_params.get("λ", design_params.get("lambda", 0.1)))

  upsilon1 = float(
      design_params.get("υ1", design_params.get("upsilon1", 1.0))
  )
  upsilon2 = float(
      design_params.get("υ2", design_params.get("upsilon2", 1.0))
  )
  upsilon3 = float(
      design_params.get("υ3", design_params.get("upsilon3", 1.0))
  )
  upsilon4 = float(
      design_params.get("υ3", design_params.get("upsilon3", 1.0))
  )

  tol_zero = 1e-8
  if abs(upsilon1) < tol_zero:
    raise ValueError(
        "O parâmetro upsilon1 (υ1) não pode ser zero para garantir a injeção do ganho de controle K."
    )
  if abs(upsilon2) < tol_zero:
    raise ValueError(
        "O parâmetro upsilon2 (υ2) não pode ser zero para garantir o acoplamento no subespaço delta_ca."
    )
  if upsilon3 <= tol_zero:
    raise ValueError(
        "O parâmetro upsilon3 (υ3) deve ser estritamente positivo (upsilon3 > 0) "
        "para contrapor o termo +h*R no bloco (4,4) associado a z_dot."
    )
  if abs(upsilon4) < tol_zero:
    raise ValueError(
        "O parâmetro upsilon2 (υ2) não pode ser zero para garantir o acoplamento no subespaço delta_ca."
    )

  # Construction/extraction of coordinate transformation matrix T = [C; T2]
  if "T" in design_params:
    T = np.asarray(design_params["T"], dtype=dtype)
    if "ny" in design_params:
      ny = int(design_params["ny"])
    elif "C" in design_params:
      ny = np.asarray(design_params["C"], dtype=dtype).shape[0]
    else:
      raise ValueError(
          "When providing 'T', specify 'ny' or 'C' explicitly.")
  else:
    if "C" not in design_params:
      raise ValueError(
          "Either 'T' or 'C' must be provided in design_params.")
    C = np.asarray(design_params["C"], dtype=dtype)
    ny = C.shape[0]
    nx2 = nx - ny

    if "T2" in design_params or "T_2" in design_params:
      T2 = np.asarray(
          design_params.get("T2", design_params.get("T_2")), dtype=dtype
      )
    else:
      T2 = np.hstack([np.zeros((nx2, ny), dtype=dtype),
                     np.eye(nx2, dtype=dtype)])

    T = np.vstack([C, T2])

  if T.shape != (nx, nx):
    raise ValueError(
        f"Matrix T must have shape ({nx}, {nx}), got {T.shape}.")

  try:
    T_inv = np.linalg.inv(T)
  except np.linalg.LinAlgError:
    raise ValueError("Transformation matrix T is singular.")

  # Transformed nominal matrices
  A_bar = T @ A @ T_inv
  B_bar = T @ B
  A_lambda = A_bar + lam * np.eye(nx, dtype=dtype)
  exp_lam_h = float(np.exp(lam * h))

  # -------------------------------------------------------------------------
  # 2. Projection Operators for Extended Vector zeta_z,m(tau) (dim = 5*nx + ny)
  #
  # zeta = [z_m(tau); z_m(0); chi_m(tau); z_dot_m(tau); delta_m^ca(0); delta_1,m^sc(0)]
  # -------------------------------------------------------------------------
  dim_zeta = 5 * nx + ny
  block_dims = [nx, nx, nx, nx, nx, ny]
  e = {}
  offset = 0

  for i, b_dim in enumerate(block_dims, start=1):
    ei = np.zeros((b_dim, dim_zeta), dtype=dtype)
    ei[:, offset: offset + b_dim] = np.eye(b_dim, dtype=dtype)
    e[i] = ei
    offset += b_dim

  # Output selection from initial state: z_{1,m}(0) = J1 @ e_2 @ zeta
  J1 = np.hstack(
      [np.eye(ny, dtype=dtype), np.zeros((ny, nx - ny), dtype=dtype)])

  # Augmented vectors: varpi(0) = kappa1 @ zeta, Pi(tau) = kappa2 @ zeta
  κ1 = np.vstack([e[2], e[5]])
  κ2 = np.vstack([e[1] - e[2], e[1] + e[2] - 2.0 * e[3]])
  e12 = e[1] - e[2]

  # -------------------------------------------------------------------------
  # 3. Decision Variables
  # -------------------------------------------------------------------------
  Ptil = cp.Variable((nx, nx), symmetric=True)
  Rtil = cp.Variable((nx, nx), symmetric=True)

  S1til = cp.Variable((nx, nx), symmetric=True)
  S2til = cp.Variable((nx, nx))
  S3til = cp.Variable((nx, nx))

  Q1til = cp.Variable((nx, nx), symmetric=True)
  Q2til = cp.Variable((nx, nx))
  Q3til = cp.Variable((nx, nx))

  Mtil = cp.Variable((2 * nx, 2 * nx), symmetric=True)

  # Decoupled ETM decision variables:
  Ψsc_til = cp.Variable((ny, ny), symmetric=True)
  Ξsc_til = cp.Variable((ny, ny), symmetric=True)

  Ψca_til = cp.Variable((nx, nx), symmetric=True)
  Ξca_til = cp.Variable((nx, nx), symmetric=True)

  Ytil = cp.Variable((2 * nx, dim_zeta))
  Ktil = cp.Variable((nu, nx))
  X = cp.Variable((nx, nx))

  # -------------------------------------------------------------------------
  # 4. Auxiliary Expressions & Finsler Multiplier
  # -------------------------------------------------------------------------
  def He(expr):
    return expr + expr.T

  zero_nx = np.zeros((nx, nx), dtype=dtype)
  Rcal = cp.bmat([
      [Rtil, zero_nx],
      [zero_nx, 3.0 * Rtil],
  ])

  Ftil = e[1].T + upsilon1 * e[2].T + \
      upsilon2 * e[4].T + upsilon4 * e[4].T + e[6].T

  Λ1 = (
      e12.T @ S1til @ e12
      + He(e12.T @ (S2til @ e[2] + S3til @ e[5]))
  )

  Λ2 = (
      κ1.T @ Mtil @ κ1
      + e[4].T @ Rtil @ e[4]
      - e[3].T @ Q1til @ e[3]
      + He(
          e12.T @ S1til @ e[4]
          + e[4].T @ (S2til @ e[2] + S3til @ e[5])
          + e[1].T @ (Q1til @ e[3] + Q2til @ e[2] + Q3til @ e[5])
      )
  )

  Λ3 = (
      e[3].T @ Q1til @ e[3]
      + κ1.T @ Mtil @ κ1
      + He(e[3].T @ (Q2til @ e[2] + Q3til @ e[5]))
  )

  # Base affine theta components
  theta0_bar = (
      He(e[1].T @ Ptil @ e[4])
      - e[6].T @ Ξsc_til @ e[6]
      - e[5].T @ Ξca_til @ e[5]
      + h * Λ2
      - Λ1
      - He(Ytil.T @ κ2)
      + He(Ftil @ (A_lambda @ X @ e[1] - X @ e[4]))
  )

  theta1_bar = -Λ2 - Λ3
  theta2 = He(Ftil @ B_bar @ Ktil @ (e[2] + e[5]))

  # Vertex matrices (affine components)
  Theta_bar_1 = theta0_bar + theta2
  Theta_bar_2 = theta0_bar + h * theta1_bar + theta2
  Theta_bar_3 = theta0_bar + h * theta1_bar + exp_lam_h * theta2

  # -------------------------------------------------------------------------
  # 5. LMIs at Simplex Vertices (via Schur Complement)
  # -------------------------------------------------------------------------
  Γ_Psi_sc = e[2].T @ X.T @ J1.T  # (5nx+ny) x ny
  Γ_Psi_ca = e[2].T               # (5nx+ny) x nx

  zero_ny_nx = np.zeros((ny, nx), dtype=dtype)
  zero_2nx_ny = np.zeros((2 * nx, ny), dtype=dtype)
  zero_2nx_nx = np.zeros((2 * nx, nx), dtype=dtype)

  Γ1 = cp.bmat([
      [Theta_bar_1, Γ_Psi_sc, Γ_Psi_ca],
      [Γ_Psi_sc.T, -Ψsc_til, zero_ny_nx],
      [Γ_Psi_ca.T, zero_ny_nx.T, -Ψca_til],
  ])

  Γ2 = cp.bmat([
      [Theta_bar_2, Ytil.T, Γ_Psi_sc, Γ_Psi_ca],
      [Ytil, -(1.0 / h) * Rcal, zero_2nx_ny, zero_2nx_nx],
      [Γ_Psi_sc.T, zero_2nx_ny.T, -Ψsc_til, zero_ny_nx],
      [Γ_Psi_ca.T, zero_2nx_nx.T, zero_ny_nx.T, -Ψca_til],
  ])

  Γ3 = cp.bmat([
      [Theta_bar_3, Ytil.T, Γ_Psi_sc, Γ_Psi_ca],
      [Ytil, -(1.0 / h) * Rcal, zero_2nx_ny, zero_2nx_nx],
      [Γ_Psi_sc.T, zero_2nx_ny.T, -Ψsc_til, zero_ny_nx],
      [Γ_Psi_ca.T, zero_2nx_nx.T, zero_ny_nx.T, -Ψca_til],
  ])

  # -------------------------------------------------------------------------
  # 6. Constraints & Optimization
  # -------------------------------------------------------------------------
  I_nx = np.eye(nx, dtype=dtype)
  I_ny = np.eye(ny, dtype=dtype)

  constraints = [
      Ptil >> eps * I_nx,
      Rtil >> eps * I_nx,

      Ξsc_til >> eps * I_ny,
      Ψsc_til >> eps * I_ny,

      Ξca_til >> eps * I_nx,
      Ψca_til >> eps * I_nx,

      Γ1 << -eps * np.eye(Γ1.shape[0], dtype=dtype),
      Γ2 << -eps * np.eye(Γ2.shape[0], dtype=dtype),
      Γ3 << -eps * np.eye(Γ3.shape[0], dtype=dtype),
  ]

  objective = cp.Minimize(cp.trace(Ψca_til + Ξca_til + Ψsc_til + Ξsc_til))
  problem = cp.Problem(objective, constraints)

  # -------------------------------------------------------------------------
  # 7. Solve
  # -------------------------------------------------------------------------
  try:
    problem.solve(solver=solver, verbose=verbose)
  except Exception as err:
    if verbose:
      print(f"[Solver Error] Solver execution failed: {err}")
    return None

  if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    if verbose:
      print(f"[Solver Status] Infeasible or Unbounded: {problem.status}")
    return None

  # -------------------------------------------------------------------------
  # 8. Variable Reconstruction
  # -------------------------------------------------------------------------
  X_val = np.asarray(X.value, dtype=dtype)

  try:
    X_inv = np.linalg.inv(X_val)
    Psisc_til_val = np.asarray(Ψsc_til.value, dtype=dtype)
    Psi_sc_rec = np.linalg.inv(Psisc_til_val)

    Psica_til_val = np.asarray(Ψca_til.value, dtype=dtype)
    Psi_ca_rec = np.linalg.inv(Psica_til_val)
  except np.linalg.LinAlgError:
    if verbose:
      print("[Recovery Error] Singular slack matrix X or Psi_til detected.")
    return None

  Ktil_val = np.asarray(Ktil.value, dtype=dtype)
  Xisc_til_val = np.asarray(Ξsc_til.value, dtype=dtype)
  Xica_til_val = np.asarray(Ξca_til.value, dtype=dtype)

  Ptil_val = np.asarray(Ptil.value, dtype=dtype)
  Rtil_val = np.asarray(Rtil.value, dtype=dtype)

  # Reconstruction formulas:
  # K = K_tilde * X^{-1} * T
  # Psi_sc = Psisc_til^{-1},  Xi_sc = Xisc_til
  # Psi_ca = Psica_til^{-1},  Xi_ca = X^{-T} * Xica_til * X^{-1}
  K_rec = Ktil_val @ X_inv @ T
  Xi_sc_rec = Xisc_til_val
  Xi_ca_rec = X_inv.T @ Xica_til_val @ X_inv

  P_rec = X_inv.T @ Ptil_val @ X_inv
  R_rec = X_inv.T @ Rtil_val @ X_inv

  return {
      "solver_status": problem.status,
      "optimal_value": problem.value,
      "controller": {
          "K": K_rec,
          "Ktil": Ktil_val,
      },
      "etm": {
          "sc": {
              "Psi": Psi_sc_rec,
              "Xi": Xi_sc_rec,
              "Psitil": Psisc_til_val,
              "Xitil": Xisc_til_val,
          },
          "ca": {
              "Psi": Psi_ca_rec,
              "Xi": Xi_ca_rec,
              "Psitil": Psica_til_val,
              "Xitil": Xica_til_val,
          },
      },
      "functional": {
          "P": P_rec,
          "R": R_rec,
      },
      "slacks": {
          "X": X_val,
      },
      "transformed": {
          "Ptil": Ptil_val,
          "Rtil": Rtil_val,
          "A_bar": A_bar,
          "B_bar": B_bar,
          "T": T,
      },
  }


def synthesize_impulsive_observer(
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
