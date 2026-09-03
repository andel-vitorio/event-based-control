from scipy.linalg import expm, null_space
from typing import Any, Dict, Optional, Tuple
from typing import Any, Dict, Optional
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


def check_system_properties(A, B, C, tol=1e-9):
  n = A.shape[0]
  controllability = B.copy()
  for i in range(1, n):
    controllability = np.hstack(
        (controllability, np.linalg.matrix_power(A, i) @ B))
  rank_ctrb = np.linalg.matrix_rank(controllability, tol=tol)
  controllable = rank_ctrb == n

  observability = C.copy()
  for i in range(1, n):
    observability = np.vstack(
        (observability, C @ np.linalg.matrix_power(A, i)))
  rank_obsv = np.linalg.matrix_rank(observability, tol=tol)
  observable = rank_obsv == n

  eig_A = np.linalg.eigvals(A)
  stabilizable = True

  for eig in eig_A:
    if np.real(eig) >= -tol:
      pbh_matrix = np.hstack((eig * np.eye(n) - A, B))
      if np.linalg.matrix_rank(pbh_matrix, tol=tol) < n:
        stabilizable = False
        break

  detectable = True
  for eig in eig_A:
    if np.real(eig) >= -tol:
      pbh_matrix = np.vstack((eig * np.eye(n) - A, C))
      if np.linalg.matrix_rank(pbh_matrix, tol=tol) < n:
        detectable = False
        break

  return {
      "controllability_rank": rank_ctrb,
      "observability_rank": rank_obsv,
      "controllable": controllable,
      "observable": observable,
      "stabilizable": stabilizable,
      "detectable": detectable,
      "eigenvalues": eig_A,
  }


def synthesize_output_based_setm(
    design_params: Dict[str, Any],
    eps: float = 1e-6,
    solver: str = cp.MOSEK,
    verbose: bool = False,
    dtype: np.dtype = np.float64,
) -> Optional[Dict[str, Any]]:
  """Synthesize controller gain K and decoupled ETM matrices (SC and CA).
  """
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
        "para contrapor o termo +h*R no bloco associado a z_dot."
    )

  if "T" in design_params:
    T = np.asarray(design_params["T"], dtype=dtype)
    if "ny" in design_params:
      ny = int(design_params["ny"])
    elif "C" in design_params:
      ny = np.asarray(design_params["C"], dtype=dtype).shape[0]
    else:
      raise ValueError(
          "When providing 'T', specify 'ny' or 'C' explicitly."
      )
  else:
    if "C" not in design_params:
      raise ValueError(
          "Either 'T' or 'C' must be provided in design_params."
      )
    C = np.asarray(design_params["C"], dtype=dtype)
    ny = C.shape[0]
    nx2 = nx - ny

    if "T2" in design_params or "T_2" in design_params:
      T2 = np.asarray(
          design_params.get("T2", design_params.get("T_2")), dtype=dtype
      )
    else:
      T2 = np.hstack(
          [np.zeros((nx2, ny), dtype=dtype), np.eye(nx2, dtype=dtype)]
      )

    T = np.vstack([C, T2])

  if T.shape != (nx, nx):
    raise ValueError(
        f"Matrix T must have shape ({nx}, {nx}), got {T.shape}.")

  try:
    T_inv = np.linalg.inv(T)
  except np.linalg.LinAlgError:
    raise ValueError("Transformation matrix T is singular.")

  A_bar = T @ A @ T_inv
  B_bar = T @ B
  A_lambda = A_bar + lam * np.eye(nx, dtype=dtype)
  exp_lam_h = float(np.exp(lam * h))

  dim_zeta = 5 * nx + ny
  block_dims = [nx, nx, nx, nx, nx, ny]
  e = {}
  offset = 0

  for i, b_dim in enumerate(block_dims, start=1):
    ei = np.zeros((b_dim, dim_zeta), dtype=dtype)
    ei[:, offset: offset + b_dim] = np.eye(b_dim, dtype=dtype)
    e[i] = ei
    offset += b_dim

  # Output selector: J1 = [I_ny, 0] such that y = J1 @ z
  J1 = np.hstack(
      [np.eye(ny, dtype=dtype), np.zeros((ny, nx - ny), dtype=dtype)])

  κ1 = np.vstack([e[2], e[6], e[5]])  # dim: (2*nx + ny) x dim_zeta
  # dim: (2*nx) x dim_zeta
  κ2 = np.vstack([e[1] - e[2], e[1] + e[2] - 2.0 * e[3]])
  e12 = e[1] - e[2]

  Ptil = cp.Variable((nx, nx), symmetric=True)
  Rtil = cp.Variable((nx, nx), symmetric=True)

  S1til = cp.Variable((nx, nx), symmetric=True)
  S2til = cp.Variable((nx, nx))
  S3til = cp.Variable((nx, ny))  # Multiplica e[6] (delta_sc de dimensao ny)
  S4til = cp.Variable((nx, nx))  # Multiplica e[5] (delta_ca de dimensao nx)

  Q1til = cp.Variable((nx, nx), symmetric=True)
  Q2til = cp.Variable((nx, nx))
  Q3til = cp.Variable((nx, ny))  # Multiplica e[6] (delta_sc de dimensao ny)
  Q4til = cp.Variable((nx, nx))  # Multiplica e[5] (delta_ca de dimensao nx)

  Mtil = cp.Variable((2 * nx + ny, 2 * nx + ny), symmetric=True)

  # Decoupled ETM decision variables
  Ψsc_til = cp.Variable((ny, ny), symmetric=True)
  # X2 = I_ny -> variavel direta
  Ξsc = cp.Variable((ny, ny), symmetric=True)

  Ψca_til = cp.Variable((nx, nx), symmetric=True)
  Ξca_til = cp.Variable((nx, nx), symmetric=True)

  Ytil = cp.Variable((2 * nx, dim_zeta))
  Ktil = cp.Variable((nu, nx))
  X = cp.Variable((nx, nx))

  def He(expr):
    return expr + expr.T

  zero_nx = np.zeros((nx, nx), dtype=dtype)
  Rcal = cp.bmat([
      [Rtil, zero_nx],
      [zero_nx, 3.0 * Rtil],
  ])

  Ftil = (
      e[1].T
      + upsilon1 * e[2].T
      + upsilon2 * e[5].T
      + upsilon3 * e[4].T
      + e[6].T @ J1
  )

  S_affine = S2til @ e[2] + S3til @ e[6] + S4til @ e[5]
  Q_affine = Q2til @ e[2] + Q3til @ e[6] + Q4til @ e[5]

  Λ1 = (
      e12.T @ S1til @ e12
      + He(e12.T @ S_affine)
  )

  Λ2 = (
      κ1.T @ Mtil @ κ1
      + e[4].T @ Rtil @ e[4]
      - e[3].T @ Q1til @ e[3]
      + He(
          e12.T @ S1til @ e[4]
          + e[4].T @ S_affine
          + e[1].T @ (Q1til @ e[3] + Q_affine)
      )
  )

  Λ3 = (
      e[3].T @ Q1til @ e[3]
      + κ1.T @ Mtil @ κ1
      + He(e[3].T @ Q_affine)
  )

  theta0_bar = (
      He(e[1].T @ Ptil @ e[4])
      - e[6].T @ Ξsc @ e[6]
      - e[5].T @ Ξca_til @ e[5]
      + h * Λ2
      - Λ1
      - He(Ytil.T @ κ2)
      + He(Ftil @ (A_lambda @ X @ e[1] - X @ e[4]))
  )

  theta1_bar = -Λ2 - Λ3
  theta2 = He(Ftil @ B_bar @ Ktil @ (e[2] + e[5]))

  Theta_bar_1 = theta0_bar + theta2
  Theta_bar_2 = theta0_bar + h * theta1_bar + theta2
  Theta_bar_3 = theta0_bar + h * theta1_bar + exp_lam_h * theta2

  Γ_Ψ_sc = e[2].T @ X.T @ J1.T  # dim: (5nx + ny) x ny
  Γ_Ψ_ca = e[2].T @ X.T         # dim: (5nx + ny) x nx

  zero_ny_nx = np.zeros((ny, nx), dtype=dtype)
  zero_2nx_ny = np.zeros((2 * nx, ny), dtype=dtype)
  zero_2nx_nx = np.zeros((2 * nx, nx), dtype=dtype)

  Γ1 = cp.bmat([
      [Theta_bar_1, Γ_Ψ_sc, Γ_Ψ_ca],
      [Γ_Ψ_sc.T, -Ψsc_til, zero_ny_nx],
      [Γ_Ψ_ca.T, zero_ny_nx.T, -Ψca_til],
  ])

  Γ2 = cp.bmat([
      [Theta_bar_2, Ytil.T, Γ_Ψ_sc, Γ_Ψ_ca],
      [Ytil, -(1.0 / h) * Rcal, zero_2nx_ny, zero_2nx_nx],
      [Γ_Ψ_sc.T, zero_2nx_ny.T, -Ψsc_til, zero_ny_nx],
      [Γ_Ψ_ca.T, zero_2nx_nx.T, zero_ny_nx.T, -Ψca_til],
  ])

  Γ3 = cp.bmat([
      [Theta_bar_3, Ytil.T, Γ_Ψ_sc, Γ_Ψ_ca],
      [Ytil, -(1.0 / h) * Rcal, zero_2nx_ny, zero_2nx_nx],
      [Γ_Ψ_sc.T, zero_2nx_ny.T, -Ψsc_til, zero_ny_nx],
      [Γ_Ψ_ca.T, zero_2nx_nx.T, zero_ny_nx.T, -Ψca_til],
  ])

  I_nx = np.eye(nx, dtype=dtype)
  I_ny = np.eye(ny, dtype=dtype)

  constraints = [
      Ptil >> eps * I_nx,
      Rtil >> eps * I_nx,
      S1til >> eps * I_nx,
      Q1til >> eps * I_nx,
      Mtil >> eps * np.eye(2 * nx + ny, dtype=dtype),

      Ξsc >> eps * I_ny,
      Ψsc_til >> eps * I_ny,

      Ξca_til >> eps * I_nx,
      Ψca_til >> eps * I_nx,

      Γ1 << -eps * np.eye(Γ1.shape[0], dtype=dtype),
      Γ2 << -eps * np.eye(Γ2.shape[0], dtype=dtype),
      Γ3 << -eps * np.eye(Γ3.shape[0], dtype=dtype),
  ]

  objective = cp.Minimize(cp.trace(Ψca_til + Ξca_til + Ψsc_til + Ξsc))
  # objective = cp.Minimize(cp.trace(Ψsc_til + Ξsc))
  # objective = cp.Minimize(cp.trace(Ψca_til + Ξca_til))
  # objective = cp.Minimize(0.0)
  problem = cp.Problem(objective, constraints)

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
    Ψsc_til_val = np.asarray(Ψsc_til.value, dtype=dtype)
    Ψ_sc_rec = np.linalg.inv(Ψsc_til_val)

    Ψca_til_val = np.asarray(Ψca_til.value, dtype=dtype)
    Ψ_ca_rec = np.linalg.inv(Ψca_til_val)
  except np.linalg.LinAlgError:
    if verbose:
      print("[Recovery Error] Singular slack matrix X or Ψ_til detected.")
    return None

  Ktil_val = np.asarray(Ktil.value, dtype=dtype)
  Ξsc_val = np.asarray(Ξsc.value, dtype=dtype)
  Ξca_til_val = np.asarray(Ξca_til.value, dtype=dtype)

  Ptil_val = np.asarray(Ptil.value, dtype=dtype)
  Rtil_val = np.asarray(Rtil.value, dtype=dtype)

  # Reconstruction formulas:
  # K = K_tilde * X^{-1} * T
  # Ψ_sc = Ψsc_til^{-1},  Ξ_sc = Ξsc (direto, pois X2 = I_ny)
  # Ψ_ca = Ψca_til^{-1},  Ξ_ca = X^{-T} * Ξca_til * X^{-1}
  K_rec = Ktil_val @ X_inv @ T
  Ξ_sc_rec = Ξsc_val
  Ξ_ca_rec = X_inv.T @ Ξca_til_val @ X_inv

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
              "Ψ": Ψ_sc_rec,
              "Ξ": Ξ_sc_rec,
              "Ψtil": Ψsc_til_val,
          },
          "ca": {
              "Ψ": Ψ_ca_rec,
              "Ξ": Ξ_ca_rec,
              "Ψtil": Ψca_til_val,
              "Ξtil": Ξca_til_val,
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


def synthesize_continuous_gains(
    A: np.ndarray,
    C: np.ndarray,
    alpha_0: float,
    alpha_z: float,
    g0_method: str = "analytical",
    p_min: float = 1.0,
    solver: str = cp.MOSEK,
    verbose: bool = False,
    dtype: np.dtype = np.float64,
) -> Dict[str, Any]:
  """Sintetiza os ganhos contínuos G0, G1, G2 e mapeia para L0 e L2 no domínio original.

  Args:
      A: Matriz de estados contínua da planta (nx, nx).
      C: Matriz de saída medida (ny, nx). Deve ter posto completo de linhas.
      alpha_0: Taxa mínima de decaimento para os modos não medidos (A22 + G0).
      alpha_z: Taxa mínima de decaimento do estimador de Luenberger contínuo.
      g0_method: Método para síntese de G0: 'analytical' (forma fechada) ou 'lmi' (otimização).
      p_min: Cota inferior para a matriz de Lyapunov contínua Pz.
      solver: Solver semidefinido via CVXPY (padrão: cp.MOSEK).
      verbose: Habilita logs detalhados do solver.
      dtype: Precisão numérica de ponto flutuante.

  Returns:
      Dicionário contendo:
          - 'G0', 'G1', 'G2': Matrizes de ganho nas coordenadas transformadas.
          - 'L0', 'L2': Matrizes de ganho no domínio original.
          - 'T', 'T_inv': Matrizes de transformação de similaridade.
          - 'A_bar': Matriz A particionada nas coordenadas transformadas.
          - 'spectral_check': Autovalores de validação de cada subsistema.
  """
  A = np.asarray(A, dtype=dtype)
  C = np.asarray(C, dtype=dtype)
  nx, ny = A.shape[0], C.shape[0]
  n2 = nx - ny

  if n2 <= 0:
    raise ValueError(
        f"A ordem aumentada requer estados não medidos (ny < nx). Fornecido: nx={nx}, ny={ny}."
    )
  if alpha_0 <= 0.0 or alpha_z <= 0.0:
    raise ValueError(
        "As taxas 'alpha_0' e 'alpha_z' devem ser estritamente positivas.")

  T2 = null_space(C).T.astype(dtype)
  T = np.vstack([C, T2]).astype(dtype)

  cond_T = float(np.linalg.cond(T))
  if cond_T > 1e10:
    raise np.linalg.LinAlgError(
        f"A transformação T é mal condicionada (cond = {cond_T:.2e}).")

  T_inv = np.linalg.inv(T).astype(dtype)
  T_C_dagger = T_inv[:, :ny]  # nx x ny
  T_2_dagger = T_inv[:, ny:]  # nx x n2

  # Partição: \bar{A} = T @ A @ T^(-1)
  A_bar = T @ A @ T_inv
  A11 = A_bar[:ny, :ny]
  A12 = A_bar[:ny, ny:]
  A21 = A_bar[ny:, :ny]
  A22 = A_bar[ny:, ny:]

  if g0_method == "analytical":
    # Alocação direta exata: (A22 + G0) = -alpha_0 * I
    G0 = (-A22 - alpha_0 * np.eye(n2, dtype=dtype)).astype(dtype)
  elif g0_method == "lmi":
    # Minimização de ganho via SDP: He(P0*A22 + W0) + 2*alpha_0*P0 < 0
    P0 = cp.Variable((n2, n2), symmetric=True)
    W0 = cp.Variable((n2, n2))
    gamma_0 = cp.Variable(nonneg=True)

    lmi_g0 = A22.T @ P0 + P0 @ A22 + W0.T + W0 + 2.0 * alpha_0 * P0
    constraints_g0 = [
        P0 >> p_min * np.eye(n2, dtype=dtype),
        lmi_g0 << -1e-6 * np.eye(n2, dtype=dtype),
        cp.bmat([[gamma_0 * np.eye(n2, dtype=dtype), W0.T], [W0, P0]]) >> 0,
    ]
    prob_g0 = cp.Problem(cp.Minimize(gamma_0), constraints_g0)
    prob_g0.solve(solver=solver, verbose=verbose)

    if prob_g0.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
      raise RuntimeError(
          f"Infactibilidade na síntese de G0 via LMI: {prob_g0.status}.")

    G0 = np.linalg.solve(P0.value, W0.value).astype(dtype)
  else:
    raise ValueError(f"Método 'g0_method' desconhecido: {g0_method}.")

  A0 = np.block([[np.zeros((ny, ny), dtype=dtype), A12],
                [np.zeros((n2, ny), dtype=dtype), A22]])
  C0 = np.hstack([np.eye(ny, dtype=dtype), np.zeros((ny, n2), dtype=dtype)])

  Pz = cp.Variable((nx, nx), symmetric=True)
  Wz = cp.Variable((nx, ny))
  gamma_z = cp.Variable(nonneg=True)

  lmi_obs = A0.T @ Pz + Pz @ A0 - C0.T @ Wz.T - Wz @ C0 + 2.0 * alpha_z * Pz

  constraints_z = [
      Pz >> p_min * np.eye(nx, dtype=dtype),
      lmi_obs << -1e-6 * np.eye(nx, dtype=dtype),
      cp.bmat([[gamma_z * np.eye(ny, dtype=dtype), Wz.T], [Wz, Pz]]) >> 0,
  ]

  prob_z = cp.Problem(cp.Minimize(gamma_z), constraints_z)
  prob_z.solve(solver=solver, verbose=verbose)

  if prob_z.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    raise RuntimeError(
        f"Infactibilidade na síntese de G1/G2 via LMI: {prob_z.status}. "
        f"Verifique a detectabilidade de (A, C) ou reduza 'alpha_z'."
    )

  G = np.linalg.solve(Pz.value, Wz.value).astype(dtype)
  G1 = G[:ny, :]
  G2 = G[ny:, :]
  L0 = T_2_dagger @ G0 @ T2
  L2 = T_C_dagger @ (A11 + G1) + T_2_dagger @ (A21 + G2)
  A22_eff = A22 + G0
  A_z_eff = np.block([[-G1, A12], [-G2, A22]])
  A_e_cont = np.block([[A + L0, -L0], [L2 @ C, A - L2 @ C]])

  eig_A22 = np.linalg.eigvals(A22_eff)
  eig_Az = np.linalg.eigvals(A_z_eff)
  eig_Ae = np.linalg.eigvals(A_e_cont)

  spectral_check = {
      "max_real_A22_plus_G0": float(np.max(np.real(eig_A22))),
      "max_real_A_z": float(np.max(np.real(eig_Az))),
      "max_real_A_e": float(np.max(np.real(eig_Ae))),
      "is_A22_stable": bool(np.all(np.real(eig_A22) < 0.0)),
      "is_Az_stable": bool(np.all(np.real(eig_Az) < 0.0)),
  }

  return {
      "G0": G0,
      "G1": G1,
      "G2": G2,
      "L0": L0,
      "L2": L2,
      "T": T,
      "T_inv": T_inv,
      "A_bar": A_bar,
      "gamma_z": float(gamma_z.value),
      "spectral_check": spectral_check,
  }


def synthesize_impulsive_observer(
    design_params: Dict[str, Any],
    q_min: float = 1.0,
    eps: float = 1e-6,
    solver: str = cp.MOSEK,
    verbose: bool = False,
    dtype: np.dtype = np.float64,
) -> Optional[Dict[str, Any]]:
  """Síntese do Observador Híbrido Aumentado (2nx) via LMIs Multimodo.

  Args:
      design_params: Dicionário contendo:
          - 'A': Matriz dinâmica contínua da planta (nx, nx).
          - 'C': Matriz de saída medida (ny, nx).
          - 'h': Período base de amostragem síncrona (segundos).
          - 'nu_bar': Limite superior de intervalo inter-eventos (segundos).
          - 'lambda_obs': Taxa de decaimento exponencial prescrita.
          - 'L0': Matriz de ganho contínuo interno (nx, nx) [Opcional se G0 fornecido].
          - 'L2': Matriz de ganho contínuo de Luenberger (nx, ny) [Opcional se G1, G2 fornecidos].
          - Opcionais para Etapa 1: 'G0', 'G1', 'G2', 'T2'.
      q_min: Limite inferior para normalização espectral de Q1 e Q2.
      eps: Margem estrita de positividade para as LMIs.
      solver: Solver semidefinido via CVXPY (padrão: cp.MOSEK).
      verbose: Habilita logs detalhados do solver e da validação.
      dtype: Precisão numérica de ponto flutuante.

  Returns:
      Dicionário com os ganhos (L0, L1, L2), variáveis de Lyapunov e relatório
      espectral de estabilidade discreta multimodo, ou None em caso de infactibilidade.
  """
  required_base = ["A", "C", "h", "nu_bar", "lambda_obs"]
  missing = [k for k in required_base if k not in design_params]
  if missing:
    raise KeyError(
        f"[Erro de Configuração] Parâmetros ausentes: {missing}.")

  A = np.asarray(design_params["A"], dtype=dtype)
  C = np.asarray(design_params["C"], dtype=dtype)
  h = float(design_params["h"])
  nu_bar = float(design_params["nu_bar"])
  lambda_obs = float(design_params["lambda_obs"])

  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError(
        f"A matriz 'A' deve ser quadrada. Dimensões: {A.shape}.")
  if C.ndim != 2 or C.shape[1] != A.shape[0]:
    raise ValueError(
        f"Incompatibilidade dimensional entre A {A.shape} e C {C.shape}.")
  if h <= 0.0 or nu_bar < h or lambda_obs < 0.0:
    raise ValueError(
        "Parâmetros temporais inválidos (exigido: h > 0, nu_bar >= h, lambda_obs >= 0).")

  nx, ny = A.shape[0], C.shape[0]
  nu_bar_steps = int(round(nu_bar / h))

  if "L0" in design_params and "L2" in design_params:
    L0 = np.asarray(design_params["L0"], dtype=dtype)
    L2 = np.asarray(design_params["L2"], dtype=dtype)
  else:
    raise KeyError(
        "É necessário fornecer 'L0' e 'L2' diretamente OU fornecer 'G0', 'G1' e 'G2' em design_params."
    )

  if L0.shape != (nx, nx) or L2.shape != (nx, ny):
    raise ValueError(
        f"Dimensões incompatíveis: L0 deve ser ({nx},{nx}) e L2 ({nx},{ny}).")

  A_e = np.block([
      [A + L0, -L0],
      [L2 @ C, A - L2 @ C]
  ]).astype(dtype)

  Q1 = cp.Variable((nx, nx), symmetric=True)
  Q2 = cp.Variable((nx, nx), symmetric=True)
  Y1 = cp.Variable((nx, ny))
  gamma_L = cp.Variable(nonneg=True)

  Q = cp.bmat([
      [Q1, np.zeros((nx, nx), dtype=dtype)],
      [np.zeros((nx, nx), dtype=dtype), Q2]
  ])
  Y_e = cp.bmat([
      [Y1],
      [np.zeros((nx, ny), dtype=dtype)]
  ])
  C_e = np.hstack([C, np.zeros((ny, nx), dtype=dtype)])  # (ny, 2nx)

  constraints = [
      Q1 >> q_min * np.eye(nx, dtype=dtype),
      Q2 >> q_min * np.eye(nx, dtype=dtype),
  ]

  for j in range(1, nu_bar_steps + 1):
    tau_j = j * h
    Phi_e_j = expm(A_e * tau_j).astype(dtype)
    decay = float(np.exp(-2.0 * lambda_obs * tau_j))

    C_Phi = C_e @ Phi_e_j  # (ny, 2nx)
    M_j = Q @ Phi_e_j - Y_e @ C_Phi
    LMI_j = cp.bmat([
        [decay * Q, M_j.T],
        [M_j, Q]
    ])
    constraints.append(LMI_j >> eps * np.eye(4 * nx, dtype=dtype))

  schur_gamma = cp.bmat([
      [gamma_L * np.eye(ny, dtype=dtype), Y1.T],
      [Y1, Q1]
  ])
  constraints.append(schur_gamma >> 0)

  problem = cp.Problem(cp.Minimize(gamma_L), constraints)
  try:
    problem.solve(solver=solver, verbose=verbose)
  except Exception as err:
    if verbose:
      print(f"[Solver Error] Falha de execução do solver: {err}")
    return None

  if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    if verbose:
      print(
          f"[Solver Status] Problema não convergente: {problem.status}")
    return None

  Q1_val = np.asarray(Q1.value, dtype=dtype)
  Q2_val = np.asarray(Q2.value, dtype=dtype)
  Y1_val = np.asarray(Y1.value, dtype=dtype)

  cond_Q1 = float(np.linalg.cond(Q1_val))
  if cond_Q1 > 1e12:
    if verbose:
      print(
          f"[Numerical Warning] Q1 mal condicionada (cond = {cond_Q1:.2e}).")
    return None

  try:
    # Reconstrução estrita: L1 = Q1^(-1) * Y1
    L1_rec = np.linalg.solve(Q1_val, Y1_val)
  except np.linalg.LinAlgError:
    if verbose:
      print(
          "[Reconstruction Error] Singularidade numérica na recuperação de L1.")
    return None

  J_e = np.block([
      [np.eye(nx, dtype=dtype) - L1_rec @ C,
       np.zeros((nx, nx), dtype=dtype)],
      [np.zeros((nx, nx), dtype=dtype),      np.eye(nx, dtype=dtype)]
  ]).astype(dtype)

  modes_validation = []
  worst_spectral_radius = -1.0
  worst_step = 1

  for j in range(1, nu_bar_steps + 1):
    tau_j = j * h
    Phi_e_j = expm(A_e * tau_j).astype(dtype)
    A_e_discrete_j = J_e @ Phi_e_j

    eigvals = np.linalg.eigvals(A_e_discrete_j)
    rho_j = float(np.max(np.abs(eigvals)))
    rho_prescribed = float(np.exp(-lambda_obs * tau_j))

    is_stable = rho_j < 1.0
    meets_rate = rho_j <= (rho_prescribed + 1e-4)

    modes_validation.append({
        "step": j,
        "tau": tau_j,
        "spectral_radius": rho_j,
        "rho_prescribed": rho_prescribed,
        "is_stable": is_stable,
        "meets_prescribed_rate": meets_rate,
    })

    if rho_j > worst_spectral_radius:
      worst_spectral_radius = rho_j
      worst_step = j

  if worst_spectral_radius >= 1.0:
    if verbose:
      print(
          f"[Validação Crítica] O ganho L1 violou a estabilidade: "
          f"Modo tau = {worst_step * h:.4f}s -> Raio Espectral = {worst_spectral_radius:.4f} >= 1.0."
      )
    return None

  return {
      "L0": L0,
      "L1": L1_rec,
      "L2": L2,
      "A_e": A_e,
      "gamma_L": float(gamma_L.value),
      "Q1": Q1_val,
      "Q2": Q2_val,
      "Y1": Y1_val,
      "cond_Q1": cond_Q1,
      "nu_bar_steps": nu_bar_steps,
      "modes": modes_validation,
      "worst_case": {
          "step": worst_step,
          "tau": worst_step * h,
          "spectral_radius": worst_spectral_radius,
      },
  }
