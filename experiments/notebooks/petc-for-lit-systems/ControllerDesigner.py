import scipy.linalg as la
from typing import Dict, Any, Optional
from typing import Dict, Any
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import cvxpy as cp
import numpy as np
from Utils import nm


@dataclass
class ProblemDataLIT:
  nx: int
  nu: int
  ny: int
  h: float
  υ: float
  λ: float
  A: cp.Parameter
  B: cp.Parameter
  e: List[Any]
  dtype: Any


@dataclass
class DecisionVariables:
  Ptil: cp.Variable
  Mtil: cp.Variable
  Q1til: cp.Variable
  Q2til: cp.Variable
  Q3til: cp.Variable
  S1til: cp.Variable
  S2til: cp.Variable
  S3til: cp.Variable
  Ktil: cp.Variable
  Rtil: cp.Variable
  Ytil: cp.Variable
  Ξtil: cp.Variable
  Ψtil: cp.Variable
  X: cp.Variable
  γ: cp.Variable


@dataclass
class Constants:
  Onx: np.ndarray
  exp_λh: float
  Rcal: cp.Expression
  Fscr: Any
  κ1: cp.Expression
  κ2: cp.Expression
  Λ1: cp.Expression
  Λ2: cp.Expression
  Λ3: cp.Expression


def synthesize_setm(
    design_params: Dict[str, Any], eps: float = 1e-6, dtype=np.float64
) -> Dict[str, Any]:
  """
  Sintetiza o ganho do controlador K e as matrizes do SETM (Ξ, Ψ) via LMIs 
  para um sistema Linear Invariante no Tempo (LIT).

  Estrutura esperada de `design_params`:
  {
      'A': np.ndarray,  # Matriz de dinâmica A (nx x nx)
      'B': np.ndarray,  # Matriz de entrada B (nx x nu)
      'h': float,       # Período de amostragem fundamental h
      'υ': float,       # Parâmetro upsilon
      'λ': float        # Parâmetro lambda da taxa de decaimento
  }
  """

  def _create_problem_data() -> ProblemDataLIT:
    mat_A = np.array(design_params['A'], dtype=dtype)
    mat_B = np.array(design_params['B'], dtype=dtype)

    _nx = mat_A.shape[0]
    _nu = mat_B.shape[1]
    _ny = _nx

    _h = float(design_params['h'])
    _υ = float(design_params['υ'])
    _λ = float(design_params['λ'])

    _A_param = cp.Parameter((_nx, _nx), value=mat_A)
    _B_param = cp.Parameter((_nx, _nu), value=mat_B)

    _e = nm.get_e(5 * [_nx])
    for idx in range(1, len(_e)):
      _e[idx] = cp.Parameter(_e[idx].shape, value=_e[idx].astype(dtype))

    return ProblemDataLIT(
        nx=_nx, nu=_nu, ny=_ny, h=_h, υ=_υ, λ=_λ,
        A=_A_param, B=_B_param, e=_e, dtype=dtype
    )

  def _create_variables() -> DecisionVariables:
    nx = data.nx
    nu = data.nu
    e_shape_1 = data.e[1].shape[1]

    return DecisionVariables(
        Ptil=cp.Variable((nx, nx), PSD=True),
        Mtil=cp.Variable((2*nx, 2*nx), PSD=True),
        Q1til=cp.Variable((nx, nx), symmetric=True),
        Q2til=cp.Variable((nx, nx)),
        Q3til=cp.Variable((nx, nx)),
        S1til=cp.Variable((nx, nx), symmetric=True),
        S2til=cp.Variable((nx, nx)),
        S3til=cp.Variable((nx, nx)),
        Ktil=cp.Variable((nu, nx)),
        Rtil=cp.Variable((nx, nx), PSD=True),
        Ξtil=cp.Variable((nx, nx), PSD=True),
        Ψtil=cp.Variable((nx, nx), PSD=True),
        X=cp.Variable((nx, nx)),
        Ytil=cp.Variable((2*nx, e_shape_1)),
        γ=cp.Variable(nonneg=True)
    )

  def _build_constants() -> Constants:
    nx = data.nx
    e = data.e
    υ = data.υ
    λ = data.λ
    h = data.h

    _Onx = np.zeros((nx, nx), dtype=data.dtype)
    _exp_λh = np.exp(- 2. * λ * h)
    _Rcal = cp.bmat([[vars.Rtil, _Onx], [_Onx, 3. * vars.Rtil]])
    _Fscr = e[1].T + υ * e[2].T + υ * e[4].T
    _κ1 = cp.bmat([[e[2]], [e[5]]])
    _κ2 = cp.bmat([[e[1] - e[2]], [e[1] + e[2] - 2. * e[3]]])

    _Λ1 = (e[1] - e[2]).T @ vars.S1til @ (e[1] - e[2]) + \
        nm.He((e[1] - e[2]).T @ (vars.S2til @ e[2] + vars.S3til @ e[5]))

    _Λ2 = _κ1.T @ vars.Mtil @ _κ1 + \
        e[4].T @ vars.Rtil @ e[4] - \
        e[3].T @ vars.Q1til @ e[3] + \
        nm.He(
            (e[1] - e[2]).T @ vars.S1til @ e[4] +
            e[4].T @ vars.S2til @ e[2] +
            e[4].T @ vars.S3til @ e[5] +
            e[1].T @ vars.Q1til @ e[3] +
            e[1].T @ vars.Q2til @ e[2] +
            e[1].T @ vars.Q3til @ e[5]
        )

    _Λ3 = e[3].T @ vars.Q1til @ e[3] + _κ1.T @ vars.Mtil @ _κ1 + \
        nm.He(e[3].T @ (vars.Q2til @ e[2] + vars.Q3til @ e[5]))

    return Constants(
        Onx=_Onx, exp_λh=_exp_λh, Rcal=_Rcal, Fscr=_Fscr,
        κ1=_κ1, κ2=_κ2, Λ1=_Λ1, Λ2=_Λ2, Λ3=_Λ3
    )

  def _theta0() -> cp.Expression:
    e = data.e
    Bscr = data.A @ vars.X @ e[1] + data.B @ vars.Ktil @ e[2] + \
        data.B @ vars.Ktil @ e[5] - vars.X @ e[4]
    return - e[5].T @ vars.Ξtil @ e[5] - \
        2. * data.λ * consts.exp_λh * consts.Λ1 + \
        (1. - consts.exp_λh) * consts.Λ2 + \
        nm.He(consts.Fscr @ Bscr + e[1].T @ vars.Ptil @ e[4] +
              data.λ * e[1].T @ vars.Ptil @ e[1] -
              2. * data.λ * consts.exp_λh * consts.κ2.T @ vars.Ytil)

  def _thetah() -> cp.Expression:
    e = data.e
    Bscr = data.A @ vars.X @ e[1] + data.B @ vars.Ktil @ e[2] + \
        data.B @ vars.Ktil @ e[5] - vars.X @ e[4]
    return - e[5].T @ vars.Ξtil @ e[5] - 2. * data.λ * consts.exp_λh * \
        (consts.Λ1 + data.h * consts.Λ3 + nm.He(consts.κ2.T @ vars.Ytil)) + \
        nm.He(consts.Fscr @ Bscr + e[1].T @ vars.Ptil @ e[4] +
              data.λ * e[1].T @ vars.Ptil @ e[1])

  def _gamma() -> Tuple[cp.Expression, cp.Expression]:
    e = data.e
    nx = data.nx

    Γ1_11 = _theta0()
    Γ1_12 = e[2].T @ vars.X.T
    Γ1_21 = Γ1_12.T
    Γ1_22 = - vars.Ψtil

    Γ1 = cp.bmat([[Γ1_11, Γ1_12],
                  [Γ1_21, Γ1_22]])

    Γ2_11 = _thetah()
    Γ2_12 = vars.Ytil.T
    Γ2_13 = e[2].T @ vars.X.T

    Γ2_21 = Γ2_12.T
    Γ2_22 = - (np.exp(2. * data.λ * data.h) /
               (2. * data.λ * data.h)) * consts.Rcal
    Γ2_23 = np.zeros((2 * nx, nx))

    Γ2_31 = Γ2_13.T
    Γ2_32 = Γ2_23.T
    Γ2_33 = - vars.Ψtil

    Γ2 = cp.bmat([[Γ2_11, Γ2_12, Γ2_13],
                  [Γ2_21, Γ2_22, Γ2_23],
                  [Γ2_31, Γ2_32, Γ2_33]])

    return Γ1, Γ2

  def _recover_solution() -> Dict[str, Any]:
    if problem.status != cp.OPTIMAL:
      return None

    nx = data.nx
    dt = data.dtype

    X_val = vars.X.value.astype(dt)
    Xinv = np.linalg.inv(X_val)
    XinvT = Xinv.T

    print("cond(X) =", np.linalg.cond(X_val))
    print("eig(X) =", np.linalg.eigvalsh(X_val))
    print("||X @ Xinv - I|| =",
          np.linalg.norm(X_val @ Xinv - np.eye(nx)))

    Xinv_2x = np.block([[Xinv, np.zeros((nx, nx), dtype=dt)],
                        [np.zeros((nx, nx), dtype=dt), Xinv]])
    XinvT_2x = Xinv_2x.T

    Ξ = XinvT @ vars.Ξtil.value.astype(dt) @ Xinv
    Ψ = np.linalg.inv(vars.Ψtil.value.astype(dt))
    K = vars.Ktil.value.astype(dt) @ Xinv

    P = XinvT @ vars.Ptil.value.astype(dt) @ Xinv
    R = XinvT @ vars.Rtil.value.astype(dt) @ Xinv

    S1 = XinvT @ vars.S1til.value.astype(dt) @ Xinv
    S2 = XinvT @ vars.S2til.value.astype(dt) @ Xinv
    S3 = XinvT @ vars.S3til.value.astype(dt) @ Xinv

    Q1 = XinvT @ vars.Q1til.value.astype(dt) @ Xinv
    Q2 = XinvT @ vars.Q2til.value.astype(dt) @ Xinv
    Q3 = XinvT @ vars.Q3til.value.astype(dt) @ Xinv

    M = XinvT_2x @ vars.Mtil.value.astype(dt) @ Xinv_2x
    Y = XinvT_2x @ vars.Ytil.value.astype(dt)

    return {
        'optimal_value': problem.value,
        'etm': {'Ξ': Ξ, 'Ψ': Ψ},
        'controller': {'K': K},
        'functional': {
            'P': P, 'R': R, 'M': M,
            'S1': S1, 'S2': S2, 'S3': S3,
            'Q1': Q1, 'Q2': Q2, 'Q3': Q3,
            'Y': Y
        },
    }

  data = _create_problem_data()
  vars = _create_variables()
  consts = _build_constants()

  Γ1, Γ2 = _gamma()
  constraints = [
      Γ1 << -eps * np.eye(Γ1.shape[0], dtype=dtype),
      Γ2 << -eps * np.eye(Γ2.shape[0], dtype=dtype),
      vars.Ψtil >> eps * np.eye(data.nx, dtype=dtype),
      vars.Ξtil >> eps * np.eye(data.nx, dtype=dtype)
  ]

  obj = cp.Minimize(cp.trace(vars.Ξtil + vars.Ψtil))
  problem = cp.Problem(obj, constraints)
  problem.solve(solver=cp.MOSEK, verbose=False, ignore_dpp=True)

  return _recover_solution()


def synthesize_setm_obs(
    design_params: Dict[str, Any], eps: float = 1e-6, dtype=np.float32
) -> Dict[str, Any]:
  """
  Sintetiza o ganho do controlador K e as matrizes do SETM (Ξ, Ψ) via LMIs 
  para um sistema Linear Invariante no Tempo (LIT).

  Estrutura esperada de `design_params`:
  {
      'A': np.ndarray,  # Matriz de dinâmica A (nx x nx)
      'B': np.ndarray,  # Matriz de entrada B (nx x nu)
      'h': float,       # Período de amostragem fundamental h
      'υ': float,       # Parâmetro upsilon
      'λ': float        # Parâmetro lambda da taxa de decaimento
  }
  """

  def _create_problem_data() -> ProblemDataLIT:
    mat_A = np.array(design_params['A'], dtype=dtype)
    mat_B = np.array(design_params['B'], dtype=dtype)

    _nx = mat_A.shape[0]
    _nu = mat_B.shape[1]
    _ny = _nx  # Assumindo saída completa para o controlador

    _h = float(design_params['h'])
    _υ = float(design_params['υ'])
    _λ = float(design_params['λ'])

    _A_param = cp.Parameter((_nx, _nx), value=mat_A)
    _B_param = cp.Parameter((_nx, _nu), value=mat_B)

    _e = nm.get_e(5 * [_nx] + [_ny])
    for idx in range(1, len(_e)):
      _e[idx] = cp.Parameter(_e[idx].shape, value=_e[idx].astype(dtype))

    return ProblemDataLIT(
        nx=_nx, nu=_nu, ny=_ny, h=_h, υ=_υ, λ=_λ,
        A=_A_param, B=_B_param, e=_e, dtype=dtype
    )

  def _create_variables() -> DecisionVariables:
    nx = data.nx
    nu = data.nu
    e_shape_1 = data.e[1].shape[1]

    return DecisionVariables(
        Ptil=cp.Variable((nx, nx), PSD=True),
        Mtil=cp.Variable((2*nx, 2*nx), PSD=True),
        Q1til=cp.Variable((nx, nx), symmetric=True),
        Q2til=cp.Variable((nx, nx)),
        Q3til=cp.Variable((nx, nx)),
        S1til=cp.Variable((nx, nx), symmetric=True),
        S2til=cp.Variable((nx, nx)),
        S3til=cp.Variable((nx, nx)),
        Ktil=cp.Variable((nu, nx)),
        Rtil=cp.Variable((nx, nx), PSD=True),
        Ξtil=cp.Variable((nx, nx), PSD=True),
        Ψtil=cp.Variable((nx, nx), PSD=True),
        X=cp.Variable((nx, nx)),
        Ytil=cp.Variable((2*nx, e_shape_1)),
        γ=cp.Variable(nonneg=True)
    )

  def _build_constants() -> Constants:
    nx = data.nx
    e = data.e
    υ = data.υ
    λ = data.λ
    h = data.h

    _Onx = np.zeros((nx, nx), dtype=data.dtype)
    _exp_λh = np.exp(- 2. * λ * h)
    _Rcal = cp.bmat([[vars.Rtil, _Onx], [_Onx, 3. * vars.Rtil]])
    _Fscr = e[1].T + υ * e[2].T + υ * e[4].T
    _κ1 = cp.bmat([[e[2]], [e[5]]])
    _κ2 = cp.bmat([[e[1] - e[2]], [e[1] + e[2] - 2. * e[3]]])

    _Λ1 = (e[1] - e[2]).T @ vars.S1til @ (e[1] - e[2]) + \
        nm.He((e[1] - e[2]).T @ (vars.S2til @ e[2] + vars.S3til @ e[5]))

    _Λ2 = _κ1.T @ vars.Mtil @ _κ1 + \
        e[4].T @ vars.Rtil @ e[4] - \
        e[3].T @ vars.Q1til @ e[3] + \
        nm.He(
            (e[1] - e[2]).T @ vars.S1til @ e[4] +
            e[4].T @ vars.S2til @ e[2] +
            e[4].T @ vars.S3til @ e[5] +
            e[1].T @ vars.Q1til @ e[3] +
            e[1].T @ vars.Q2til @ e[2] +
            e[1].T @ vars.Q3til @ e[5]
        )

    _Λ3 = e[3].T @ vars.Q1til @ e[3] + _κ1.T @ vars.Mtil @ _κ1 + \
        nm.He(e[3].T @ (vars.Q2til @ e[2] + vars.Q3til @ e[5]))

    return Constants(
        Onx=_Onx, exp_λh=_exp_λh, Rcal=_Rcal, Fscr=_Fscr,
        κ1=_κ1, κ2=_κ2, Λ1=_Λ1, Λ2=_Λ2, Λ3=_Λ3
    )

  def _theta0() -> cp.Expression:
    e = data.e
    Bscr = data.A @ vars.X @ e[1] + data.B @ vars.Ktil @ e[2] + \
        data.B @ vars.Ktil @ e[5] - vars.X @ e[4] - data.B @ vars.Ktil @ e[6]
    return - e[5].T @ vars.Ξtil @ e[5] - vars.γ * e[6].T @ e[6] - \
        2. * data.λ * consts.exp_λh * consts.Λ1 + \
        (1. - consts.exp_λh) * consts.Λ2 + \
        nm.He(consts.Fscr @ Bscr + e[1].T @ vars.Ptil @ e[4] +
              data.λ * e[1].T @ vars.Ptil @ e[1] -
              2. * data.λ * consts.exp_λh * consts.κ2.T @ vars.Ytil)

  def _thetah() -> cp.Expression:
    e = data.e
    Bscr = data.A @ vars.X @ e[1] + data.B @ vars.Ktil @ e[2] + \
        data.B @ vars.Ktil @ e[5] - vars.X @ e[4] - data.B @ vars.Ktil @ e[6]
    return - e[5].T @ vars.Ξtil @ e[5] - vars.γ * e[6].T @ e[6] - 2. * data.λ * consts.exp_λh * \
        (consts.Λ1 + data.h * consts.Λ3 + nm.He(consts.κ2.T @ vars.Ytil)) + \
        nm.He(consts.Fscr @ Bscr + e[1].T @ vars.Ptil @ e[4] +
              data.λ * e[1].T @ vars.Ptil @ e[1])

  def _gamma() -> Tuple[cp.Expression, cp.Expression]:
    e = data.e
    nx = data.nx
    ny = data.ny

    Γ1_11 = _theta0()
    Γ1_12 = e[2].T @ vars.X.T
    Γ1_13 = e[1].T @ vars.X.T
    Γ1_21 = Γ1_12.T
    Γ1_22 = - vars.Ψtil
    Γ1_23 = np.zeros((nx, nx))
    Γ1_31 = Γ1_13.T
    Γ1_32 = Γ1_23.T
    Γ1_33 = - np.eye(ny, dtype=data.dtype)

    Γ1 = cp.bmat([[Γ1_11, Γ1_12, Γ1_13],
                  [Γ1_21, Γ1_22, Γ1_23],
                  [Γ1_31, Γ1_32, Γ1_33]])

    Γ2_11 = _thetah()
    Γ2_12 = vars.Ytil.T
    Γ2_13 = e[2].T @ vars.X.T
    Γ2_14 = e[1].T @ vars.X.T

    Γ2_21 = Γ2_12.T
    Γ2_22 = - (np.exp(2. * data.λ * data.h) /
               (2. * data.λ * data.h)) * consts.Rcal
    Γ2_23 = np.zeros((2 * nx, nx))
    Γ2_24 = np.zeros((2 * nx, ny))

    Γ2_31 = Γ2_13.T
    Γ2_32 = Γ2_23.T
    Γ2_33 = - vars.Ψtil
    Γ2_34 = np.zeros((nx, ny))

    Γ2_41 = Γ2_14.T
    Γ2_42 = Γ2_24.T
    Γ2_43 = Γ2_34.T
    Γ2_44 = - np.eye(ny, dtype=data.dtype)

    Γ2 = cp.bmat([[Γ2_11, Γ2_12, Γ2_13, Γ2_14],
                  [Γ2_21, Γ2_22, Γ2_23, Γ2_24],
                  [Γ2_31, Γ2_32, Γ2_33, Γ2_34],
                  [Γ2_41, Γ2_42, Γ2_43, Γ2_44]])

    return Γ1, Γ2

  def _recover_solution() -> Dict[str, Any]:
    if problem.status in ["infeasible", "unbounded"]:
      return None

    nx = data.nx
    dt = data.dtype

    X_val = vars.X.value.astype(dt)
    Xinv = np.linalg.inv(X_val)
    XinvT = Xinv.T

    Xinv_2x = np.block([[Xinv, np.zeros((nx, nx), dtype=dt)],
                        [np.zeros((nx, nx), dtype=dt), Xinv]])
    XinvT_2x = Xinv_2x.T

    Ξ = XinvT @ vars.Ξtil.value.astype(dt) @ Xinv
    Ψ = np.linalg.inv(vars.Ψtil.value.astype(dt))
    K = vars.Ktil.value.astype(dt) @ Xinv

    P = XinvT @ vars.Ptil.value.astype(dt) @ Xinv
    R = XinvT @ vars.Rtil.value.astype(dt) @ Xinv

    S1 = XinvT @ vars.S1til.value.astype(dt) @ Xinv
    S2 = XinvT @ vars.S2til.value.astype(dt) @ Xinv
    S3 = XinvT @ vars.S3til.value.astype(dt) @ Xinv

    Q1 = XinvT @ vars.Q1til.value.astype(dt) @ Xinv
    Q2 = XinvT @ vars.Q2til.value.astype(dt) @ Xinv
    Q3 = XinvT @ vars.Q3til.value.astype(dt) @ Xinv

    M = XinvT_2x @ vars.Mtil.value.astype(dt) @ Xinv_2x
    Y = XinvT_2x @ vars.Ytil.value.astype(dt)

    return {
        'optimal_value': {
            'gamma_L': vars.γ.value,
            'etm_matrices': problem.value - vars.γ.value
        },
        'etm': {'Ξ': Ξ, 'Ψ': Ψ},
        'controller': {'K': K},
        'functional': {
            'P': P, 'R': R, 'M': M,
            'S1': S1, 'S2': S2, 'S3': S3,
            'Q1': Q1, 'Q2': Q2, 'Q3': Q3,
            'Y': Y
        },
    }

  data = _create_problem_data()
  vars = _create_variables()
  consts = _build_constants()

  Γ1, Γ2 = _gamma()
  constraints = [
      Γ1 << -eps * np.eye(Γ1.shape[0], dtype=dtype),
      Γ2 << -eps * np.eye(Γ2.shape[0], dtype=dtype),
      vars.Ψtil >> eps * np.eye(data.nx, dtype=dtype),
      vars.Ξtil >> eps * np.eye(data.nx, dtype=dtype)
  ]

  obj = cp.Minimize(cp.trace(vars.Ξtil + vars.Ψtil) + vars.γ)
  problem = cp.Problem(obj, constraints)
  problem.solve(solver=cp.MOSEK, verbose=False, ignore_dpp=True)

  return _recover_solution()


def synthesize_impulsive_observer(
    design_params: Dict[str, Any],
    eps: float = 1e-7,
    solver: Any = cp.MOSEK,
    dtype=np.float64
) -> Optional[Dict[str, Any]]:
  """
  Synthesize an impulsive observer gain for a fixed decay rate alpha
  and impulsive contraction factor rho using Finsler's Lemma.

  The synthesis is based on a clock-dependent Lyapunov function
  P(tau) = P0 + tau P1 and the continuous-time error dynamics
      dot{e}(t) = A e(t),
  with the impulsive reset
      e(t_k^+) = (I - L C) e(t_k^-).

  By employing Finsler's Lemma, a slack matrix X decouples P(tau) 
  from the system matrices. The gain is recovered as L = X^{-1} Y.
  The solver strictly minimizes the upper bound (gamma_L) of the 
  control effort Y.

  Parameters
  ----------
  design_params : Dict[str, Any]
      Dictionary containing the plant matrices A and C, sampling 
      parameters h and nu_bar, and the Finsler tuning scalar 'epsilon'.
  eps : float, optional
      Numerical regularization used to enforce strict definiteness.
  solver : Any, optional
      CVXPY solver used for the semidefinite program.
  dtype : numpy.dtype, optional
      Numerical data type used internally.

  Returns
  -------
  Optional[Dict[str, Any]]
      Dictionary containing the synthesized observer gain, Lyapunov 
      matrices, and solver metrics if feasible. Returns None otherwise.
  """

  # ------------------------------------------------------------------
  # 1. Retrieve plant and PETC parameters
  # ------------------------------------------------------------------
  A = np.asarray(design_params["A"], dtype=dtype)
  C = np.asarray(design_params["C"], dtype=dtype)

  h = float(design_params["h"])
  nu_bar = float(design_params["nu_bar"])

  # Parâmetro escalar de sintonia do Lema de Finsler (default = 1.0)
  epsilon = float(design_params.get("epsilon", 1.0))

  nx = A.shape[0]
  ny = C.shape[0]

  # ------------------------------------------------------------------
  # 2. Basic consistency checks
  # ------------------------------------------------------------------
  alpha = design_params['alpha']
  rho = design_params['rho']

  if A.shape != (nx, nx):
    raise ValueError("A must be a square matrix.")
  if C.shape[1] != nx:
    raise ValueError("C has incompatible dimensions with A.")
  if h <= 0.0:
    raise ValueError("The sampling period h must be positive.")
  if nu_bar < h:
    raise ValueError("nu_bar must satisfy nu_bar >= h.")
  if alpha < 0.0:
    raise ValueError("alpha must be nonnegative.")
  if rho <= 0.0:
    raise ValueError("rho must be strictly positive.")

  # ------------------------------------------------------------------
  # 3. Check the global decay condition: rho * exp(2 alpha nu_bar) < 1
  # ------------------------------------------------------------------
  decay_factor = rho * np.exp(2.0 * alpha * nu_bar)
  if decay_factor >= 1.0:
    return None

  # ------------------------------------------------------------------
  # 4. Decision variables
  # ------------------------------------------------------------------
  P0 = cp.Variable((nx, nx), symmetric=True)
  P1 = cp.Variable((nx, nx), symmetric=True)

  # Variáveis de Finsler
  X = cp.Variable((nx, nx))
  Y = cp.Variable((nx, ny))

  # Limitante de energia do esforço do observador
  gamma_L = cp.Variable(nonneg=True)

  P_h = P0 + h * P1
  P_bar = P0 + nu_bar * P1

  # ------------------------------------------------------------------
  # 5. Numerical matrices
  # ------------------------------------------------------------------
  I_x = np.eye(nx, dtype=dtype)
  I_y = np.eye(ny, dtype=dtype)

  # ------------------------------------------------------------------
  # 6. Positivity conditions
  # ------------------------------------------------------------------
  LMI_P0 = P0 - eps * I_x
  LMI_Pbar = P_bar - eps * I_x

  # ------------------------------------------------------------------
  # 7. Flow conditions (Affine in tau -> checking endpoints)
  # ------------------------------------------------------------------
  LMI_FLOW_0 = (P1 + A.T @ P0 + P0 @ A - 2.0 * alpha * P0)
  LMI_FLOW_bar = (P1 + A.T @ P_bar + P_bar @ A - 2.0 * alpha * P_bar)

  # ------------------------------------------------------------------
  # 8. Impulsive jump conditions (Finsler Formulation)
  # ------------------------------------------------------------------
  X_minus_YC = X - Y @ C
  eps_X_T = epsilon * X.T

  # Sub-blocos independentes de tau
  Psi_11 = P0 - X - X.T
  Psi_12 = X_minus_YC - eps_X_T
  Psi_22_base = epsilon * X_minus_YC + epsilon * X_minus_YC.T

  # Matrizes finais para os extremos do salto
  Psi_22_h = -rho * P_h + Psi_22_base
  Psi_22_bar = -rho * P_bar + Psi_22_base

  JUMP_h = cp.bmat([
      [Psi_11,   Psi_12],
      [Psi_12.T, Psi_22_h]
  ])

  JUMP_bar = cp.bmat([
      [Psi_11,   Psi_12],
      [Psi_12.T, Psi_22_bar]
  ])

  # ------------------------------------------------------------------
  # 9. Gain Norm Bound (Energy Limit for Y)
  # Schur complement para garantir Y^T * Y <= gamma_L * I
  # ------------------------------------------------------------------
  LMI_NORM = cp.bmat([
      [gamma_L * I_y, Y.T],
      [Y,             I_x]
  ])

  # ------------------------------------------------------------------
  # 10. Optimization problem setup (Minimize objective gamma_L)
  # ------------------------------------------------------------------
  constraints = [
      LMI_P0 >> 0.0,
      LMI_Pbar >> 0.0,
      LMI_FLOW_0 << 0.0,
      LMI_FLOW_bar << 0.0,
      JUMP_h << 0.0,
      JUMP_bar << 0.0,
      LMI_NORM >> 0.0
  ]

  problem = cp.Problem(cp.Minimize(gamma_L), constraints)

  # ------------------------------------------------------------------
  # 11. Solve SDP
  # ------------------------------------------------------------------
  try:
    problem.solve(solver=solver, verbose=False, ignore_dpp=True)
  except cp.SolverError:
    try:
      problem.solve(solver=cp.SCS, verbose=False)
    except cp.SolverError:
      return None

  # ------------------------------------------------------------------
  # 12. Check feasibility
  # ------------------------------------------------------------------
  if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
    return None

  if P0.value is None or X.value is None or Y.value is None:
    return None

  # ------------------------------------------------------------------
  # 13. Recover numerical solution
  # ------------------------------------------------------------------
  P0_val = np.asarray(P0.value, dtype=dtype)
  P1_val = np.asarray(P1.value, dtype=dtype)
  X_val = np.asarray(X.value, dtype=dtype)
  Y_val = np.asarray(Y.value, dtype=dtype)
  g_L_val = float(gamma_L.value)

  # Numerical symmetrization
  P0_val = 0.5 * (P0_val + P0_val.T)
  P1_val = 0.5 * (P1_val + P1_val.T)

  # ------------------------------------------------------------------
  # 14. Recover observer gain
  #
  #     Y = X L => X L = Y
  #     L = X^{-1} Y
  # ------------------------------------------------------------------
  try:
    # np.linalg.solve resolve A x = B -> X L = Y
    L = np.linalg.solve(X_val, Y_val)
  except np.linalg.LinAlgError:
    return None

  # ------------------------------------------------------------------
  # 15. Post-processing
  # ------------------------------------------------------------------
  eig_P0 = np.linalg.eigvalsh(P0_val)
  P_h_val = P0_val + h * P1_val
  P_bar_val = P0_val + nu_bar * P1_val
  eig_P_h = np.linalg.eigvalsh(P_h_val)
  eig_P_bar = np.linalg.eigvalsh(P_bar_val)

  eta = rho * np.exp(2.0 * alpha * nu_bar)
  event_decay_rate = -np.log(eta) if eta > 0.0 else np.inf

  # ------------------------------------------------------------------
  # 16. Return synthesis result
  # ------------------------------------------------------------------
  return {
      "L": L,
      "P0": P0_val,
      "P1": P1_val,
      "X": X_val,
      "gamma_L": g_L_val,
      "rho": float(rho),
      "alpha": float(alpha),
      "nu_bar": float(nu_bar),
      "h": float(h),
      "epsilon": epsilon,
      "eta": float(eta),
      "event_decay_rate": float(event_decay_rate),
      "min_eig_P0": float(np.min(eig_P0)),
      "min_eig_P_h": float(np.min(eig_P_h)),
      "min_eig_P_bar": float(np.min(eig_P_bar)),
      "solver_status": problem.status,
      "solver_value": problem.value,
  }
