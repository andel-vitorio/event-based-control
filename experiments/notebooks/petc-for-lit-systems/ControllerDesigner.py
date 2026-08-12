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
  Ξtil: cp.Variable
  Ψtil: cp.Variable
  X: cp.Variable
  Ytil: cp.Variable


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
    design_params: Dict[str, Any],
    eps: float = 1e-6,
    dtype=np.float32
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

    _h = float(design_params['h'])
    _υ = float(design_params['υ'])
    _λ = float(design_params['λ'])

    _A_param = cp.Parameter((_nx, _nx), value=mat_A)
    _B_param = cp.Parameter((_nx, _nu), value=mat_B)

    _e = nm.get_e(5 * [_nx])
    for idx in range(1, len(_e)):
      _e[idx] = cp.Parameter(_e[idx].shape, value=_e[idx].astype(dtype))

    return ProblemDataLIT(
        nx=_nx, nu=_nu, h=_h, υ=_υ, λ=_λ,
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
        Ytil=cp.Variable((2*nx, e_shape_1))
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


@dataclass
class ObserverProblemData:
  nx: int
  rho_o: float
  Ad: np.ndarray
  dtype: Any


@dataclass
class ObserverDecisionVariables:
  Po: cp.Variable
  Yo: cp.Variable
  gamma_o: cp.Variable


def synthesize_observer_gain(
    design_params: Dict[str, Any],
    eps: float = 1e-6,
    solver: Any = cp.MOSEK,
    dtype=np.float32
) -> Dict[str, Any]:
  """
  Sintetiza o ganho L do observador de tempo discreto e a taxa de 
  atenuação gamma_o via LMIs para um sistema Linear Invariante no Tempo (LIT).

  Estrutura esperada em `design_params`:
  {
      'Ad': np.ndarray,     # Matriz discreta de dinâmica A_d (nx x nx)
      'rho_o': float        # Fator de decaimento/convergência do observador rho_o
  }

  Retorna:
  --------
  Dict[str, Any] contendo:
      - 'L': Matriz de ganho do observador (nx x nx)
      - 'Po': Matriz de Lyapunov do observador Po
      - 'gamma_o': Valor ótimo do parâmetro de ruído/transmitância minimizado
  """

  def _create_problem_data() -> ObserverProblemData:
    mat_Ad = np.array(design_params['Ad'], dtype=dtype)
    _nx = mat_Ad.shape[0]
    _rho_o = float(design_params['rho_o'])

    return ObserverProblemData(
        nx=_nx,
        rho_o=_rho_o,
        Ad=mat_Ad,
        dtype=dtype
    )

  def _create_variables() -> ObserverDecisionVariables:
    nx = data.nx
    return ObserverDecisionVariables(
        Po=cp.Variable((nx, nx), PSD=True),
        Yo=cp.Variable((nx, nx)),
        gamma_o=cp.Variable(nonneg=True)
    )

  def _build_lmi_matrix() -> cp.Expression:
    nx = data.nx
    Po = vars.Po
    Yo = vars.Yo
    gamma_o = vars.gamma_o
    rho_o = data.rho_o
    Ad = data.Ad

    # [ rho_o * Po      0             Ad.T @ Po - Yo.T ]
    # [     0       gamma_o * I             -Yo.T      ]
    # [ Po @ Ad - Yo   -Yo                   Po        ]

    row1 = cp.hstack(
        [rho_o * Po, np.zeros((nx, nx), dtype=data.dtype), Ad.T @ Po - Yo.T])
    row2 = cp.hstack([np.zeros((nx, nx), dtype=data.dtype),
                     gamma_o * np.eye(nx, dtype=data.dtype), -Yo.T])
    row3 = cp.hstack([Po @ Ad - Yo, -Yo, Po])

    return cp.vstack([row1, row2, row3])

  def _recover_solution() -> Dict[str, Any]:
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
      return None

    dt = data.dtype
    Po_val = vars.Po.value.astype(dt)
    Yo_val = vars.Yo.value.astype(dt)
    gamma_o_val = float(vars.gamma_o.value)

    # Recuperação do ganho do observador: L = Po^{-1} @ Yo
    L = np.linalg.solve(Po_val, Yo_val)

    return {
        'optimal_value': problem.value,
        'gamma_o': gamma_o_val,
        'Po': Po_val,
        'Yo': Yo_val,
        'L': L
    }

  data = _create_problem_data()
  vars = _create_variables()

  LMI = _build_lmi_matrix()

  constraints = [
      LMI >> eps * np.eye(LMI.shape[0], dtype=dtype),
      vars.Po >> eps * np.eye(data.nx, dtype=dtype),
      vars.gamma_o >= eps
  ]

  obj = cp.Minimize(vars.gamma_o)
  problem = cp.Problem(obj, constraints)

  try:
    problem.solve(solver=solver, verbose=False, ignore_dpp=True)
  except cp.SolverError:
    problem.solve(solver=cp.MOSEK, verbose=False)

  return _recover_solution()
