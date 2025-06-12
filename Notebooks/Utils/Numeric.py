from typing import Callable, Optional, Union, Dict
import itertools
import matplotlib.pyplot as plt
from typing import List, Sequence, Tuple, Union
import numpy as np
from typing import List, Tuple
from scipy.spatial import ConvexHull


def binary_pairs(p: int) -> List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
  """
  Generates all unique pairs of binary tuples of length `p`,
  including identical pairs.

  Each pair is represented as a list of one or two tuples:
  - [(a, a)] for identical elements
  - [(a, b), (b, a)] for distinct elements (symmetric pair)

  Parameters
  ----------
  p : int
      The number of bits in each binary tuple.

  Returns
  -------
  List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
      A list of binary tuple pairs, grouped by identity or symmetry.
  """
  num_elements = 2 ** p
  result = []

  for i in range(num_elements):
    bin_i = tuple(int(b) for b in f"{i:0{p}b}")
    result.append([(bin_i, bin_i)])  # identical pair
    for j in range(i + 1, num_elements):
      bin_j = tuple(int(b) for b in f"{j:0{p}b}")
      result.append([(bin_i, bin_j), (bin_j, bin_i)])  # symmetric pair

  return result


def get_vertices(x_bounds: Union[List[Tuple[float, float]], np.ndarray]) -> List[np.ndarray]:
  """
  Returns the set of vectors a_q ∈ R^{n×1} such that the box-constrained polyhedral set
  D = {x ∈ R^n : x_i_min ≤ x_i ≤ x_i_max for all i}
  can be rewritten as D = {x ∈ R^n : a_q.T @ x ≤ 1 for all q}.

  Parameters
  ----------
  x_bounds : list or array-like of shape (n, 2)
      Each element is a tuple or list (a, b) representing the lower and upper
      bounds for the i-th component of x (in any order).

  Returns
  -------
  aq_list : list of numpy.ndarray
      A list with 2n numpy arrays, each of shape (n, 1), corresponding to a_q vectors.
  """
  x_bounds = np.array(x_bounds)
  n = x_bounds.shape[0]
  aq_list: List[np.ndarray] = []

  for i in range(n):
    x_min, x_max = sorted(x_bounds[i])

    a_max = np.zeros((n, 1))
    a_max[i, 0] = 1 / x_max
    aq_list.append(a_max)

    a_min = np.zeros((n, 1))
    a_min[i, 0] = -1 / x_min
    a_min_scaled = a_min / -1
    aq_list.append(a_min_scaled)

  return aq_list


def _intersection_of_lines(a_q):
  """
  Computes the intersection points of the lines represented by a_q.T @ x = 1
  for each pair of vectors a_q.

  Parameters:
  ----------
  a_q : list of numpy.ndarray
      A list of numpy arrays representing the hyperplanes a_q.T @ x = 1.

  Returns:
  -------
  intersections : numpy.ndarray
      A numpy array of shape (m, 2) where each row represents the intersection
      of two hyperplanes.
  """
  intersections = []
  for i in range(len(a_q)):
    for j in range(i + 1, len(a_q)):
      # Flatten to ensure proper shape
      A = np.array([a_q[i].flatten(), a_q[j].flatten()]).T
      b = np.array([1, 1])
      if np.linalg.matrix_rank(A) == 2:
        x = np.linalg.solve(A, b)
        intersections.append(x)
  return np.array(intersections)


def get_polyhedral_set(a_q, x_bounds, margin=1.0, num_points=400):
  """
  Generates data for plotting the lines a_q^T x = 1, their intersections,
  and the polygon formed by their intersection (if it exists).

  Parameters:
  ----------
  a_q : list of numpy.ndarray
      A list of numpy arrays representing the hyperplanes a_q.T @ x = 1.
  x_bounds : list of (min, max) tuples
      Bounds for each variable, used to define plot limits.
  margin : float, optional
      Additional range beyond bounds for visualization purposes.
  num_points : int, optional
      Number of points per line segment.

  Returns:
  -------
  dict
      A dictionary with the following keys:
      - "line_segments" : list of (x_vals, y_vals) tuples for each line
      - "intersection_points" : array of intersection points between lines
      - "polygon_coords" : coordinates of the polygon formed by the intersections (if any)
  """
  # Sorting bounds
  x_bounds = np.array(x_bounds)
  x1_min, x1_max = sorted(x_bounds[0])
  x2_min, x2_max = sorted(x_bounds[1])

  # Define the range for plotting
  x1_range = np.linspace(x1_min - margin, x1_max + margin, num_points)
  x2_range = np.linspace(x2_min - margin, x2_max + margin, num_points)

  line_segments = []

  for a in a_q:
    a = a.flatten()  # Ensure a is a 1D array for easier indexing
    if np.count_nonzero(a) == 1:
      # Only one non-zero coefficient
      idx = np.argmax(np.abs(a))
      x_fixed = 1 / a[idx]
      if idx == 0:
        segment = ([x_fixed, x_fixed], [x2_range[0], x2_range[-1]])
      else:
        segment = ([x1_range[0], x1_range[-1]], [x_fixed, x_fixed])
    else:
      # General case for oblique lines
      if a[1] != 0:
        x2_vals = (1 - a[0] * x1_range) / a[1]
        segment = (x1_range, x2_vals)
      else:
        x_fixed = 1 / a[0]
        segment = ([x_fixed, x_fixed], [x2_range[0], x2_range[-1]])
    line_segments.append(segment)

  # Calculate intersection points
  intersection_points = _intersection_of_lines(a_q)

  # Determine the polygon formed by the intersection points
  polygon_coords = None
  if intersection_points.size > 0:
    try:
      hull = ConvexHull(intersection_points)
      polygon_coords = intersection_points[hull.vertices]
    except:
      polygon_coords = None

  return {
      "line_segments": line_segments,
      "intersection_points": intersection_points,
      "polygon_coords": polygon_coords
  }


def get_ellipsoid_boundary(P, beta, num_points=100):
  """
  Computes the boundary of a 2D ellipsoidal region defined by the quadratic form
  x.T @ P @ x = beta, and returns the data needed to plot its boundary.

  Parameters:
  -----------
  P : numpy.ndarray, shape (2, 2)
      A 2x2 matrix defining the quadratic form for the ellipsoid.

  beta : float
      The constant that defines the boundary of the ellipsoid (x.T @ P @ x = beta).

  num_points : int
      The number of points to compute along the boundary.

  Returns:
  --------
  boundary_data : dict
      A dictionary containing the following keys:
      - "x1_vals": The x1 coordinates of the boundary.
      - "x2_vals": The x2 coordinates of the boundary.
  """
  # Eigenvalue decomposition of P to get the scaling factors
  eigvals, eigvecs = np.linalg.eig(P)

  # A parameterization of the ellipse using theta
  theta = np.linspace(0, 2 * np.pi, num_points)

  # Compute the ellipse's boundary points using the parametrization
  a = np.sqrt(beta / eigvals[0])  # Scaling for x1 direction
  b = np.sqrt(beta / eigvals[1])  # Scaling for x2 direction

  # Parametric equations for the ellipse
  x1_vals = a * np.cos(theta)
  x2_vals = b * np.sin(theta)

  # Rotate the points according to the eigenvectors of P
  rotated_points = np.dot(np.vstack((x1_vals, x2_vals)).T, eigvecs.T)

  # Return the boundary data
  return {
      "x1_vals": rotated_points[:, 0],
      "x2_vals": rotated_points[:, 1]
  }


def get_region_boundary(V_func, x1_range=(-0.2, 0.2), x2_range=(-0.2, 0.2), level=0, num_points=100):
  """
   Calculates the data for plotting the level set of the function V(x) = 0.
   This function computes the values of V(x) over a grid defined by x1_range and x2_range.

   Parameters:
   -----------
   V_func : callable
       A function that computes V(x) for given x. It must accept 2D numpy arrays as inputs.

   x1_range : tuple of floats, optional
       The range of values for the x1 dimension. Default is (-0.2, 0.2).

   x2_range : tuple of floats, optional
       The range of values for the x2 dimension. Default is (-0.2, 0.2).

   level : float, optional
       The level at which to calculate the contour. Default is 0.

   num_points : int, optional
       The number of points along each axis to generate. Default is 100.

   Returns:
   --------
   X1, X2 : numpy.ndarray
       Grids of x1 and x2 coordinates.

   V_values : numpy.ndarray
       Values of V(x) on the grid.

   contour_data : tuple
       Contour data for plotting, including X1, X2, and V_values.
   """
  # Geração de pontos no espaço de estados
  x1 = np.linspace(x1_range[0], x1_range[1], num_points)
  x2 = np.linspace(x2_range[0], x2_range[1], num_points)
  X1, X2 = np.meshgrid(x1, x2)

  # Calcula os valores de V(x) para todos os pontos (X1, X2)
  V_values = V_func(X1, X2)

  # Dados do contorno para plotagem
  contour_data = (X1, X2, V_values)

  return X1, X2, V_values, contour_data


def binary_set(n):
  return list(itertools.product([0, 1], repeat=n))


def ellipsoid_boundary_points(P: np.ndarray, beta: float, num_points: int = 100) -> np.ndarray:
  """
  Generate uniformly spaced points on the boundary of a 2D ellipsoid defined by x.T @ P @ x = beta.

  Args:
      P (np.ndarray): A 2x2 positive definite matrix defining the ellipsoid shape.
      beta (float): A positive scalar defining the ellipsoid size.
      num_points (int): Number of points to generate on the boundary.

  Returns:
      np.ndarray: A 2xN array of points lying on the ellipsoid boundary.
  """
  eigvals, eigvecs = np.linalg.eigh(P)

  angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
  circle_points = np.array([np.cos(angles), np.sin(angles)])

  scaling_matrix = np.diag(1 / np.sqrt(eigvals))
  points = eigvecs @ scaling_matrix @ circle_points * np.sqrt(beta)

  return points


def diag(values: Sequence[float]) -> np.ndarray:
  """
  Creates a diagonal matrix with the given real values on the main diagonal.

  Parameters
  ----------
  values : Sequence[float]
      A sequence (e.g., list, tuple, or NumPy array) of real numbers to be placed on the main diagonal of the matrix.

  Returns
  -------
  np.ndarray
      A square NumPy array with the input values on its main diagonal and zeros elsewhere.

  Examples
  --------
  >>> diag([1.0, 2.0, 3.0])
  array([[1., 0., 0.],
         [0., 2., 0.],
         [0., 0., 3.]])
  """
  return np.diag(values)


def rk5_step(
    f: Callable[[float, np.ndarray, np.ndarray, Optional[Dict]], np.ndarray],
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    T_s: float,
    params: Optional[Dict] = None
) -> np.ndarray:
  """
  Performs a single fixed-step integration using the Runge-Kutta method of order 5 (RK5).

  This implementation uses a Dormand-Prince-like formulation, without embedded error control,
  and advances the state `x` over a fixed step `T_s`.

  Parameters
  ----------
  f : Callable[[float, np.ndarray, np.ndarray, Optional[dict]], np.ndarray]
      System dynamics function with signature `f(t, x, u, params)`, where:
          - t : float
              Current simulation time.
          - x : np.ndarray
              State vector.
          - u : np.ndarray
              Control input vector.
          - params : Optional[dict]
              Optional dictionary of parameters for the dynamics.

  t : float
      Current time.

  x : np.ndarray
      Current state vector.

  u : np.ndarray
      Input vector at time `t`.

  T_s : float
      Integration step size (sampling period).

  params : dict, optional
      Dictionary of additional parameters passed to the function `f`.

  Returns
  -------
  np.ndarray
      Estimated state vector at time `t + T_s`.

  Notes
  -----
  This implementation assumes constant input `u` over the integration interval
  [t, t + T_s], and is intended for use in fixed-step simulation environments.

  References
  ----------
  Dormand, J. R., & Prince, P. J. (1980). A family of embedded Runge-Kutta formulae.
  Journal of Computational and Applied Mathematics, 6(1), 19–26.
  """
  k1 = f(t, x, u, params)
  k2 = f(t, x + T_s * k1 / 4, u, params)
  k3 = f(t, x + T_s * (3*k1 + 9*k2) / 32, u, params)
  k4 = f(t, x + T_s * (1932*k1 - 7200*k2 + 7296*k3) / 2197, u, params)
  k5 = f(t, x + T_s * (439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104), u, params)
  k6 = f(t, x - T_s * (8*k1/27 - 2*k2 + 3544*k3 /
         2565 - 1859*k4/4104 + 11*k5/40), u, params)

  return x + T_s * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)


def get_e(lines_number: List[int]) -> Dict[int, np.ndarray]:
  """
  Splits an identity matrix into consecutive row blocks according to the provided sizes.

  Args:
      lines_number (List[int]): A list of integers where each value represents 
                                the number of rows in the corresponding block.

  Returns:
      Dict[int, np.ndarray]: A dictionary mapping block indices (starting from 1) 
                             to their corresponding slices of the identity matrix.
  """
  total_columns = sum(lines_number)
  identity = np.eye(total_columns)
  slices = {}
  start = 0
  for idx, size in enumerate(lines_number, start=1):
    slices[idx] = identity[start:start + size]
    start += size
  return slices


def matrix_definiteness(A: np.ndarray, tol: float = 1e-8) -> str:
  """
  Determines the definiteness of a symmetric matrix.

  Args:
      A (np.ndarray): Square symmetric matrix to classify.
      tol (float): Numerical tolerance for zero comparisons (default 1e-8).

  Returns:
      str: One of {'pd', 'psd', 'nd', 'nsd', 'und'} representing the definiteness.
           - 'pd' : Positive definite
           - 'psd': Positive semidefinite
           - 'nd' : Negative definite
           - 'nsd': Negative semidefinite
           - 'ind': Indefinite
  """
  # Ensure the matrix is symmetric to avoid numerical issues
  A_sym = (A + A.T) / 2
  eigenvalues = np.linalg.eigvalsh(A_sym)

  if np.all(eigenvalues > tol):
    return 'pd'
  elif np.all(eigenvalues >= -tol):
    return 'psd'
  elif np.all(eigenvalues < -tol):
    return 'nd'
  elif np.all(eigenvalues <= tol):
    return 'nsd'
  else:
    return 'ind'
