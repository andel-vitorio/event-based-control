from typing import Optional, Sequence, Tuple, Dict, Any, Union
from matplotlib.ticker import FuncFormatter
from typing import Any, Dict, Optional, Tuple
import math
from typing import Any, Dict, Optional
import matplotlib.tri as tri
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Numeric import format_magnitudes


def use_latex():
  """
  Configures Matplotlib to use LaTeX if available.
  If LaTeX is not available, uses default fonts.
  """
  try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Palatino"
    })
    plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs}'
    print("LaTeX has been enabled for text rendering.")
  except Exception:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif"
    })
    print("LaTeX is not available. Using default fonts.")


def plot(ax, x_data, y_data,
         xlabel=None, ylabel=None, title=None, label='',
         cfg={}, *,
         x_unit='', y_unit='',
         x_use_prefixes=False, y_use_prefixes=False,
         x_pad=(0.0, 0.0), y_pad=(0.0, 0.0)):

  import numpy as np

  # normalizar curvas
  if isinstance(y_data, (list, tuple)):
    if len(y_data) > 0 and not isinstance(y_data[0], (list, tuple, np.ndarray)):
      y_arrays = [np.asarray(y_data, dtype=float).ravel()]
    else:
      y_arrays = [np.asarray(y, dtype=float).ravel() for y in y_data]
  else:
    y_arrays = [np.asarray(y_data, dtype=float).ravel()]

  # x em 1D
  x_arr = np.asarray(x_data, dtype=float).ravel()

  # remover curvas inconsistentes
  y_arrays = [y for y in y_arrays if y.size == x_arr.size]

  if len(y_arrays) == 0:
    return [], None

  # aplicar format_magnitudes em x
  scaled_x, x_label_suffix, x_order = format_magnitudes(
      x_arr, x_unit, x_use_prefixes, return_order=True)

  # concatenar y e formatar
  all_y_values = np.concatenate(y_arrays)
  scaled_y_all, y_label_suffix, y_order = format_magnitudes(
      all_y_values, y_unit, y_use_prefixes, return_order=True)

  scaled_x = np.asarray(scaled_x, dtype=float)
  scaled_y_all = np.asarray(scaled_y_all, dtype=float)

  scale_factor_y = 10 ** (-y_order)

  # configurações
  style = cfg.get('style', {})
  axis_cfg = cfg.get('axis', {})
  legend_cfg = cfg.get('legend', {})

  def normalize(param, n):
    if isinstance(param, (list, tuple, np.ndarray)):
      if len(param) in (3, 4) and all(isinstance(x, (float, int)) for x in param):
        return [param] * n
      if len(param) == n:
        return list(param)
      return [param[0]] * n
    return [param] * n

  n_curves = len(y_arrays)
  labels = label if isinstance(label, (list, tuple)) else [label] * n_curves
  colors = normalize(style.get('color', 'black'), n_curves)
  linestyles = normalize(style.get('linestyle', '-'), n_curves)
  linewidths = normalize(style.get('linewidth', 1.67), n_curves)

  lines = []

  # plotar
  for i, y_arr in enumerate(y_arrays):
    y_scaled = np.asarray(y_arr, dtype=float) * scale_factor_y
    line = ax.plot(scaled_x, y_scaled,
                   label=labels[i],
                   color=colors[i],
                   linewidth=linewidths[i],
                   linestyle=linestyles[i])
    lines += line

  # função de padding
  def apply_padding(vals, pad):
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if np.isclose(vmin, vmax):
      return vmin - 1e-3, vmax + 1e-3
    vr = vmax - vmin
    return vmin - pad[0] * vr, vmax + pad[1] * vr

  # limites Y
  if np.isclose(np.min(all_y_values), np.max(all_y_values)):
    const_scaled = scaled_y_all[0]
    delta = max(abs(const_scaled) * 0.1, 1e-3)
    ax.set_ylim(const_scaled - delta, const_scaled + delta)
  else:
    ax.set_ylim(apply_padding(scaled_y_all, y_pad))

  # limites X
  ax.set_xlim(apply_padding(scaled_x, x_pad))

  # ---------- RÓTULOS ----------
  if xlabel is not None:
    ax.set_xlabel(
        xlabel + x_label_suffix,
        fontsize=axis_cfg.get('x_label_fontsize', 16),
        labelpad=axis_cfg.get('x_label_pad', 8)
    )

  if ylabel is not None:
    ax.set_ylabel(
        ylabel + y_label_suffix,
        fontsize=axis_cfg.get('y_label_fontsize', 16),
        labelpad=axis_cfg.get('y_label_pad', 8)
    )

  # ---------- ESTILO ----------
  ax.grid(linestyle='--')
  ax.ticklabel_format(style='plain')
  ax.get_xaxis().get_major_formatter().set_useOffset(False)
  ax.get_yaxis().get_major_formatter().set_useOffset(False)

  ax.tick_params(
      axis='both',
      direction='in', length=4, width=1,
      colors='black', top=True, right=True,
      labelsize=axis_cfg.get('tick_fontsize', 16)
  )

  if any(labels):
    ax.legend(
        frameon=True,
        loc=legend_cfg.get('loc', 'best'),
        ncol=legend_cfg.get('ncol', 1),
        framealpha=1,
        prop={'size': legend_cfg.get('fontsize', 12)}
    )

  return lines, ((scaled_x, x_label_suffix, x_order),
                 (scaled_y_all, y_label_suffix, y_order))


def stem(ax: Axes,
         x_data,
         y_data,
         xlabel: Optional[str] = None,
         ylabel: Optional[str] = None,
         title: Optional[str] = None,
         label: str = '',
         cfg: Dict[str, Any] = {},
         *,
         x_unit: str = '',
         y_unit: str = '',
         x_use_prefixes: bool = False,
         y_use_prefixes: bool = False,
         x_pad: Tuple[float, float] = (0.0, 0.0),
         y_pad: Tuple[float, float] = (0.0, 0.0),
         x_range: Optional[Tuple[float, float]] = None,
         reuse_previous: bool = True) -> Tuple[Any, Any, Any]:
  """
  Plots a stem graph with automatic scaling, formatting, and optional fixed x-range.

  Parameters
  ----------
  x_range : tuple(float, float), optional
      If provided, defines the fixed (x_min, x_max) range for analysis
      instead of using min(x_data) and max(x_data).
  reuse_previous : bool, optional
      If True, reuses previous axis configuration when adding new data.
  """

  style = cfg.get('style', {})
  axis = cfg.get('axis', {})
  legend_cfg = cfg.get('legend', {})

  color = style.get('color', 'black')
  stem_width = style.get('linewidth', 1.67)
  marker_size = style.get('marker_size', 4)

  x_label_fontsize = axis.get('x_label_fontsize', 16)
  y_label_fontsize = axis.get('y_label_fontsize', 16)
  tick_fontsize = axis.get('tick_fontsize', 16)
  x_label_pad = axis.get('x_label_pad', 8)
  y_label_pad = axis.get('y_label_pad', 8)
  title_pad = axis.get('title_pad', 8)

  # --- Handle axis reuse ----------------------------------------------------
  prev_xlim = ax.get_xlim()
  prev_ylim = ax.get_ylim()
  prev_xlabel = ax.get_xlabel()
  prev_ylabel = ax.get_ylabel()
  prev_title = ax.get_title()

  if reuse_previous and hasattr(ax, "_scale_info"):
    scale_info = ax._scale_info
    x_multiplier = scale_info["x_multiplier"]
    y_multiplier = scale_info["y_multiplier"]
    x_label = scale_info["x_label"]
    y_label = scale_info["y_label"]
    x_exp = scale_info["x_exp"]
    y_exp = scale_info["y_exp"]

    scaled_x = [v * x_multiplier for v in x_data]
    scaled_y = [v * y_multiplier for v in y_data]
  else:
    n_divs = max(5, len(np.unique(x_data)) - 1)
    scaled_x, x_label, _ = format_magnitudes(
        x_data, x_unit, x_use_prefixes, n_divs)
    scaled_y, y_label, _ = format_magnitudes(
        y_data, y_unit, y_use_prefixes, n_divs)

    def get_exp(multiplier: float) -> int:
      return int(round(math.log10(1 / multiplier))) if multiplier != 0 else 0

    x_multiplier = (10 ** (-get_exp(max(abs(max(x_data)), 1e-12))))
    y_multiplier = (10 ** (-get_exp(max(abs(max(y_data)), 1e-12))))
    x_exp = get_exp(1 / x_multiplier)
    y_exp = get_exp(1 / y_multiplier)

    ax._scale_info = {
        "x_multiplier": x_multiplier,
        "y_multiplier": y_multiplier,
        "x_label": x_label,
        "y_label": y_label,
        "x_exp": x_exp,
        "y_exp": y_exp
    }

  # --- Create stem plot -----------------------------------------------------
  markerline, stemlines, baseline = ax.stem(
      scaled_x, scaled_y, linefmt=color, markerfmt='o',
      basefmt=' ', bottom=0, label=label
  )
  plt.setp(stemlines, 'linewidth', stem_width)
  plt.setp(markerline, 'markersize', marker_size)

  # --- Compute automatic or fixed paddings ---------------------------------
  def apply_padding(values, pad, fixed_range=None):
    if fixed_range is not None:
      vmin, vmax = fixed_range
    else:
      vmin, vmax = np.min(values), np.max(values)
    if vmin == vmax:
      return vmin - 1, vmax + 1
    range_ = vmax - vmin
    return vmin - pad[0] * range_, vmax + pad[1] * range_

  if not reuse_previous:
    xlim = apply_padding(scaled_x, x_pad, x_range)
    ylim = apply_padding(scaled_y, y_pad)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

  # --- Labels and formatting ------------------------------------------------
  if xlabel is not None:
    ax.set_xlabel(xlabel + x_label, fontsize=x_label_fontsize,
                  labelpad=x_label_pad)
  elif not reuse_previous:
    ax.set_xlabel(prev_xlabel)

  if ylabel is not None:
    ax.set_ylabel(ylabel + y_label, fontsize=y_label_fontsize,
                  labelpad=y_label_pad)
  elif not reuse_previous:
    ax.set_ylabel(prev_ylabel)

  if title is not None:
    ax.set_title(title, fontsize=tick_fontsize, pad=title_pad)
  elif not reuse_previous:
    ax.set_title(prev_title)

  ax.grid(linestyle='--')
  ax.tick_params(axis='both', direction='in', length=4, width=1,
                 colors='black', top=True, right=True, labelsize=tick_fontsize)

  if label:
    legend_size = legend_cfg.get('fontsize', 12)
    ax.legend(frameon=True, loc='best', framealpha=1,
              prop={'size': legend_size})

  return markerline, stemlines, baseline


def plot_2d_projection(X, Y, Z, x_min=None, x_max=None, levels=25, cmap='magma',
                       labels=['X', 'Y', 'Z'], title=None, show_colorbar=True,
                       solid_contours=False, alpha=1.0, contour_linewidth=1.5):
  """
  Generates a 2D plot of Z as a function of X and Y using triangulation without interpolation,
  with various customization options.

  Parameters:
      X : numpy.ndarray
          1D array containing the values for the X variable.
      Y : numpy.ndarray
          1D array containing the values for the Y variable.
      Z : numpy.ndarray
          1D array containing the values for the Z variable.
      x_min : float, optional
          Minimum value for the X range to consider (if None, uses min(X)).
      x_max : float, optional
          Maximum value for the X range to consider (if None, uses max(X)).
      levels : int, optional
          Number of contour levels.
      cmap : str, optional
          Colormap to use for the plot (default is 'magma').
      labels : list of str, optional
          Labels for the axes and colorbar, in the order [X, Y, Z].
      title : str, optional
          Title for the plot.
      show_colorbar : bool, optional
          Whether to display the colorbar.
      solid_contours : bool, optional
          If True, the contours will be solid, otherwise, they will be filled.
      alpha : float, optional
          Transparency of the plot, ranges from 0 (fully transparent) to 1 (fully opaque).
      contour_linewidth : float, optional
          Line width of the contours if `solid_contours` is True.

  Returns:
      None
  """
  # Set default limits if not provided
  if x_min is None:
    x_min = np.min(X)
  if x_max is None:
    x_max = np.max(X)

  # Filter points within the specified range
  mask = (X >= x_min) & (X <= x_max)
  X_filtered, Y_filtered, Z_filtered = X[mask], Y[mask], Z[mask]

  # Check if there are enough points for triangulation
  if len(X_filtered) < 3:
    print("Insufficient points for triangulation.")
    return

  # Create triangulation
  triang = tri.Triangulation(X_filtered, Y_filtered)

  # Create the plot
  plt.figure(figsize=(8, 6))

  if solid_contours:
    contour = plt.tricontour(
        triang, Z_filtered, levels=levels, cmap=cmap, linewidths=contour_linewidth)
  else:
    contour = plt.tricontourf(
        triang, Z_filtered, levels=levels, cmap=cmap, alpha=alpha)

  # Set title if provided
  if title:
    plt.title(title, fontsize=22)

  # Set axis labels
  plt.xlabel(labels[0], fontsize=20, labelpad=12)
  plt.ylabel(labels[1], fontsize=20, labelpad=12)
  plt.grid(linestyle='--', c='#121212', linewidth=0.5)

  # Adjust ticks
  plt.tick_params(axis='both', direction='in', length=4, width=1,
                  colors='black', top=True, right=True, labelsize=18)

  # Access current axes to modify offset labels
  ax = plt.gca()
  ax.xaxis.get_offset_text().set_fontsize(20)
  ax.yaxis.get_offset_text().set_fontsize(20)

  # Add colorbar
  if show_colorbar:
    cbar = plt.colorbar(contour, pad=0.02)
    cbar.set_label(labels[2], fontsize=20, labelpad=12)
    cbar.ax.tick_params(labelsize=16)

  plt.tight_layout()
  plt.show()


def create_custom_legend(
    ax: plt.Axes,
    elements: List[Tuple[str, dict]],
    labels: List[str],
    fontsize: int = 14,
    loc: str = "lower left",
    frameon: bool = True
) -> None:
  """
  Create a manual legend with customizable line and marker elements.

  Args:
      ax (plt.Axes): The matplotlib axes to attach the legend to.
      elements (List[Tuple[str, dict]]): A list of tuples where each tuple contains:
          - The type of element ('line', 'marker')
          - A dictionary of keyword arguments to pass to the corresponding matplotlib constructor
      labels (List[str]): A list of legend labels corresponding to each element.
      fontsize (int): Font size of the legend text.
      loc (str): Location of the legend on the plot.
      frameon (bool): Whether to draw a frame around the legend.
  """
  handles = []
  for element_type, kwargs in elements:
    if element_type == 'line':
      handle = plt.Line2D([0], [0], **kwargs)
    elif element_type == 'marker':
      handle, = ax.plot([0], [0], **kwargs)
    else:
      raise ValueError(f"Unsupported element type: {element_type}")
    handles.append(handle)

  ax.legend(handles, labels, fontsize=fontsize, loc=loc, frameon=frameon)


def _solve_intersection(A_inv, C):
  """
  Auxiliary function to compute the intersection point x = A_inv * C.

  Parameters
  ----------
  A_inv : np.ndarray
      Inverse of the 2×2 coefficient matrix A.
  C : np.ndarray
      Right-hand side vector of shape (2,).

  Returns
  -------
  np.ndarray
      Flattened intersection point [x1, x2].
  """
  x = A_inv @ C.reshape(2, 1)
  return x.flatten()


def get_parallelogram2D_vertices(H1, H2, mag):
  """
  Computes the four ordered vertices of a 2D parallelogram formed
  by the intersection of two directional constraint sets (corridors).

  Parameters
  ----------
  H1 : np.ndarray
      First direction vector (1×2 or 2×1) defining a pair of parallel lines.
  H2 : np.ndarray
      Second direction vector (1×2 or 2×1) defining another pair of parallel lines.
  mag : float
      Magnitude that defines the distance of each parallel line from the origin.

  Returns
  -------
  tuple[np.ndarray, np.ndarray]
      ordered_vertices : np.ndarray
          Array with the four 2D vertices ordered counterclockwise.
      centroid : np.ndarray
          Centroid (geometric center) of the parallelogram.

  Notes
  -----
  - The parallelogram is defined by the intersections of the four combinations
    of the lines H1·x = ±u_bar and H2·x = ±u_bar.
  - Vertices are ordered based on their polar angles relative to the centroid.
  """
  # 1. Build the coefficient matrix A
  A = np.array([H1.flatten(), H2.flatten()])

  # 2. Check if the two lines are parallel (determinant ≈ 0)
  det_A = np.linalg.det(A)
  if np.abs(det_A) < 1e-9:
    print(f"Error: The lines defined by H1={H1} and H2={H2} are parallel.")
    return None, None

  # 3. Compute the inverse of A
  A_inv = np.linalg.inv(A)

  # 4. Define the four right-hand side vectors (line offsets)
  C_vectors = [
      np.array([mag, mag]),
      np.array([mag, -mag]),
      np.array([-mag, mag]),
      np.array([-mag, -mag])
  ]

  # 5. Compute the four intersection points (vertices)
  vertices = np.array([
      _solve_intersection(A_inv, C_vectors[0]),
      _solve_intersection(A_inv, C_vectors[1]),
      _solve_intersection(A_inv, C_vectors[2]),
      _solve_intersection(A_inv, C_vectors[3])
  ])

  # 6. Order vertices counterclockwise
  centroid = vertices.mean(axis=0)
  angles = np.arctan2(vertices[:, 1] - centroid[1],
                      vertices[:, 0] - centroid[0])
  ordered_vertices = vertices[np.argsort(angles)]

  return ordered_vertices, centroid
