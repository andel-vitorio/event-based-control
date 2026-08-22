from dataclasses import dataclass
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
from .Numeric import format_magnitudes


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


@dataclass
class PlotStyle:
  """
  Optional style container.

  Any attribute left as None falls back to the defaults
  defined inside plot().
  """

  linewidth: float | list | None = None
  linestyle: str | list | None = None

  tick_fontsize: int | None = None

  label_fontsize: int | None = None
  title_fontsize: int | None = None

  title_pad: float | None = None
  x_label_pad: float | None = None
  y_label_pad: float | None = None

  legend_fontsize: int | None = None
  legend_loc: str | None = None
  legend_ncol: int | None = None


def plot(
    ax, x, y, *,
    xlabel=None, ylabel=None, title=None, label=None,
    color=None, linestyle='-', linewidth=1.67,
    x_unit='', y_unit='',
    x_use_prefixes=False, y_use_prefixes=False,
    x_pad=(0.0, 0.0), y_pad=(0.0, 0.0),
    style: PlotStyle | None = None
):
  if style is None:
    style = PlotStyle()

  linewidth = style.linewidth if style.linewidth is not None else linewidth
  linestyle = style.linestyle if style.linestyle is not None else linestyle
  tick_fontsize = style.tick_fontsize if style.tick_fontsize is not None else 16
  label_fontsize = style.label_fontsize if style.label_fontsize is not None else 16
  title_fontsize = style.title_fontsize if style.title_fontsize is not None else 16
  title_pad = style.title_pad if style.title_pad is not None else 8
  x_label_pad = style.x_label_pad if style.x_label_pad is not None else 8
  y_label_pad = style.y_label_pad if style.y_label_pad is not None else 8
  legend_fontsize = style.legend_fontsize if style.legend_fontsize is not None else 12
  legend_loc = style.legend_loc if style.legend_loc is not None else 'best'
  legend_ncol = style.legend_ncol if style.legend_ncol is not None else 1

  # --------------------------------------------------
  # Helpers
  # --------------------------------------------------
  def normalize(value, n):
    if value is None:
      return [None] * n
    if isinstance(value, (list, tuple, np.ndarray)):
      if isinstance(value, tuple) and len(value) in (3, 4) and all(isinstance(v, (int, float)) for v in value):
        return [value] * n
      if len(value) == 0:
        return [None] * n
      return [value[i % len(value)] for i in range(n)]
    return [value] * n

  def apply_padding(values, pad):
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if np.isclose(vmin, vmax):
      return (vmin - 1e-3, vmax + 1e-3)
    vrange = vmax - vmin
    return (vmin - pad[0] * vrange, vmax + pad[1] * vrange)

  # --------------------------------------------------
  # X & Y data normalization
  # --------------------------------------------------
  x_arr = np.asarray(x, dtype=float).ravel()

  if isinstance(y, (list, tuple)):
    if len(y) > 0 and not isinstance(y[0], (list, tuple, np.ndarray)):
      y_arrays = [np.asarray(y, dtype=float).ravel()]
    else:
      y_arrays = [np.asarray(curve, dtype=float).ravel() for curve in y]
  else:
    y_tmp = np.asarray(y, dtype=float)
    if y_tmp.ndim == 2:
      if y_tmp.shape[1] == x_arr.size:
        y_arrays = list(y_tmp)
      elif y_tmp.shape[0] == x_arr.size:
        y_arrays = [y_tmp[:, i] for i in range(y_tmp.shape[1])]
      else:
        y_arrays = [y_tmp.ravel()]
    else:
      y_arrays = [y_tmp.ravel()]

  y_arrays = [curve for curve in y_arrays if curve.size == x_arr.size]
  if len(y_arrays) == 0:
    return []

  # --------------------------------------------------
  # Scaling & Style Normalization
  # --------------------------------------------------
  scaled_x, x_suffix, _ = format_magnitudes(
      x_arr, x_unit, x_use_prefixes, return_order=True)
  all_y = np.concatenate(y_arrays)
  _, y_suffix, y_order = format_magnitudes(
      all_y, y_unit, y_use_prefixes, return_order=True)

  y_scale = 10.0**(-y_order)
  scaled_x = np.asarray(scaled_x, dtype=float)

  n_curves = len(y_arrays)
  labels = normalize(label, n_curves)
  colors = normalize(color, n_curves)
  linestyles = normalize(linestyle, n_curves)
  linewidths = normalize(linewidth, n_curves)

  # --------------------------------------------------
  # Plot
  # --------------------------------------------------
  lines = []
  for i, curve in enumerate(y_arrays):
    curve_scaled = np.asarray(curve, dtype=float) * y_scale
    lines.extend(ax.plot(
        scaled_x, curve_scaled,
        label=labels[i], color=colors[i],
        linestyle=linestyles[i], linewidth=linewidths[i]
    ))

  # --------------------------------------------------
  # Limits
  # --------------------------------------------------
  scaled_all_y = all_y * y_scale
  ax.set_xlim(apply_padding(scaled_x, x_pad))

  if np.isclose(np.min(scaled_all_y), np.max(scaled_all_y)):
    value = scaled_all_y[0]
    delta = max(abs(value) * 0.1, 1e-3)
    ax.set_ylim(value - delta, value + delta)
  else:
    ax.set_ylim(apply_padding(scaled_all_y, y_pad))

  # --------------------------------------------------
  # Labels, Appearance & Legend
  # --------------------------------------------------
  if xlabel is not None:
    ax.set_xlabel(xlabel + x_suffix, fontsize=label_fontsize,
                  labelpad=x_label_pad)
  if ylabel is not None:
    ax.set_ylabel(ylabel + y_suffix, fontsize=label_fontsize,
                  labelpad=y_label_pad)
  if title is not None:
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)

  ax.grid(True, linestyle='--')
  ax.ticklabel_format(style='plain')
  ax.get_xaxis().get_major_formatter().set_useOffset(False)
  ax.get_yaxis().get_major_formatter().set_useOffset(False)
  ax.tick_params(
      axis='both', direction='in', length=4, width=1,
      top=True, right=True, labelsize=tick_fontsize
  )

  if any(lbl is not None for lbl in labels):
    ax.legend(
        frameon=True, framealpha=1, loc=legend_loc,
        ncol=legend_ncol, prop={'size': legend_fontsize}
    )

  return lines


def stem(
    ax, x, y, *,
    xlabel=None, ylabel=None, title=None, label=None,
    color=None, linewidth=1.67, marker_size=4.0, markerfmt='o', basefmt=' ', bottom=0.0,
    x_unit='', y_unit='',
    x_use_prefixes=False, y_use_prefixes=False,
    x_pad=(0.0, 0.0), y_pad=(0.0, 0.0),
    x_range=None,
    style: PlotStyle | None = None
):
  if style is None:
    style = PlotStyle()

  linewidth = style.linewidth if getattr(
      style, 'linewidth', None) is not None else linewidth
  marker_size = style.marker_size if getattr(
      style, 'marker_size', None) is not None else marker_size
  tick_fontsize = style.tick_fontsize if style.tick_fontsize is not None else 16
  label_fontsize = style.label_fontsize if style.label_fontsize is not None else 16
  title_fontsize = style.title_fontsize if style.title_fontsize is not None else 16
  title_pad = style.title_pad if style.title_pad is not None else 8
  x_label_pad = style.x_label_pad if style.x_label_pad is not None else 8
  y_label_pad = style.y_label_pad if style.y_label_pad is not None else 8
  legend_fontsize = style.legend_fontsize if style.legend_fontsize is not None else 12
  legend_loc = style.legend_loc if style.legend_loc is not None else 'best'
  legend_ncol = style.legend_ncol if style.legend_ncol is not None else 1

  # --------------------------------------------------
  # Helpers
  # --------------------------------------------------
  def normalize(value, n):
    if value is None:
      return [None] * n
    if isinstance(value, (list, tuple, np.ndarray)):
      if isinstance(value, tuple) and len(value) in (3, 4) and all(isinstance(v, (int, float)) for v in value):
        return [value] * n
      if len(value) == 0:
        return [None] * n
      return [value[i % len(value)] for i in range(n)]
    return [value] * n

  def apply_padding(values, pad, fixed_range=None):
    vmin, vmax = (float(fixed_range[0]), float(fixed_range[1])) if fixed_range is not None else (
        float(np.min(values)), float(np.max(values)))
    if np.isclose(vmin, vmax):
      return (vmin - 1e-3, vmax + 1e-3)
    vrange = vmax - vmin
    return (vmin - pad[0] * vrange, vmax + pad[1] * vrange)

  # --------------------------------------------------
  # X & Y data normalization
  # --------------------------------------------------
  x_arr = np.asarray(x, dtype=float).ravel()

  if isinstance(y, (list, tuple)):
    if len(y) > 0 and not isinstance(y[0], (list, tuple, np.ndarray)):
      y_arrays = [np.asarray(y, dtype=float).ravel()]
    else:
      y_arrays = [np.asarray(curve, dtype=float).ravel() for curve in y]
  else:
    y_tmp = np.asarray(y, dtype=float)
    if y_tmp.ndim == 2:
      if y_tmp.shape[1] == x_arr.size:
        y_arrays = list(y_tmp)
      elif y_tmp.shape[0] == x_arr.size:
        y_arrays = [y_tmp[:, i] for i in range(y_tmp.shape[1])]
      else:
        y_arrays = [y_tmp.ravel()]
    else:
      y_arrays = [y_tmp.ravel()]

  y_arrays = [curve for curve in y_arrays if curve.size == x_arr.size]
  if len(y_arrays) == 0:
    return []

  # --------------------------------------------------
  # Scaling & Style Normalization
  # --------------------------------------------------
  scaled_x, x_suffix, _ = format_magnitudes(
      x_arr, x_unit, x_use_prefixes, return_order=True)
  all_y = np.concatenate(y_arrays)
  _, y_suffix, y_order = format_magnitudes(
      all_y, y_unit, y_use_prefixes, return_order=True)

  y_scale = 10.0**(-y_order)
  scaled_x = np.asarray(scaled_x, dtype=float)

  n_curves = len(y_arrays)
  labels = normalize(label, n_curves)
  colors = normalize(color, n_curves)
  linewidths = normalize(linewidth, n_curves)
  marker_sizes = normalize(marker_size, n_curves)

  # --------------------------------------------------
  # Plot
  # --------------------------------------------------
  containers = []
  for i, curve in enumerate(y_arrays):
    curve_scaled = np.asarray(curve, dtype=float) * y_scale
    stem_kwargs = {'basefmt': basefmt, 'bottom': bottom}
    if labels[i] is not None:
      stem_kwargs['label'] = labels[i]

    container = ax.stem(
        scaled_x, curve_scaled,
        linefmt=colors[i] if colors[i] is not None else None,
        markerfmt=markerfmt,
        **stem_kwargs
    )
    if linewidths[i] is not None:
      plt.setp(container.stemlines, linewidth=linewidths[i])
    if marker_sizes[i] is not None:
      plt.setp(container.markerline, markersize=marker_sizes[i])
    if colors[i] is not None:
      plt.setp(container.stemlines, color=colors[i])
      plt.setp(container.markerline, color=colors[i])

    containers.append(container)

  # --------------------------------------------------
  # Limits
  # --------------------------------------------------
  scaled_all_y = all_y * y_scale
  ax.set_xlim(apply_padding(scaled_x, x_pad, fixed_range=x_range))

  if np.isclose(np.min(scaled_all_y), np.max(scaled_all_y)):
    value = scaled_all_y[0]
    delta = max(abs(value) * 0.1, 1e-3)
    ax.set_ylim(value - delta, value + delta)
  else:
    ax.set_ylim(apply_padding(scaled_all_y, y_pad))

  # --------------------------------------------------
  # Labels, Appearance & Legend
  # --------------------------------------------------
  if xlabel is not None:
    ax.set_xlabel(xlabel + x_suffix, fontsize=label_fontsize,
                  labelpad=x_label_pad)
  if ylabel is not None:
    ax.set_ylabel(ylabel + y_suffix, fontsize=label_fontsize,
                  labelpad=y_label_pad)
  if title is not None:
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)

  ax.grid(True, linestyle='--')
  ax.ticklabel_format(style='plain')
  ax.get_xaxis().get_major_formatter().set_useOffset(False)
  ax.get_yaxis().get_major_formatter().set_useOffset(False)
  ax.tick_params(
      axis='both', direction='in', length=4, width=1,
      top=True, right=True, labelsize=tick_fontsize
  )

  if any(lbl is not None for lbl in labels):
    ax.legend(
        frameon=True, framealpha=1, loc=legend_loc,
        ncol=legend_ncol, prop={'size': legend_fontsize}
    )

  return containers


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
