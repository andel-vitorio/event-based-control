import matplotlib.tri as tri
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pyplot as plt


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


def plot(ax: Axes,
         x_data,
         y_data,
         xlabel: Optional[str] = None,
         ylabel: Optional[str] = None,
         title: Optional[str] = None,
         label: str = '',
         plot_cfg: Dict[str, Any] = {}) -> Any:
  """
  Plots a 2D line on the given Axes object using an external structured configuration dictionary.

  Parameters
  ----------
  ax : matplotlib.axes.Axes
      The Matplotlib Axes object to draw the plot on.
  x_data : array_like
      X-axis data.
  y_data : array_like
      Y-axis data.
  xlabel : str, optional
      Label for the x-axis.
  ylabel : str, optional
      Label for the y-axis.
  title : str, optional
      Title of the plot.
  label : str, optional
      Legend label for the plotted line.
  plot_cfg : dict, optional
      Dictionary with structured configuration options. Expected keys:

      - 'style':
          - 'color' (str)
          - 'linewidth' (float)
          - 'linestyle' (str)

      - 'axis':
          - 'x_digits' (int)
          - 'y_digits' (int)
          - 'x_label_pad' (int)
          - 'y_label_pad' (int)
          - 'title_pad' (int)

      - 'limits':
          - 'x_min', 'x_max' (float)
          - 'y_min', 'y_max' (float)

      - 'ticks':
          - 'x_tick_interval' (float)
          - 'y_tick_interval' (float)

      - 'legend':
          - 'fontsize' (int)

  Returns
  -------
  line : Line2D
      The line object created by the plot.
  """
  style = plot_cfg.get('style', {})
  axis = plot_cfg.get('axis', {})
  limits = plot_cfg.get('limits', {})
  ticks = plot_cfg.get('ticks', {})
  legend_cfg = plot_cfg.get('legend', {})

  # Extract style parameters
  color = style.get('color', 'black')
  linewidth = style.get('linewidth', 1.67)
  linestyle = style.get('linestyle', '-')

  # Extract axis formatting
  x_digits = axis.get('x_digits', 1)
  y_digits = axis.get('y_digits', 1)
  x_label_pad = axis.get('x_label_pad', 8)
  y_label_pad = axis.get('y_label_pad', 8)
  x_label_fontsize = axis.get('x_label_fontsize', 16)
  y_label_fontsize = axis.get('y_label_fontsize', 16)
  tick_fontsize = axis.get('tick_fontsize', 16)
  title_pad = axis.get('title_pad', 20)

  # Tick formatting
  ax.xaxis.set_major_formatter(
      FuncFormatter(lambda v, _: f'{v:.{x_digits}f}'))
  ax.yaxis.set_major_formatter(
      FuncFormatter(lambda v, _: f'{v:.{y_digits}f}'))

  # Axis limits
  if 'x_min' in limits and 'x_max' in limits:
    ax.set_xlim(limits['x_min'], limits['x_max'])
  if 'y_min' in limits and 'y_max' in limits:
    ax.set_ylim(limits['y_min'], limits['y_max'])

  # Tick intervals
  if 'x_tick_interval' in ticks:
    ax.xaxis.set_major_locator(MultipleLocator(ticks['x_tick_interval']))
  if 'y_tick_interval' in ticks:
    ax.yaxis.set_major_locator(MultipleLocator(ticks['y_tick_interval']))

  # Plot data
  line, = ax.plot(x_data, y_data, label=label,
                  color=color, linewidth=linewidth, linestyle=linestyle)

  # Optional labels and title
  if xlabel is not None:
    ax.set_xlabel(xlabel, fontsize=x_label_fontsize, labelpad=x_label_pad)
  if ylabel is not None:
    ax.set_ylabel(ylabel, fontsize=y_label_fontsize, labelpad=y_label_pad)
  if title is not None:
    ax.set_title(title, fontsize=tick_fontsize, pad=title_pad)

  ax.grid(linestyle='--')
  ax.tick_params(axis='both', direction='in', length=4, width=1,
                 colors='black', top=True, right=True, labelsize=12)

  # Optional legend
  if label:
      legend_size = legend_cfg.get('fontsize', 12)
      legend_ncol = legend_cfg.get('ncol', 1)        # Novo parâmetro
      legend_loc = legend_cfg.get('loc', 'best')     # Novo parâmetro

      ax.legend(frameon=True,
                loc=legend_loc,
                ncol=legend_ncol,
                framealpha=1,
                prop={'size': legend_size})
  return line


def stem(ax: Axes,
         x_data,
         y_data,
         xlabel: Optional[str] = None,
         ylabel: Optional[str] = None,
         title: Optional[str] = None,
         label: str = '',
         stem_cfg: Dict[str, Any] = {}) -> Tuple[Any, Any, Any]:
  """
  Plots a stem graph on the given Axes object using an external structured configuration dictionary.

  Parameters
  ----------
  ax : matplotlib.axes.Axes
      The Matplotlib Axes object to draw the plot on.
  x_data : array_like
      X-axis data.
  y_data : array_like
      Y-axis data.
  xlabel : str, optional
      Label for the x-axis.
  ylabel : str, optional
      Label for the y-axis.
  title : str, optional
      Title of the plot.
  label : str, optional
      Legend label for the stem plot.
  stem_cfg : dict, optional
      Dictionary with structured configuration options. Expected keys:

      - 'style':
          - 'color' (str)
          - 'linewidth' (float)
          - 'marker_size' (float)

      - 'axis':
          - 'x_digits' (int)
          - 'y_digits' (int)
          - 'x_label_pad' (int)
          - 'y_label_pad' (int)
          - 'title_pad' (int)

      - 'limits':
          - 'x_min', 'x_max' (float)
          - 'y_min', 'y_max' (float)

      - 'legend':
          - 'fontsize' (int)

  Returns
  -------
  markerline, stemlines, baseline : tuple
      Elements of the stem plot returned by `ax.stem`.
  """
  style = stem_cfg.get('style', {})
  axis = stem_cfg.get('axis', {})
  limits = stem_cfg.get('limits', {})
  legend_cfg = stem_cfg.get('legend', {})

  # Extract style parameters
  color = style.get('color', '#120a8f')
  stem_width = style.get('linewidth', 1.67)
  marker_size = style.get('marker_size', 4)

  # Axis formatting
  x_digits = axis.get('x_digits', 3)
  y_digits = axis.get('y_digits', 2)
  x_label_pad = axis.get('x_label_pad', 8)
  y_label_pad = axis.get('y_label_pad', 8)
  title_pad = axis.get('title_pad', 8)
  x_label_fontsize = axis.get('x_label_fontsize', 16)
  y_label_fontsize = axis.get('y_label_fontsize', 16)
  tick_fontsize = axis.get('tick_fontsize', 16)

  # Formatter with spacing
  total_width = x_digits + y_digits + 1
  ax.xaxis.set_major_formatter(FuncFormatter(
      lambda v, _: f'{v:.{x_digits}f}'.rjust(total_width)))
  ax.yaxis.set_major_formatter(FuncFormatter(
      lambda v, _: f'{v:.{y_digits}f}'.rjust(total_width)))

  # Create stem plot
  markerline, stemlines, baseline = ax.stem(
      x_data, y_data,
      linefmt=color,
      markerfmt='o',
      basefmt=' ',
      bottom=0,
      label=label
  )
  plt.setp(stemlines, 'linewidth', stem_width)
  plt.setp(markerline, 'markersize', marker_size)

  # Optional labels and title
  if xlabel is not None:
    ax.set_xlabel(xlabel, fontsize=x_label_fontsize, labelpad=x_label_pad)
  if ylabel is not None:
    ax.set_ylabel(ylabel, fontsize=y_label_fontsize, labelpad=y_label_pad)
  if title is not None:
    ax.set_title(title, fontsize=tick_fontsize, pad=title_pad)

  ax.grid(linestyle='--')

  # Axis limits
  if 'x_min' in limits and 'x_max' in limits:
    ax.set_xlim(limits['x_min'], limits['x_max'])
  if 'y_min' in limits and 'y_max' in limits:
    ax.set_ylim(limits['y_min'], limits['y_max'])

  ax.tick_params(axis='both', direction='in', length=4, width=1,
                 colors='black', top=True, right=True, labelsize=12)

  # Legend
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
