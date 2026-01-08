from IPython.display import display, Math
import re
from IPython.display import display, HTML, Math, Latex
import numpy as np
import math


import re
import numpy as np
from typing import Union, List
from IPython.display import display, Math


def dec2tex(value: Union[str, float, int],
            format_type: str = "scientific",
            decimals: int = 2) -> str:
  """
  Converts a numeric or symbolic value into a LaTeX-compatible string.

  Parameters
  ----------
  value : str | float | int
      Value to be converted. Can contain symbolic terms like 'a1', 'k_2', or numeric values.
  format_type : str, optional
      Either "scientific" (default) for scientific notation or "decimal" for fixed-point.
  decimals : int, optional
      Number of decimal places to show for numeric values.

  Returns
  -------
  str
      LaTeX-formatted string representing the value.
  """
  if isinstance(value, str):
    expr = value.replace("*", "\\,")  # spacing for products
    # Convert variables like a1, k_2, theta3 → a_{1}, k_{2}, theta_{3}
    expr = re.sub(r"([a-zA-Zα-ωΑ-Ω]+)_?(\d+)", r"\1_{\2}", expr)
    return expr

  try:
    num = float(value)
  except (ValueError, TypeError):
    return str(value)

  if format_type == "scientific":
    formatted = f"{num:.{decimals}e}"
    base, exp = formatted.split("e")
    exp = int(exp)
    return f"{base}\\times10^{{{exp}}}"
  else:
    return f"{num:.{decimals}f}"


def mat2tex(matrix: Union[List[List[Union[str, float, int]]], np.ndarray],
            format_type: str = "scientific",
            decimals: int = 2) -> str:
  """
  Converts a numeric or symbolic matrix into a LaTeX-formatted string.

  Parameters
  ----------
  matrix : list[list[str | float | int]] | np.ndarray
      Matrix to be formatted. Elements may contain symbolic expressions.
  format_type : str, optional
      Either "scientific" or "decimal". Default is "scientific".
  decimals : int, optional
      Number of decimal places to show for numeric entries.

  Returns
  -------
  str
      LaTeX string representing the matrix (e.g., '\\begin{bmatrix} ... \\end{bmatrix}').
  """
  matrix = np.array(matrix, dtype=object)
  formatted_rows = []
  for row in matrix:
    formatted_row = " & ".join(
        dec2tex(val, format_type, decimals) for val in row)
    formatted_rows.append(formatted_row)
  formatted_matrix = " \\\\ ".join(formatted_rows)
  return f"\\begin{{bmatrix}} {formatted_matrix} \\end{{bmatrix}}"


def display_matrix(matrix: Union[List[List[Union[str, float, int]]], np.ndarray],
                   name: str = None,
                   format_type: str = "scientific",
                   decimals: int = 2) -> None:
  """
  Displays a numeric or symbolic matrix as a LaTeX equation inside a Jupyter Notebook.

  Parameters
  ----------
  matrix : list[list[str | float | int]] | np.ndarray
      Matrix to be displayed.
  name : str, optional
      Optional label to show before the matrix (e.g., "A =").
  format_type : str, optional
      Either "scientific" or "decimal". Default is "scientific".
  decimals : int, optional
      Number of decimal places for numeric values.

  Returns
  -------
  None
      The matrix is rendered in LaTeX within the notebook.
  """
  latex_matrix = mat2tex(matrix, format_type, decimals)
  if name:
    latex_matrix = f"{name} = " + latex_matrix
  display(Math(latex_matrix))
