from IPython.display import display, HTML, Math, Latex
import numpy as np
import math


def display_latex(latex_string: str, inline: bool = True) -> None:
  """
  Displays a LaTeX-formatted expression using IPython's Math display.

  Parameters
  ----------
  latex_string : str
      The LaTeX-formatted expression to display (e.g., '\\beta', '\\frac{1}{2}', '\\sqrt{2}', etc.).
  inline : bool, optional
      Whether to display the math inline. Default is True.
  """
  display(Math(latex_string if inline else f'\\[{latex_string}\\]'))


def dec2tex(value: float, format_type: str = "scientific", decimals: int = 2) -> str:
  """
  Formata um número decimal para o estilo LaTeX.

  Parâmetros:
  - value (float): O número a ser formatado.
  - format_type (str): "scientific" para notação científica ou "decimal" para truncado.
  - decimals (int): Número de casas decimais a serem exibidas.

  Retorna:
  - str: Representação formatada do número em LaTeX.
  """
  if format_type == "scientific":
    if value == 0:
      return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10 ** exponent)
    return (f'{mantissa:.{decimals}f} \\times 10^{{{exponent}}}')
  elif format_type == "decimal":
    factor = 10 ** decimals
    truncated_value = math.trunc(value * factor) / factor
    return Math(rf"{truncated_value:.{decimals}f})")
  else:
    raise ValueError("format_type deve ser 'scientific' ou 'decimal'")


def mat2tex(matrix, format_type: str = "scientific", decimals: int = 2) -> str:
  """
  Formata uma matriz para o estilo LaTeX.

  Parâmetros:
  - matrix (list ou np.ndarray): A matriz a ser formatada.
  - format_type (str): "scientific" para notação científica ou "decimal" para truncado.
  - decimals (int): Número de casas decimais a serem exibidas.

  Retorna:
  - str: Representação formatada da matriz em LaTeX.
  """
  matrix = np.array(matrix)
  formatted_rows = [" & ".join(dec2tex(
      num, format_type, decimals) for num in row) for row in matrix]
  formatted_matrix = " \\\\ ".join(formatted_rows)
  return f"\\begin{{bmatrix}} {formatted_matrix} \\end{{bmatrix}}"
