import matplotlib as mpl
from IPython import get_ipython

try:
  from IPython.display import HTML, display
  from IPython.core.magic import register_cell_magic
  _IN_IPYTHON = True
except Exception:
  _IN_IPYTHON = False


def setup():
  if _IN_IPYTHON:
    display(HTML("""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap');

          body {
            font-family: 'JetBrains Mono', -apple-system, BlinkMacSystemFont,
                         'Segoe WPC', 'Segoe UI', system-ui, 'Ubuntu',
                         'Droid Sans', sans-serif;
            line-height: 1.67;
          }
        </style>
        """))

    @register_cell_magic
    def skip(line, cell):
      ip = get_ipython()
      user_ns = ip.user_ns

      if not line.strip():
        print("Cell skipped (unconditional).")
        return

      try:
        condition = eval(line, user_ns)
      except Exception as e:
        print(f"Error evaluating condition '{line}': {e}")
        return

      if condition:
        print(f"Cell skipped (condition: {line})")
        return

      exec(cell, user_ns)

  mpl.rcParams["figure.dpi"] = 100

  try:
    from . import Graphs as gph
    gph.use_latex()
  except Exception:
    pass
