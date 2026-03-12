import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from ui.main_window import MainWindow


def resource_path(relative_path: str) -> str:
  """
  Resolves the absolute path to a resource.
  """
  try:
    base_path = sys._MEIPASS
  except AttributeError:
    base_path = os.path.abspath(".")
  return os.path.join(base_path, relative_path)


def main() -> None:
  app = QApplication(sys.argv)

  fonts_dir = Path(resource_path("fonts"))
  if fonts_dir.is_dir():
    for f in list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf")):
      QFontDatabase.addApplicationFont(str(f))

  default_font = QFont("Poppins", 13) if "Poppins" in QFontDatabase.families(
  ) else QFont("Segoe UI", 13)
  app.setFont(default_font)

  window = MainWindow()
  window.show()
  sys.exit(app.exec())


if __name__ == "__main__":
  main()
