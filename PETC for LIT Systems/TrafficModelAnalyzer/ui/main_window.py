from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QApplication, QGraphicsDropShadowEffect
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from ui.tabs.cycle_analysis_tab import CycleAnalysisTab
from ui.tabs.savings_analysis_tab import SavingsAnalysisTab
from ui.tabs.bounds_analysis_tab import BoundsAnalysisTab
from ui.tabs.entropy_analysis_tab import EntropyAnalysisTab
from ui.styles import DARK_STYLESHEET, LIGHT_STYLESHEET


class MainWindow(QMainWindow):
  """
  Primary application window managing top-level navigation, global theme states, 
  and cross-tab state synchronization.
  """

  def __init__(self):
    super().__init__()
    self.setWindowTitle("Event-Based Traffic Model Analysis")
    self.setGeometry(100, 100, 1600, 900)
    self._setup_ui()
    self.apply_theme("Dark")

  def _setup_ui(self) -> None:
    self.central_widget = QWidget()
    self.setCentralWidget(self.central_widget)
    self.main_layout = QVBoxLayout(self.central_widget)
    self.main_layout.setContentsMargins(0, 0, 0, 0)
    self.main_layout.setSpacing(0)

    self.tabs = QTabWidget()
    shadow = QGraphicsDropShadowEffect()
    shadow.setOffset(0, 2)
    shadow.setBlurRadius(15)
    shadow.setColor(QColor(0, 0, 0, 100))
    self.tabs.setGraphicsEffect(shadow)

    self._setup_corner_widgets()

    # Instanciação das Abas
    self.cycle_tab = CycleAnalysisTab()
    self.tabs.addTab(self.cycle_tab, "Cycle Analysis")

    self.savings_tab = SavingsAnalysisTab()
    self.tabs.addTab(self.savings_tab, "Savings Analysis")

    self.bounds_tab = BoundsAnalysisTab()
    self.tabs.addTab(self.bounds_tab, "Bounds Analysis")

    self.entropy_tab = EntropyAnalysisTab()
    self.tabs.addTab(self.entropy_tab, "Entropy Analysis")

    # Conexão dos mediadores de sincronização de diretório
    self.cycle_tab.sidebar.directoryChanged.connect(
        lambda directory: self._sync_directories(
            directory, source_tab=self.cycle_tab)
    )
    self.savings_tab.sidebar.directoryChanged.connect(
        lambda directory: self._sync_directories(
            directory, source_tab=self.savings_tab)
    )
    self.bounds_tab.sidebar.directoryChanged.connect(
        lambda directory: self._sync_directories(
            directory, source_tab=self.bounds_tab)
    )
    self.entropy_tab.sidebar.directoryChanged.connect(
        lambda directory: self._sync_directories(
            directory, source_tab=self.entropy_tab)
    )

    self.main_layout.addWidget(self.tabs)

  def _sync_directories(self, new_directory: str, source_tab: QWidget) -> None:
    """
    Propagates directory path updates to all sidebars except the one that triggered the change.
    """
    all_tabs = [self.cycle_tab, self.savings_tab,
                self.bounds_tab, self.entropy_tab]

    for tab in all_tabs:
      if tab != source_tab:
        if tab.sidebar.dir_input.text() != new_directory:
          tab.sidebar.dir_input.setText(new_directory)
          tab.sidebar.perform_scan(emit_sync=False)

  def _setup_corner_widgets(self) -> None:
    self.theme_button = QPushButton("🌙")
    self.theme_button.setCheckable(True)
    self.theme_button.setToolTip("Toggle Light/Dark Theme")
    self.theme_button.setFixedWidth(40)
    self.theme_button.setStyleSheet(
        "QPushButton { border: none; font-size: 18px; } QPushButton:checked { background-color: #444; }"
    )
    self.theme_button.clicked.connect(self._toggle_theme)

    corner_widget = QWidget()
    corner_layout = QHBoxLayout(corner_widget)
    corner_layout.setContentsMargins(0, 5, 10, 0)
    corner_layout.addStretch()
    corner_layout.addWidget(QLabel("Theme:"))
    corner_layout.addWidget(self.theme_button)
    self.tabs.setCornerWidget(corner_widget, Qt.Corner.TopRightCorner)

  def _toggle_theme(self) -> None:
    if self.theme_button.isChecked():
      self.apply_theme("Dark")
    else:
      self.apply_theme("Light")

  def apply_theme(self, theme_name: str) -> None:
    app = QApplication.instance()
    self.theme_button.blockSignals(True)

    if theme_name == "Dark":
      app.setStyleSheet(DARK_STYLESHEET)
      self.cycle_tab.update_theme("dark", "#DDFFFFFF")
      self.savings_tab.update_theme("dark", "#DDFFFFFF")
      self.bounds_tab.update_theme("dark", "#DDFFFFFF")
      self.entropy_tab.update_theme("dark", "#DDFFFFFF")
      self.theme_button.setText("☀️")
      self.theme_button.setChecked(True)
    elif theme_name == "Light":
      app.setStyleSheet(LIGHT_STYLESHEET)
      self.cycle_tab.update_theme("light", "#111111")
      self.savings_tab.update_theme("light", "#111111")
      self.bounds_tab.update_theme("light", "#111111")
      self.entropy_tab.update_theme("light", "#111111")
      self.theme_button.setText("🌙")
      self.theme_button.setChecked(False)

    self.theme_button.blockSignals(False)
