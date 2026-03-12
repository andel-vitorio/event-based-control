from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit,
    QPushButton, QFileDialog, QGraphicsDropShadowEffect,
    QComboBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase
from core.data_loader import scan_traffic_directory


class FileChipWidget(QWidget):
  """
  Custom item widget designed to display file metadata as categorized chips.
  Includes explicit layout constraints and scaled fonts for presentation visibility.
  """
  stateChanged = Signal(str, bool)

  def __init__(self, file_data: dict):
    super().__init__()
    self.file_path = file_data['path']
    self._setup_ui(file_data)

  def _setup_ui(self, file_data: dict) -> None:
    layout = QHBoxLayout(self)
    layout.setContentsMargins(4, 6, 4, 6)
    layout.setSpacing(8)

    self.checkbox = QCheckBox()
    self.checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
    self.checkbox.toggled.connect(self._on_toggled)
    layout.addWidget(self.checkbox)

    subdir_chip = self._create_chip(
        file_data['subdirectory'], "52, 152, 219")
    layout.addWidget(subdir_chip)

    if file_data['kmax'] is not None:
      kmax_chip = self._create_chip(
          f"K-Max: {file_data['kmax']}", "46, 204, 113")
      layout.addWidget(kmax_chip)

    design_text = "Emulation" if file_data['is_emulation'] else "Co-design"
    design_color = "155, 89, 182" if file_data['is_emulation'] else "230, 126, 34"
    design_chip = self._create_chip(design_text, design_color)
    layout.addWidget(design_chip)

    layout.addStretch()

  def _create_chip(self, text: str, rgb_color: str) -> QLabel:
    """
    Generates a styled label chip.
    Padding and font-size are scaled up for 1080p presentation displays.
    """
    label = QLabel(text)
    label.setStyleSheet(f"""
            background-color: rgba({rgb_color}, 0.15);
            color: rgb({rgb_color});
            border: 1px solid rgba({rgb_color}, 0.5);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 14px;
            font-family: "Poppins";
            font-weight: bold;
        """)
    return label

  def _on_toggled(self, checked: bool) -> None:
    self.stateChanged.emit(self.file_path, checked)

  def is_checked(self) -> bool:
    return self.checkbox.isChecked()


class SidebarWidget(QWidget):
  """
  Sidebar configuration panel for parameter inputs, directory selection, 
  and file filtering/selection.
  """

  selectionChanged = Signal(list)
  directoryChanged = Signal(str)

  def __init__(self):
    super().__init__()
    self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    self.current_scanned_files = []
    self._setup_ui()
    self._connect_signals()

  def _create_field_title(self, text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldTitle")
    font = QFont()
    if "Poppins" in QFontDatabase.families():
      font.setFamily("Poppins Bold")
    else:
      font.setWeight(QFont.Weight.Bold)
    label.setFont(font)
    return label

  def _setup_ui(self) -> None:
    self.setObjectName("sidebar")

    shadow = QGraphicsDropShadowEffect()
    shadow.setOffset(2, 2)
    shadow.setBlurRadius(15)
    shadow.setColor(QColor(0, 0, 0, 80))
    self.setGraphicsEffect(shadow)

    layout = QVBoxLayout(self)
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(10)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    title_font = QFont()
    title_font.setPointSize(22)
    title_font.setBold(True)

    self.sidebar_title_label = QLabel("Controls")
    self.sidebar_title_label.setFont(title_font)
    self.sidebar_title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(self.sidebar_title_label)

    layout.addWidget(self._create_field_title("Target Directory:"))
    dir_layout = QHBoxLayout()
    dir_layout.setContentsMargins(0, 0, 0, 0)

    self.dir_input = QLineEdit("data")
    self.dir_picker_btn = QPushButton("...")
    self.dir_picker_btn.setFixedWidth(30)

    dir_layout.addWidget(self.dir_input)
    dir_layout.addWidget(self.dir_picker_btn)
    layout.addLayout(dir_layout)

    self.refresh_btn = QPushButton("Refresh Files Now")
    layout.addWidget(self.refresh_btn)

    layout.addWidget(self._create_field_title("Filter by Subdirectory:"))
    self.sub_dir_combo = QComboBox()
    self.sub_dir_combo.addItem("All")
    layout.addWidget(self.sub_dir_combo)

    layout.addWidget(self._create_field_title("Design Approach:"))
    self.design_combo = QComboBox()
    self.design_combo.addItems(["All", "Co-design Only", "Emulation Only"])
    layout.addWidget(self.design_combo)

    layout.addWidget(self._create_field_title("Filter by K-Max:"))
    self.kmax_combo = QComboBox()
    self.kmax_combo.addItem("All")
    layout.addWidget(self.kmax_combo)

    layout.addWidget(self._create_field_title("Select Files to Compare:"))
    self.file_list_widget = QListWidget()
    self.file_list_widget.setSelectionMode(
        QAbstractItemView.SelectionMode.NoSelection)
    layout.addWidget(self.file_list_widget)

  def _connect_signals(self) -> None:
    self.dir_picker_btn.clicked.connect(self._select_directory)
    self.refresh_btn.clicked.connect(self._trigger_scan_and_sync)
    self.sub_dir_combo.currentIndexChanged.connect(self._apply_filters)
    self.design_combo.currentIndexChanged.connect(self._apply_filters)
    self.kmax_combo.currentIndexChanged.connect(self._apply_filters)

  def _select_directory(self) -> None:
    directory = QFileDialog.getExistingDirectory(
        self, "Select Directory", self.dir_input.text())
    if directory:
      self.dir_input.setText(directory)
      self._trigger_scan_and_sync()

  def _trigger_scan_and_sync(self) -> None:
    """Invoked by user actions to explicitly broadcast the directory change."""
    self.perform_scan(emit_sync=True)

  def perform_scan(self, emit_sync: bool = False) -> None:
    """
    Scans the selected directory and populates the filter dropdowns.
    Args:
        emit_sync (bool): If True, broadcasts the directoryChanged signal to the MainWindow.
    """
    directory = self.dir_input.text()
    self.current_scanned_files = scan_traffic_directory(directory)

    subdirs = sorted(list(set(f['subdirectory']
                     for f in self.current_scanned_files)))
    kmax_vals = sorted(list(
        set(f['kmax'] for f in self.current_scanned_files if f['kmax'] is not None)))

    self.sub_dir_combo.blockSignals(True)
    self.sub_dir_combo.clear()
    self.sub_dir_combo.addItem("All")
    self.sub_dir_combo.addItems(subdirs)
    self.sub_dir_combo.blockSignals(False)

    self.kmax_combo.blockSignals(True)
    self.kmax_combo.clear()
    self.kmax_combo.addItem("All")
    self.kmax_combo.addItems([str(k) for k in kmax_vals])
    self.kmax_combo.blockSignals(False)

    self._apply_filters()

    if emit_sync:
      self.directoryChanged.emit(directory)

  def _apply_filters(self) -> None:
    self.file_list_widget.clear()

    sel_subdir = self.sub_dir_combo.currentText()
    sel_design = self.design_combo.currentText()
    sel_kmax = self.kmax_combo.currentText()

    for f in self.current_scanned_files:
      if sel_subdir != "All" and f['subdirectory'] != sel_subdir:
        continue

      if sel_design == "Co-design Only" and f['is_emulation']:
        continue
      if sel_design == "Emulation Only" and not f['is_emulation']:
        continue

      if sel_kmax != "All":
        if f['kmax'] is None or str(f['kmax']) != sel_kmax:
          continue

      item = QListWidgetItem(self.file_list_widget)
      chip_widget = FileChipWidget(f)

      item.setSizeHint(chip_widget.sizeHint())
      self.file_list_widget.setItemWidget(item, chip_widget)

      chip_widget.stateChanged.connect(self._handle_chip_state_changed)

  def _handle_chip_state_changed(self, file_path: str, is_checked: bool) -> None:
    selected_paths = []
    for index in range(self.file_list_widget.count()):
      item = self.file_list_widget.item(index)
      widget = self.file_list_widget.itemWidget(item)
      if isinstance(widget, FileChipWidget) and widget.is_checked():
        selected_paths.append(widget.file_path)

    self.selectionChanged.emit(selected_paths)
