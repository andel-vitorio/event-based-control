import re
import pandas as pd
import numpy as np
from pathlib import Path
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGraphicsDropShadowEffect, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush

from ui.widgets.sidebar import SidebarWidget


class SavingsAnalysisTab(QWidget):
  """
  Analyzes and visualizes traffic savings and robustness metrics from event-based control models,
  segregated by subsequence lengths (l_word).
  """

  def __init__(self):
    super().__init__()
    self.selected_files: list[str] = []
    self.current_df: pd.DataFrame | None = None
    self.raw_table_data: list[dict] = []
    self._setup_ui()

  def _setup_ui(self) -> None:
    """
    Constructs the internal layout containing the sidebar, chart selectors, 
    line graph, independent table filters, and comparative metrics table.
    """
    self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    self.main_layout = QHBoxLayout(self)
    self.main_layout.setContentsMargins(15, 15, 15, 15)
    self.main_layout.setSpacing(15)

    self.sidebar = SidebarWidget()
    self.sidebar.selectionChanged.connect(self.update_selected_files)
    self.main_layout.addWidget(self.sidebar, 1)

    self.content_widget = QWidget()
    self.content_layout = QVBoxLayout(self.content_widget)
    self.content_layout.setContentsMargins(0, 0, 0, 0)
    self.content_layout.setSpacing(15)

    control_layout = QHBoxLayout()
    self.graph_file_selector = QComboBox()
    self.graph_file_selector.currentIndexChanged.connect(
        self._on_file_selected)

    self.l_word_selector = QComboBox()
    self.l_word_selector.currentIndexChanged.connect(self._render_graph)

    control_label = QLabel("Display Chart For:")
    control_label.setObjectName("fieldTitle")

    l_label = QLabel("Length (L):")
    l_label.setObjectName("fieldTitle")

    control_layout.addWidget(control_label)
    control_layout.addWidget(self.graph_file_selector, 3)
    control_layout.addSpacing(20)
    control_layout.addWidget(l_label)
    control_layout.addWidget(self.l_word_selector, 1)

    self.content_layout.addLayout(control_layout)

    self.graph_card = QWidget()
    self.graph_card.setObjectName("graphCard")
    self.graph_card.setAttribute(
        Qt.WidgetAttribute.WA_StyledBackground, True)

    shadow = QGraphicsDropShadowEffect()
    shadow.setOffset(2, 2)
    shadow.setBlurRadius(15)
    shadow.setColor(QColor(0, 0, 0, 80))
    self.graph_card.setGraphicsEffect(shadow)

    graph_layout = QVBoxLayout(self.graph_card)

    pg.setConfigOption('antialias', True)
    self.plot_widget = pg.PlotWidget(title="Traffic Savings vs. Parameter")
    self.plot_widget.setLabel('left', 'Min Savings (%)')
    self.plot_widget.showGrid(x=False, y=True, alpha=0.3)
    self.plot_widget.setBackground('#171717')
    self.plot_widget.setYRange(0, 105, padding=0)

    graph_layout.addWidget(self.plot_widget)
    self.content_layout.addWidget(self.graph_card, 3)

    table_header_layout = QHBoxLayout()
    table_label = QLabel("Comparative Savings & Robustness Metrics")
    table_label.setObjectName("fieldTitle")
    table_header_layout.addWidget(table_label)

    table_header_layout.addStretch()

    table_header_layout.addWidget(QLabel("K-Max:"))
    self.table_filter_kmax = QComboBox()
    self.table_filter_kmax.currentIndexChanged.connect(
        self._apply_table_filters)
    table_header_layout.addWidget(self.table_filter_kmax)

    table_header_layout.addWidget(QLabel("Design:"))
    self.table_filter_design = QComboBox()
    self.table_filter_design.currentIndexChanged.connect(
        self._apply_table_filters)
    table_header_layout.addWidget(self.table_filter_design)

    table_header_layout.addWidget(QLabel("L:"))
    self.table_filter_l = QComboBox()
    self.table_filter_l.currentIndexChanged.connect(
        self._apply_table_filters)
    table_header_layout.addWidget(self.table_filter_l)

    self.content_layout.addLayout(table_header_layout)

    self.metrics_table = QTableWidget()
    self.metrics_table.setColumnCount(8)
    self.metrics_table.setHorizontalHeaderLabels([
        "K-Max", "Design Approach", "L", "Lowest Min Savings (%)",
        "Highest Min Savings (%)", "Avg Min Savings (%)", "Avg Rob (Inf)", "Avg Rob (Sup)"
    ])

    header = self.metrics_table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    for i in range(3):
      header.setSectionResizeMode(
          i, QHeaderView.ResizeMode.ResizeToContents)

    self.metrics_table.setEditTriggers(
        QAbstractItemView.EditTrigger.NoEditTriggers)
    self.metrics_table.setSelectionBehavior(
        QAbstractItemView.SelectionBehavior.SelectRows)

    self.content_layout.addWidget(self.metrics_table, 2)
    self.main_layout.addWidget(self.content_widget, 4)

  def update_selected_files(self, file_paths: list[str]) -> None:
    """
    Updates the internal state and dropdowns based on the selected file paths,
    triggering data extraction for the comparative table.
    """
    self.selected_files = file_paths

    self.graph_file_selector.blockSignals(True)
    self.graph_file_selector.clear()

    for path in file_paths:
      filename = Path(path).name
      self.graph_file_selector.addItem(filename, userData=path)

    self.graph_file_selector.blockSignals(False)

    if file_paths:
      self.graph_file_selector.setCurrentIndex(0)
      self._on_file_selected()
    else:
      self.current_df = None
      self.plot_widget.clear()

    self._populate_comparative_table()

  def _on_file_selected(self) -> None:
    """
    Loads the dataframe into memory and extracts discrete l_word values.
    """
    file_path = self.graph_file_selector.currentData()
    if not file_path:
      self.l_word_selector.clear()
      self.current_df = None
      return

    try:
      self.current_df = pd.read_csv(file_path)
    except Exception:
      self.current_df = None
      return

    self.l_word_selector.blockSignals(True)
    self.l_word_selector.clear()

    if 'l_word' in self.current_df.columns:
      l_vals = sorted(self.current_df['l_word'].dropna().unique())
      self.l_word_selector.addItems([str(int(val)) for val in l_vals])
    else:
      self.l_word_selector.addItem("N/A")

    self.l_word_selector.blockSignals(False)
    self._render_graph()

  def _render_graph(self) -> None:
    """
    Constructs a line chart with markers evaluating traffic savings.
    """
    self.plot_widget.clear()

    if self.current_df is None or self.current_df.empty:
      return

    df = self.current_df.copy()

    if 'l_word' in df.columns:
      selected_l = self.l_word_selector.currentText()
      if selected_l and selected_l != "N/A":
        df = df[df['l_word'] == int(selected_l)]

    is_log_scale = False
    if 'eps_tr' in df.columns:
      x_col = 'eps_tr'
      x_label = 'log10(eps_tr) (Parameter)'
      is_log_scale = True
    elif 'sigma' in df.columns:
      x_col = 'sigma'
      x_label = 'σ (Parameter)'
    else:
      return

    if 'SavingsPct' not in df.columns:
      return

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df = df.dropna(subset=[x_col, 'SavingsPct'])
    df = df[df[x_col] > 0.0]

    df_agg = df.groupby(x_col, as_index=False).agg({
        'SavingsPct': 'min'
    }).sort_values(by=x_col)

    x_vals = df_agg[x_col].values
    savings_vals = df_agg['SavingsPct'].values

    if len(x_vals) == 0:
      return

    if is_log_scale:
      x_vals = np.log10(x_vals)

    self.plot_widget.setLogMode(x=False, y=False)
    self.plot_widget.setLabel('bottom', x_label)

    # Line plot with diamond markers mimicking the reference style
    plot_item = pg.PlotDataItem(
        x_vals, savings_vals,
        pen=pg.mkPen('#F39C12', width=2.5),
        symbol='d', symbolSize=10,
        symbolBrush='#F39C12', symbolPen=None
    )
    self.plot_widget.addItem(plot_item)

    current_file = self.graph_file_selector.currentText()
    current_l = self.l_word_selector.currentText()
    title_suffix = f" (for l={current_l})" if current_l != "N/A" else ""
    self.plot_widget.plotItem.setTitle(
        f"Traffic Savings vs. Parameter - {current_file}{title_suffix}")

  def _populate_comparative_table(self) -> None:
    """
    Caches aggregated savings and robustness KPIs to power the comparative table.
    """
    self.raw_table_data = []

    for path in self.selected_files:
      try:
        df = pd.read_csv(path)
        filename = Path(path).name

        # Aplica o mesmo filtro dos gráficos: remove "valores lixo" (sigma/eps_tr <= 0)
        if 'eps_tr' in df.columns:
          df['eps_tr'] = pd.to_numeric(df['eps_tr'], errors='coerce')
          df = df[df['eps_tr'] > 1e-6]
        elif 'sigma' in df.columns:
          df['sigma'] = pd.to_numeric(df['sigma'], errors='coerce')
          df = df[df['sigma'] > 1e-6]

        # Se o arquivo ficar vazio após o filtro, ignora
        if df.empty:
          continue

        kmax_match = re.search(r'kmax(\d+)', filename, re.IGNORECASE)
        kmax_val = kmax_match.group(1) if kmax_match else "N/A"
        design_val = "Emulation" if "emulation" in filename.lower() else "Co-design"

        if 'l_word' in df.columns:
          l_groups = df.groupby('l_word')
        else:
          l_groups = [(None, df)]

        for l_val, group in l_groups:
          min_sav = group['SavingsPct'].min(
          ) if 'SavingsPct' in group.columns else 0.0
          max_sav = group['SavingsPct'].max(
          ) if 'SavingsPct' in group.columns else 0.0
          avg_sav = group['SavingsPct'].mean(
          ) if 'SavingsPct' in group.columns else 0.0

          rob_inf = group['RobInfLimAvg'].mean(
          ) if 'RobInfLimAvg' in group.columns else 0.0
          rob_sup = group['RobSupLimAvg'].mean(
          ) if 'RobSupLimAvg' in group.columns else 0.0

          l_display = str(int(l_val)) if l_val is not None else "N/A"

          self.raw_table_data.append({
              'kmax': kmax_val,
              'design': design_val,
              'l': l_display,
              'min_sav': min_sav,
              'max_sav': max_sav,
              'avg_sav': avg_sav,
              'rob_inf': rob_inf,
              'rob_sup': rob_sup
          })

      except Exception:
        continue

    self._update_table_filter_options()
    self._apply_table_filters()

  def _update_table_filter_options(self) -> None:
    """
    Updates filter comboboxes sequentially.
    """
    self.table_filter_kmax.blockSignals(True)
    self.table_filter_design.blockSignals(True)
    self.table_filter_l.blockSignals(True)

    self.table_filter_kmax.clear()
    self.table_filter_design.clear()
    self.table_filter_l.clear()

    self.table_filter_kmax.addItem("All")
    self.table_filter_design.addItem("All")
    self.table_filter_l.addItem("All")

    if not self.raw_table_data:
      self.table_filter_kmax.blockSignals(False)
      self.table_filter_design.blockSignals(False)
      self.table_filter_l.blockSignals(False)
      return

    kmax_set = sorted(list(set(d['kmax'] for d in self.raw_table_data)))
    design_set = sorted(list(set(d['design']
                        for d in self.raw_table_data)))
    l_set = sorted(list(set(d['l'] for d in self.raw_table_data)))

    self.table_filter_kmax.addItems(kmax_set)
    self.table_filter_design.addItems(design_set)
    self.table_filter_l.addItems(l_set)

    self.table_filter_kmax.blockSignals(False)
    self.table_filter_design.blockSignals(False)
    self.table_filter_l.blockSignals(False)

  def _apply_table_filters(self) -> None:
    """
    Reconstructs table data dynamically honoring active UI filters.
    """
    self.metrics_table.setRowCount(0)

    f_kmax = self.table_filter_kmax.currentText()
    f_design = self.table_filter_design.currentText()
    f_l = self.table_filter_l.currentText()

    row_idx = 0
    for data in self.raw_table_data:
      if f_kmax != "All" and data['kmax'] != f_kmax:
        continue
      if f_design != "All" and data['design'] != f_design:
        continue
      if f_l != "All" and data['l'] != f_l:
        continue

      self.metrics_table.insertRow(row_idx)

      items = [
          QTableWidgetItem(data['kmax']),
          QTableWidgetItem(data['design']),
          QTableWidgetItem(data['l']),
          QTableWidgetItem(f"{data['min_sav']:.2f}%"),
          QTableWidgetItem(f"{data['max_sav']:.2f}%"),
          QTableWidgetItem(f"{data['avg_sav']:.2f}%"),
          QTableWidgetItem(f"{data['rob_inf']:.4f}"),
          QTableWidgetItem(f"{data['rob_sup']:.4f}")
      ]

      for col_idx, item in enumerate(items):
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metrics_table.setItem(row_idx, col_idx, item)

      row_idx += 1

  def update_theme(self, theme_name: str, fg_color: str) -> None:
    """
    Modifies chart component styles in reaction to theme swaps.
    """
    if hasattr(self, 'sidebar'):
      self.sidebar.sidebar_title_label.setStyleSheet(
          f"color: {fg_color};")

    bg_color = '#F8F9FA' if theme_name == 'light' else '#171717'
    self.plot_widget.setBackground(bg_color)

    axis_pen = pg.mkPen(color=fg_color)
    for axis_name in ['left', 'bottom']:
      axis = self.plot_widget.getAxis(axis_name)
      axis.setPen(axis_pen)
      axis.setTextPen(axis_pen)

    if self.plot_widget.plotItem.titleLabel.text:
      self.plot_widget.plotItem.setTitle(
          self.plot_widget.plotItem.titleLabel.text, color=fg_color
      )
