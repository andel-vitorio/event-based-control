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
from PySide6.QtGui import QColor, QBrush, QFont

from ui.widgets.sidebar import SidebarWidget


class CycleAnalysisTab(QWidget):
  """
  Analyzes and visualizes stable and unstable limit cycles from event-based traffic models,
  with built-in separation for subsequence lengths (l_word) and isolated table filtering.
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
    stacked bar graph, independent table filters, and comparative metrics table.
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
    self.plot_widget = pg.PlotWidget(
        title="Stability Transition: Stable vs. Unstable Cycles")
    self.plot_widget.setLabel('left', 'Number of Cycles')
    self.plot_widget.showGrid(x=False, y=True, alpha=0.3)
    self.plot_widget.setBackground('#171717')

    self.legend = self.plot_widget.addLegend(offset=(10, 10))
    graph_layout.addWidget(self.plot_widget)

    self.content_layout.addWidget(self.graph_card, 3)

    table_header_layout = QHBoxLayout()
    table_label = QLabel("Comparative Cycle Metrics")
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
        "K-Max", "Design Approach", "L", "Max Total",
        "Max Stable", "Max Unstable", "Avg Stable", "Avg Unstable"
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
    Updates the internal state and the dropdown based on the provided list of file paths.
    Triggers data extraction for the comparative table.
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
      self.legend.clear()

    self._populate_comparative_table()

  def _on_file_selected(self) -> None:
    """
    Loads the dataframe into memory and extracts unique l_word values 
    to populate the length selector dropdown.
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
    Constructs a stacked bar chart differentiating stable and unstable limit cycles.
    Applies an ultra-thin width rendering technique to prevent dense data from visually 
    collapsing into continuous blocks.
    """
    self.plot_widget.clear()
    self.legend.clear()

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

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df = df.dropna(subset=[x_col])
    df = df[df[x_col] > 1e-6]

    df_agg = df.groupby(x_col, as_index=False).agg({
        'StableCycles': 'mean',
        'UnstableCycles': 'mean'
    }).sort_values(by=x_col)

    x_vals = df_agg[x_col].values
    stable_vals = df_agg['StableCycles'].values
    unstable_vals = df_agg['UnstableCycles'].values

    if len(x_vals) == 0:
      return

    if is_log_scale:
      x_vals = np.log10(x_vals)

    self.plot_widget.setLogMode(x=False, y=False)
    self.plot_widget.setLabel('bottom', x_label)

    diffs = np.diff(x_vals)
    valid_diffs = diffs[diffs > 0]
    min_gap = np.min(valid_diffs) if valid_diffs.size > 0 else 0.1

    width = min_gap * 0.25
    if width <= 0:
      width = 1e-6

    stable_mask = stable_vals > 0
    if np.any(stable_mask):
      bar_stable = pg.BarGraphItem(
          x=x_vals[stable_mask],
          height=stable_vals[stable_mask],
          width=width,
          brush=pg.mkBrush(46, 204, 113, 200),
          pen=pg.mkPen(46, 204, 113)
      )
      self.plot_widget.addItem(bar_stable)
      self.legend.addItem(bar_stable, "Stable Cycles")

    unstable_mask = unstable_vals > 0
    if np.any(unstable_mask):
      bar_unstable = pg.BarGraphItem(
          x=x_vals[unstable_mask],
          y0=stable_vals[unstable_mask],
          height=unstable_vals[unstable_mask],
          width=width,
          brush=pg.mkBrush(231, 76, 60, 200),
          pen=pg.mkPen(231, 76, 60)
      )
      self.plot_widget.addItem(bar_unstable)
      self.legend.addItem(bar_unstable, "Unstable Cycles")

    current_file = self.graph_file_selector.currentText()
    current_l = self.l_word_selector.currentText()
    title_suffix = f" (L = {current_l})" if current_l != "N/A" else ""
    self.plot_widget.plotItem.setTitle(
        f"Stability Transition - {current_file}{title_suffix}")

  def _populate_comparative_table(self) -> None:
    """
    Parses all selected files, extracts metadata, computes group KPIs, 
    and caches the results for independent table filtering.
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
          max_total = group['TotalCycles'].max(
          ) if 'TotalCycles' in group.columns else 0
          max_stable = group['StableCycles'].max(
          ) if 'StableCycles' in group.columns else 0
          max_unstable = group['UnstableCycles'].max(
          ) if 'UnstableCycles' in group.columns else 0
          avg_stable = group['StableCycles'].mean(
          ) if 'StableCycles' in group.columns else 0.0
          avg_unstable = group['UnstableCycles'].mean(
          ) if 'UnstableCycles' in group.columns else 0.0

          l_display = str(int(l_val)) if l_val is not None else "N/A"

          self.raw_table_data.append({
              'kmax': kmax_val,
              'design': design_val,
              'l': l_display,
              'max_total': max_total,
              'max_stable': max_stable,
              'max_unstable': max_unstable,
              'avg_stable': avg_stable,
              'avg_unstable': avg_unstable
          })

      except Exception:
        continue

    self._update_table_filter_options()
    self._apply_table_filters()

  def _update_table_filter_options(self) -> None:
    """
    Evaluates the cached table data to dynamically populate the filter comboboxes.
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
    Reconstructs the comparative table view from the data cache, 
    respecting the current state of the table-specific filter widgets.
    Applies Qt.AlignmentFlag.AlignCenter to ensure centralized text rendering.
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
          QTableWidgetItem(f"{int(data['max_total'])}"),
          QTableWidgetItem(f"{int(data['max_stable'])}"),
          QTableWidgetItem(f"{int(data['max_unstable'])}"),
          QTableWidgetItem(f"{data['avg_stable']:.2f}"),
          QTableWidgetItem(f"{data['avg_unstable']:.2f}")
      ]

      for col_idx, item in enumerate(items):
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metrics_table.setItem(row_idx, col_idx, item)

      row_idx += 1

  def update_theme(self, theme_name: str, fg_color: str) -> None:
    """
    Propagates UI theme state changes to internal components.
    """
    if hasattr(self, 'sidebar'):
      self.sidebar.sidebar_title_label.setStyleSheet(
          f"color: {fg_color};")

    bg_color = '#F8F9FA' if theme_name == 'light' else '#171717'
    legend_bg = '#FFFFFF' if theme_name == 'light' else '#242424'

    label_style = {'color': fg_color, 'font-size': '15pt'}  # Eixos (X e Y)
    tick_font = QFont()
    tick_font.setPixelSize(16)  # Números dos eixos

    for plot in [self.plot_widget]:
      plot.setBackground(bg_color)
      axis_pen = pg.mkPen(color=fg_color)

      for axis_name in ['left', 'bottom']:
        axis = plot.getAxis(axis_name)
        axis.setPen(axis_pen)
        axis.setTextPen(axis_pen)
        axis.setLabel(**label_style)
        axis.setTickFont(tick_font)

      if plot.plotItem.titleLabel.text:
        plot.plotItem.setTitle(
            plot.plotItem.titleLabel.text, color=fg_color, size='18px'
        )

    if hasattr(self, 'legend') and self.legend:
      brush_color = QColor(legend_bg)
      brush_color.setAlpha(255)
      self.legend.setBrush(QBrush(brush_color))

      for sample, label in self.legend.items:
        label.setText(label.text, color=fg_color, size='14px')

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

    if self.legend:
      brush_color = QColor(legend_bg)
      brush_color.setAlpha(255)
      self.legend.setBrush(QBrush(brush_color))

      for sample, label in self.legend.items:
        label.setText(label.text, color=fg_color)
