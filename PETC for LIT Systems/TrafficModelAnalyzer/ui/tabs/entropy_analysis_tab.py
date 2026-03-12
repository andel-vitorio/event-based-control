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


class EntropyAnalysisTab(QWidget):
  """
  Analyzes behavioral entropy and chaos probability within the system.
  Features a dual-view graph card: a 2D line plot for specific sequence lengths (L)
  and a holistic heatmap mapping entropy across all L and parameter values.
  """

  def __init__(self):
    super().__init__()
    self.selected_files: list[str] = []
    self.current_df: pd.DataFrame | None = None
    self.raw_table_data: list[dict] = []
    self.heatmap_grid_lines: list[pg.InfiniteLine] = []
    self._setup_ui()

  def _setup_ui(self) -> None:
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

    # --- Top Control Bar ---
    control_layout = QHBoxLayout()
    self.graph_file_selector = QComboBox()
    self.graph_file_selector.currentIndexChanged.connect(
        self._on_file_selected)

    self.l_word_selector = QComboBox()
    self.l_word_selector.currentIndexChanged.connect(self._render_graphs)

    control_label = QLabel("Display Charts For:")
    control_label.setObjectName("fieldTitle")

    l_label = QLabel("Length (L) [Line Chart]:")
    l_label.setObjectName("fieldTitle")

    control_layout.addWidget(control_label)
    control_layout.addWidget(self.graph_file_selector, 3)
    control_layout.addSpacing(20)
    control_layout.addWidget(l_label)
    control_layout.addWidget(self.l_word_selector, 1)

    self.content_layout.addLayout(control_layout)

    # --- Dual Graph Card ---
    self.graph_card = QWidget()
    self.graph_card.setObjectName("graphCard")
    self.graph_card.setAttribute(
        Qt.WidgetAttribute.WA_StyledBackground, True)

    shadow = QGraphicsDropShadowEffect()
    shadow.setOffset(2, 2)
    shadow.setBlurRadius(15)
    shadow.setColor(QColor(0, 0, 0, 80))
    self.graph_card.setGraphicsEffect(shadow)

    graph_layout = QHBoxLayout(self.graph_card)
    pg.setConfigOption('antialias', True)

    # 1. Line Plot
    self.plot_line = pg.PlotWidget(
        title="Behavioral Entropy vs. Parameter")
    self.plot_line.setLabel('left', 'Entropy h(S) [bits]')
    self.plot_line.showGrid(x=False, y=True, alpha=0.3)
    self.plot_line.setBackground('#171717')
    graph_layout.addWidget(self.plot_line, 1)

    # 2. Heatmap Plot
    self.plot_heatmap = pg.PlotWidget(
        title="Behavioral Entropy Heatmap: L vs. Parameter")
    self.plot_heatmap.setLabel('left', 'Parameter')
    self.plot_heatmap.setLabel('bottom', 'L (Word Length)')
    self.plot_heatmap.setBackground('#171717')
    self.image_item = pg.ImageItem()
    self.plot_heatmap.addItem(self.image_item)

    # Colormap Setup (Viridis approximation)
    pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    color = np.array([
        [68, 1, 84, 255],
        [59, 82, 139, 255],
        [33, 145, 140, 255],
        [94, 201, 98, 255],
        [253, 231, 37, 255]
    ], dtype=np.ubyte)
    cmap = pg.ColorMap(pos, color)
    self.image_item.setColorMap(cmap)

    # Color Scale Bar Integration
    self.color_bar = pg.ColorBarItem(
        colorMap=cmap, label='Entropy h(S) [bits]')
    self.color_bar.setImageItem(self.image_item)
    # Position 2,5 targets the right margin boundary in PlotItem layout
    self.plot_heatmap.getPlotItem().layout.addItem(self.color_bar, 2, 5)

    graph_layout.addWidget(self.plot_heatmap, 1)
    self.content_layout.addWidget(self.graph_card, 3)

    # --- Metrics Table ---
    table_header_layout = QHBoxLayout()
    table_label = QLabel("Comparative Entropy Metrics")
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
    self.metrics_table.setColumnCount(7)
    self.metrics_table.setHorizontalHeaderLabels([
        "K-Max", "Design Approach", "L",
        "Min Entropy", "Max Entropy", "Avg Entropy", "Chaos Rate (%)"
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
      self.plot_line.clear()
      self.image_item.clear()

    self._populate_comparative_table()

  def _on_file_selected(self) -> None:
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
    self._render_graphs()

  def _render_graphs(self) -> None:
    self._render_line_chart()
    self._render_heatmap()

  def _render_line_chart(self) -> None:
    self.plot_line.clear()

    if self.current_df is None or self.current_df.empty or 'Entropy' not in self.current_df.columns:
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
    df = df.dropna(subset=[x_col, 'Entropy'])
    df = df[df[x_col] > 1e-6]

    df_agg = df.groupby(x_col, as_index=False)[
        'Entropy'].mean().sort_values(by=x_col)
    x_vals = df_agg[x_col].values
    entropy_vals = df_agg['Entropy'].values

    if len(x_vals) == 0:
      return

    if is_log_scale:
      x_vals = np.log10(x_vals)

    self.plot_line.setLogMode(x=False, y=False)
    self.plot_line.setLabel('bottom', x_label)

    plot_item = pg.PlotDataItem(
        x_vals, entropy_vals,
        pen=pg.mkPen('#C0392B', width=2.5, style=Qt.PenStyle.DashLine),
        symbol='s', symbolSize=8,
        symbolBrush='#C0392B', symbolPen=None
    )
    self.plot_line.addItem(plot_item)

    current_file = self.graph_file_selector.currentText()
    current_l = self.l_word_selector.currentText()
    title_suffix = f" (for L={current_l})" if current_l != "N/A" else ""
    self.plot_line.plotItem.setTitle(
        f"Behavioral Entropy - {current_file}{title_suffix}")

  def _render_heatmap(self) -> None:
    # Clear legacy grid artifacts
    for line in self.heatmap_grid_lines:
      self.plot_heatmap.removeItem(line)
    self.heatmap_grid_lines.clear()

    self.image_item.clear()
    self.plot_heatmap.getAxis('bottom').setTicks(None)
    self.plot_heatmap.getAxis('left').setTicks(None)

    if self.current_df is None or self.current_df.empty or 'Entropy' not in self.current_df.columns:
      return

    df = self.current_df.copy()

    if 'l_word' not in df.columns:
      return

    is_log_scale = False
    if 'eps_tr' in df.columns:
      x_col = 'eps_tr'
      y_label = 'log10(eps_tr) (Parameter)'
      is_log_scale = True
    elif 'sigma' in df.columns:
      x_col = 'sigma'
      y_label = 'σ (Parameter)'
    else:
      return

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df = df.dropna(subset=[x_col, 'l_word', 'Entropy'])
    df = df[df[x_col] > 1e-6]

    if is_log_scale:
      df[x_col] = np.log10(df[x_col])

    pivot_df = df.pivot_table(
        index=x_col, columns='l_word', values='Entropy', aggfunc='mean')
    pivot_df = pivot_df.sort_index()

    if pivot_df.empty:
      return

    # Z-data transpose aligning matrix shape with GraphicsScene coordinates
    z_data = pivot_df.values.T
    self.image_item.setImage(z_data, autoLevels=True)

    num_cols, num_rows = z_data.shape

    # Render explicit grid delimiters over pixel boundaries
    grid_pen = pg.mkPen(color=(120, 120, 120, 100), width=1)
    for x in range(num_cols + 1):
      v_line = pg.InfiniteLine(pos=x, angle=90, pen=grid_pen)
      self.plot_heatmap.addItem(v_line)
      self.heatmap_grid_lines.append(v_line)

    for y in range(num_rows + 1):
      h_line = pg.InfiniteLine(pos=y, angle=0, pen=grid_pen)
      self.plot_heatmap.addItem(h_line)
      self.heatmap_grid_lines.append(h_line)

    l_words = pivot_df.columns.values
    params = pivot_df.index.values

    # Center axis labels to geometric middle of respective cells
    x_ticks = [(i + 0.5, str(int(val))) for i, val in enumerate(l_words)]

    y_step = max(1, len(params) // 15)
    y_ticks = [(i + 0.5, f"{val:.4g}")
               for i, val in enumerate(params) if i % y_step == 0]

    self.plot_heatmap.getAxis('bottom').setTicks([x_ticks])
    self.plot_heatmap.getAxis('left').setTicks([y_ticks])
    self.plot_heatmap.setLabel('left', y_label)

  def _populate_comparative_table(self) -> None:
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
          if 'Entropy' not in group.columns:
            continue

          min_ent = group['Entropy'].min()
          max_ent = group['Entropy'].max()
          avg_ent = group['Entropy'].mean()

          chaos_rate = 0.0
          if 'IsChaotic' in group.columns:
            chaos_rate = (
                group['IsChaotic'].sum() / len(group)) * 100

          l_display = str(int(l_val)) if l_val is not None else "N/A"

          self.raw_table_data.append({
              'kmax': kmax_val,
              'design': design_val,
              'l': l_display,
              'min_ent': min_ent,
              'max_ent': max_ent,
              'avg_ent': avg_ent,
              'chaos_rate': chaos_rate
          })

      except Exception:
        continue

    self._update_table_filter_options()
    self._apply_table_filters()

  def _update_table_filter_options(self) -> None:
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
          QTableWidgetItem(f"{data['min_ent']:.4f}"),
          QTableWidgetItem(f"{data['max_ent']:.4f}"),
          QTableWidgetItem(f"{data['avg_ent']:.4f}"),
          QTableWidgetItem(f"{data['chaos_rate']:.2f}%")
      ]

      for col_idx, item in enumerate(items):
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metrics_table.setItem(row_idx, col_idx, item)

      row_idx += 1

  def update_theme(self, theme_name: str, fg_color: str) -> None:
    if hasattr(self, 'sidebar'):
      self.sidebar.sidebar_title_label.setStyleSheet(
          f"color: {fg_color};")

    bg_color = '#F8F9FA' if theme_name == 'light' else '#171717'

    for plot in [self.plot_line, self.plot_heatmap]:
      plot.setBackground(bg_color)
      axis_pen = pg.mkPen(color=fg_color)
      for axis_name in ['left', 'bottom']:
        axis = plot.getAxis(axis_name)
        axis.setPen(axis_pen)
        axis.setTextPen(axis_pen)

      if plot.plotItem.titleLabel.text:
        plot.plotItem.setTitle(
            plot.plotItem.titleLabel.text, color=fg_color
        )

    if hasattr(self, 'color_bar'):
      try:
        cb_axis = self.color_bar.getAxis('right')
        cb_axis.setPen(pg.mkPen(color=fg_color))
        cb_axis.setTextPen(pg.mkPen(color=fg_color))
      except Exception:
        pass
