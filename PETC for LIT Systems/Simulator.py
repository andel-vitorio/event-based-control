# -*- coding: utf-8 -*-
"""
Simulator.py
Functional implementation for simulation using C++ backend.
Provides strict I/O handling and native OpenMP parallel processing support.
Ensures determinism and bypasses Windows CLI limits using binary state transfer.
"""

from typing import List, Tuple, Set, Optional
from typing import Tuple, List, Optional, Any
from typing import Any, Dict, Optional, List, Tuple
import scipy.sparse as sp
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as sla
from fractions import Fraction as PyFraction
from z3 import Solver, Real, sat, Q
from scipy.signal import cont2discrete
import networkx as nx
from pyvis.network import Network
import numpy as np
import json
import os
import csv
import tempfile
from Utils.CppInterface import CppSimulator
import Utils.Numeric as nm

# Single internal instance to manage the executable path
_backend = CppSimulator(exe_name="ETCforLinearSystemMain")


class NumpyEncoder(json.JSONEncoder):
  """
  A custom JSON encoder for serializing NumPy data types.

  This encoder ensures that NumPy arrays and scalars are converted to standard
  Python lists and native types (float, int) for JSON compatibility.
  """

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.longdouble)):
      return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16)):
      return int(obj)
    return super(NumpyEncoder, self).default(obj)


def projective_eps_net(eps):
  """
  Generates a uniform sampling of the projective space S^1 (unit circle in 2D).

  This function creates an epsilon-net on the unit circle, useful for covering
  the state space in homogeneous systems.

  Args:
      eps (float): Precision parameter for the epsilon-net.

  Returns:
      tuple: (X, theta) where X contains Cartesian coordinates and theta angular values.
  """
  N = int(np.ceil(np.pi / eps))
  theta = np.linspace(0, np.pi, N, endpoint=False)
  X = np.vstack([np.cos(theta), np.sin(theta)]).T
  return X, theta


def _prepare_temp_config(config_path, results_dict):
  """
  Merges the base configuration with optimization results and writes to a temporary JSON file.

  This function handles the mapping of Greek symbols to ASCII keys expected by the C++ backend
  and ensures atomic file operations to avoid race conditions.

  Args:
      config_path (str): Path to the original experiment JSON.
      results_dict (dict): Dictionary containing controller K and ETM matrices.

  Returns:
      str: Path to the temporary configuration file.
  """
  with open(config_path, 'r') as f:
    full_data = json.load(f)

  full_data['results'] = {
      "controller": {
          "K": np.array(results_dict['controller']['K']).tolist()
      },
      "etm": {
          "Xi": np.array(results_dict['etm']['Ξ']).tolist(),
          "Psi": np.array(results_dict['etm']['Ψ']).tolist()
      }
  }

  tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
  try:
    json.dump(full_data, tmp, cls=NumpyEncoder)
    tmp.flush()
    os.fsync(tmp.fileno())
    return tmp.name
  finally:
    tmp.close()


def open_loop(x0, config_path, u_constant=0.0):
  """
  Executes an open-loop simulation using the high-performance C++ backend.

  This function simulates the system dynamics without feedback control.

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the experiment configuration.
      u_constant (float): Constant input for open-loop simulation.

  Returns:
      tuple: (y, t, u) arrays containing output, time, and input history.
  """
  x0_clean = np.array(x0, copy=True).flatten()

  results = _backend.run(
      json_config_path=config_path,
      x0=x0_clean,
      u_constant=u_constant,
      closed_loop=False
  )
  if not results:
    return None, None, None
  return results['y'], results['t'], results['u']


def closed_loop_setm(x0, config_path, results_dict):
  """
  Executes a closed-loop simulation with a Static Event-Triggered Mechanism (SETM).

  This function utilizes the C++ backend to simulate the system under event-triggered
  control using Runge-Kutta integration.

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the base JSON.
      results_dict (dict): Optimization results containing K, Xi, and Psi.

  Returns:
      tuple: (y, t, u, event_times) arrays containing simulation data.
  """
  if results_dict is None:
    raise ValueError("Optimization results are required for closed-loop.")

  temp_path = _prepare_temp_config(config_path, results_dict)
  x0_clean = np.array(x0, copy=True).flatten()

  try:
    sim_data = _backend.run(
        json_config_path=temp_path,
        x0=x0_clean,
        u_constant=0.0,
        closed_loop=True
    )
    if not sim_data:
      return None, None, None, None

    return sim_data['y'], sim_data['t'], sim_data['u'], sim_data['event_times']
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def recurrence_model_setm(x0, config_path, results_dict):
  """
  Simulates a single trajectory using the discrete-time recurrence map.

  This function is optimized for high-speed detection of event times without
  full state trajectory logging.

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the base JSON.
      results_dict (dict): Optimization results containing K, Xi, and Psi.

  Returns:
      np.ndarray: Array of event timestamps.
  """
  if results_dict is None:
    raise ValueError(
        "Optimization results are required for recurrence map.")

  temp_path = _prepare_temp_config(config_path, results_dict)
  x0_clean = np.array(x0, copy=True).flatten()

  try:
    sim_data = _backend.run(
        json_config_path=temp_path,
        x0=x0_clean,
        u_constant=0.0,
        closed_loop=False,
        recurrence=True
    )

    if not sim_data:
      return None

    return sim_data['event_times']
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def build_symbolic_sequences_parallel(initial_states, config_path, results_dict):
  """
  Performs parallel recurrence mapping using native C++/OpenMP threads.

  This function efficiently computes symbolic sequences for a batch of initial states,
  leveraging the C++ backend for performance and bypassing CLI limitations via binary
  data transfer.

  Args:
      initial_states (np.ndarray): Array of sampled initial states [N x dim].
      config_path (str): Path to the base JSON configuration.
      results_dict (dict): Optimization results containing K, Xi, and Psi.

  Returns:
      list: A list of numpy arrays, each representing an event sequence (k_seq).
  """
  if results_dict is None:
    raise ValueError(
        "Optimization results are required for parallel mapping.")

  temp_path = _prepare_temp_config(config_path, results_dict)
  h = 0.0

  with open(config_path, 'r') as f:
    h = json.load(f)["design_params"]["h"]

  try:
    all_event_times = _backend.run_parallel_recurrence(
        temp_path, initial_states)

    all_k_seqs = []
    for event_times in all_event_times:
      if len(event_times) > 1:
        inter_event_times = np.diff(event_times)
        k_seq = np.round(inter_event_times / h).astype(int).tolist()
        all_k_seqs.append(k_seq)
      else:
        all_k_seqs.append([])

    return all_k_seqs
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def visualize_graph(G, sequence=None, filename="symbolic_model_final.html"):
  """
  Generates an interactive HTML visualization of the symbolic graph.

  The visualization includes navigation controls, node highlighting for specific
  sequences, and topological analysis features.

  Args:
      G (networkx.DiGraph): The symbolic graph.
      sequence (list, optional): A sequence of nodes to highlight as a trajectory.
      filename (str): Output HTML filename.
  """
  net = Network(notebook=True, directed=True, width="100%",
                height="100vh", bgcolor="#fdfdfd", cdn_resources="in_line")

  sccs = list(nx.strongly_connected_components(G))
  attractor_nodes = set()
  for scc in sccs:
    if len(scc) > 1 or (len(scc) == 1 and G.has_edge(list(scc)[0], list(scc)[0])):
      attractor_nodes.update(scc)

  node_id_map = {node: str(i) for i, node in enumerate(G.nodes)}

  sequence_ids = []
  if sequence:
    sequence_ids = [node_id_map[node]
                    for node in sequence if node in node_id_map]

  for node, node_id in node_id_map.items():
    is_seq = sequence and node in sequence
    is_attr = node in attractor_nodes

    if is_seq:
      color = "#007AFF"      # iOS Blue
      size = 40
      font = {'size': 20, 'color': "#000000",
              'face': 'Segoe UI', 'vadjust': -35}
      border_width = 3
    elif is_attr:
      color = "#FF9500"      # Orange
      size = 30
      font = {'size': 16, 'color': "#555555",
              'face': 'Segoe UI', 'vadjust': -30}
      border_width = 2
    else:
      color = "#E5E5EA"      # Light Gray
      size = 15
      font = {'size': 10, 'color': "#AAAAAA", 'face': 'arial'}
      border_width = 1

    net.add_node(
        node_id,
        label=str(node),
        title=f"State: {node}",
        shape="dot",
        size=size,
        color={'background': color, 'border': 'white',
               'highlight': {'border': '#333', 'background': color}},
        font=font,
        borderWidth=border_width
    )

  for u, v, data in G.edges(data=True):
    u_id, v_id = node_id_map[u], node_id_map[v]
    weight = data.get("weight")
    label_text = str(weight) if weight is not None else ""

    is_in_sequence = False
    if sequence:
      for i in range(len(sequence) - 1):
        if u == sequence[i] and v == sequence[i + 1]:
          is_in_sequence = True
          break

    if is_in_sequence:
      color = "#FF2D55"
      width = 4
      font = {
          'size': 22, 'color': '#000000', 'background': 'white',
          'strokeWidth': 0, 'align': 'middle', 'multi': 'html'
      }
      dashes = False
    else:
      color = "#C7C7CC"
      width = 1
      font = {
          'size': 12, 'color': '#8E8E93', 'background': 'white',
          'strokeWidth': 0, 'align': 'middle'
      }
      dashes = True

    net.add_edge(u_id, v_id, label=label_text, color=color,
                 width=width, font=font, dashes=dashes, arrows="to")

  options = {
      "nodes": {"font": {"strokeWidth": 2, "strokeColor": "#ffffff"}},
      "edges": {
          "smooth": {"type": "curvedCW", "roundness": 0.2},
          "font": {"align": "middle"}
      },
      "physics": {
          "forceAtlas2Based": {
              "gravitationalConstant": -120,
              "centralGravity": 0.008,
              "springLength": 230,
              "damping": 0.9
          },
          "solver": "forceAtlas2Based",
          "stabilization": {"iterations": 200, "updateInterval": 50}
      },
      "interaction": {
          "hover": True,
          "navigationButtons": True,
          "keyboard": True,
          "zoomView": True
      }
  }
  net.set_options(f"var options = {json.dumps(options)}")
  net.save_graph(filename)

  with open(filename, "r", encoding="utf-8") as f:
    html_content = f.read()

  custom_ui = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

        #thesis-ui {{
            position: absolute; top: 20px; right: 20px; width: 260px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.5);
            font-family: 'Inter', sans-serif;
            z-index: 1000;
            transition: all 0.3s ease;
        }}

        #thesis-ui:hover {{
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }}

        h3 {{ margin: 0 0 15px 0; font-size: 16px; color: #1c1c1e; font-weight: 600; letter-spacing: -0.5px; }}

        .legend-row {{ display: flex; align-items: center; margin-bottom: 8px; font-size: 13px; color: #3a3a3c; }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 12px; }}
        .line {{ width: 20px; height: 3px; margin-right: 12px; border-radius: 2px; }}

        .divider {{ height: 1px; background: rgba(0,0,0,0.08); margin: 15px 0; }}

        .ctrl-panel {{ display: flex; justify-content: space-between; align-items: center; }}

        button {{
            background: #007AFF; color: white; border: none; padding: 8px 16px;
            border-radius: 8px; font-weight: 600; font-size: 13px; cursor: pointer;
            transition: background 0.2s;
            flex: 1; margin: 0 4px;
        }}
        button:hover {{ background: #0056D2; }}
        button:active {{ transform: scale(0.96); }}

        #step-counter {{
            text-align: center; margin-top: 10px; font-size: 12px;
            color: #8E8E93; font-variant-numeric: tabular-nums;
        }}
    </style>

    <div id="thesis-ui">
        <h3>Symbolic Control Viewer</h3>

        <div class="legend-row"><span class="dot" style="background:#007AFF; box-shadow: 0 0 8px rgba(0,122,255,0.4);"></span> Trajectory</div>
        <div class="legend-row"><span class="dot" style="background:#FF9500;"></span> Attractor</div>
        <div class="legend-row"><span class="line" style="background:#FF2D55;"></span> Transition (k)</div>
        <div class="legend-row"><span class="line" style="background:#C7C7CC; border:1px dashed #aaa;"></span> Others</div>

        <div class="divider"></div>

        <div class="ctrl-panel">
            <button onclick="prevStep()">← Prev</button>
            <button onclick="nextStep()">Next →</button>
        </div>
        <div id="step-counter">Initial State</div>
    </div>

    <script>
        var sequenceIds = {json.dumps(sequence_ids)};
        var currentStep = 0;

        // Animation Configuration
        var focusOptions = {{
            scale: 1.5,
            offset: {{x:0, y:0}},
            animation: {{
                duration: 1000,
                easingFunction: "easeInOutQuart"
            }}
        }};

        network.once("afterDrawing", function() {{
            if(sequenceIds.length > 0) focusNode(0);
        }});

        function updateUI() {{
            document.getElementById('step-counter').innerText =
                "Step " + (currentStep + 1) + " of " + sequenceIds.length;
        }}

        function focusNode(index) {{
            if(sequenceIds.length === 0) return;
            var nodeId = sequenceIds[index];

            network.selectNodes([nodeId]);
            network.focus(nodeId, focusOptions);
            updateUI();
        }}

        function nextStep() {{
            if (currentStep < sequenceIds.length - 1) {{
                currentStep++;
                focusNode(currentStep);
            }}
        }}

        function prevStep() {{
            if (currentStep > 0) {{
                currentStep--;
                focusNode(currentStep);
            }}
        }}

        // Keyboard Shortcuts
        document.addEventListener('keydown', function(event) {{
            if (event.key === "ArrowRight") nextStep();
            if (event.key === "ArrowLeft") prevStep();
        }});
    </script>
    </body>
    """

  final_html = html_content.replace('</body>', custom_ui)
  with open(filename, "w", encoding="utf-8") as f:
    f.write(final_html)


class SequenceReconstructor:
  """
  Reconstructs initial states (x0) compatible with a predefined event sequence in ETC/SETM systems.

  This class uses the Z3 theorem prover to solve quadratic triggering constraints and find a valid
  initial state for a given symbolic sequence.
  """

  def __init__(self, A_c, B_c, K, Xi, Psi, h, iet_max):
    """
    Initializes the reconstructor and pre-computes transition and triggering matrices.

    Args:
        A_c, B_c: Continuous system matrices.
        K: Controller gain.
        Xi, Psi: Triggering mechanism weight matrices.
        h: Discretization step.
        iet_max: Maximum inter-event time.
    """
    self.nx = A_c.shape[0]
    self.h = h
    self.m_max = int(round(iet_max / h))

    sys_d = cont2discrete((A_c, B_c, np.zeros((1, self.nx)),
                           np.zeros((1, B_c.shape[1]))), h, method='zoh')
    self.Ad, self.Bd = sys_d[0], sys_d[1]

    self.Phi_cache = [None] * (self.m_max + 1)
    self.M_cache = [None] * (self.m_max + 1)

    I = np.eye(self.nx)
    self.Phi_cache[0] = I

    for m in range(1, self.m_max + 1):
      Ad_pow_m = np.linalg.matrix_power(self.Ad, m)
      S_m = np.zeros_like(self.Ad)
      for p in range(m):
        S_m += np.linalg.matrix_power(self.Ad, p)
      Phi_m = Ad_pow_m + S_m @ self.Bd @ K
      self.Phi_cache[m] = Phi_m
      diff = I - Phi_m
      M_val = (Phi_m.T @ Psi @ Phi_m) - (diff.T @ Xi @ diff)
      self.M_cache[m] = M_val

  def _to_rational(self, matrix):
    return [[Q(PyFraction(val).limit_denominator(10**10).numerator,
               PyFraction(val).limit_denominator(10**10).denominator)
             for val in row] for row in matrix]

  def find_compatible_state(self, sequence):
    """
    Finds a non-zero initial state vector x0 that generates the exact provided event sequence.

    Args:
        sequence (list[int]): List of inter-event intervals (in steps h).

    Returns:
        np.array: Valid state vector x0, or None if the sequence is impossible.
    """
    solver = Solver()
    x0 = [Real(f'x0_{i}') for i in range(self.nx)]
    solver.add(sum([x * x for x in x0]) > Q(1, 1000))
    G_acc = np.eye(self.nx)

    for m_target in sequence:
      if m_target > self.m_max:
        return None

      for k in range(1, m_target):
        M_k = self.M_cache[k]
        M_proj = G_acc.T @ M_k @ G_acc
        M_z3 = self._to_rational(M_proj)

        quad_form = sum(x0[r] * M_z3[r][c] * x0[c]
                        for r in range(self.nx) for c in range(self.nx))
        solver.add(quad_form >= 0)

      if m_target < self.m_max:
        M_target_mat = self.M_cache[m_target]
        M_proj_trig = G_acc.T @ M_target_mat @ G_acc
        M_z3_trig = self._to_rational(M_proj_trig)

        quad_form_trig = sum(x0[r] * M_z3_trig[r][c] * x0[c]
                             for r in range(self.nx) for c in range(self.nx))
        solver.add(quad_form_trig < 0)

      G_acc = self.Phi_cache[m_target] @ G_acc

    if solver.check() == sat:
      model = solver.model()
      return np.array([float(model[v].as_fraction()) for v in x0])
    else:
      return None


def build_symbolic_exactly_graph_old(target_l, K_set, h, A, B, K, Xi, Psi, iet_max, verbose=True):
  """
  Constructs the symbolic graph by progressively increasing the sequence length.

  This function reduces computational complexity by pruning infeasible branches early
  in the construction process.
  """
  if verbose:
    print(f"=== [Init] Pre-calculating dynamics (Z3 Backend) ===")
  reconstructor = SequenceReconstructor(A, B, K, Xi, Psi, h, iet_max)
  valid_sequences = []
  if verbose:
    print(f"--- Layer 1 ---")
  for k in K_set:
    seq = (k,)
    if reconstructor.find_compatible_state(seq) is not None:
      valid_sequences.append(seq)

  for length in range(2, target_l + 1):
    if verbose:
      print(f"--- Layer {length} ---")
    next_valid_sequences = []
    candidates_count = len(valid_sequences) * len(K_set)

    for seq in valid_sequences:
      for k_next in K_set:
        candidate = seq + (k_next,)

        if reconstructor.find_compatible_state(candidate) is not None:
          next_valid_sequences.append(candidate)

    dropped = candidates_count - len(next_valid_sequences)
    if verbose:
      print(
          f"   Candidates: {candidates_count} | Valid: {len(next_valid_sequences)} | Pruned: {dropped}")

    if not next_valid_sequences:
      if verbose:
        print("WARNING: No valid sequences found at this length!")
      break

    valid_sequences = next_valid_sequences

  feasible_nodes = valid_sequences
  feasible_nodes_set = set(feasible_nodes)
  G = nx.DiGraph()
  G.add_nodes_from(feasible_nodes)

  if verbose:
    print(
        f"=== [Edges] Generating edges (Extension to size {target_l + 1}) ===")
  count_edges = 0

  for node_a in feasible_nodes:
    suffix = node_a[1:]

    for k_next in K_set:
      node_b = suffix + (k_next,)

      if node_b in feasible_nodes_set:
        transition_seq = node_a + (k_next,)
        if reconstructor.find_compatible_state(transition_seq) is not None:
          G.add_edge(
              node_a,
              node_b,
              weight=node_a[0],
              label=str(node_a[0])
          )
          count_edges += 1

  if verbose:
    print("\n=== Final Report ===")
    print(f"Final Nodes: {len(feasible_nodes)}")
    print(f"Edges:       {count_edges}")

  return G


def build_symbolic_exactly_graph(
    target_l: int,
    K_set: List[int],
    h: float,
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    Xi: np.ndarray,
    Psi: np.ndarray,
    iet_max: float,
    verbose: bool = True
) -> nx.DiGraph:
  """
  Constructs the l-complete symbolic graph utilizing topological pruning.

  Extracts base transitions at layer 2 to act as an adjacency filter for higher 
  dimensions, strictly bounding the combinatorial explosion of SMT verification queries.
  """
  if verbose:
    print(f"=== [Init] Pre-calculating dynamics (Z3 Backend) ===")

  reconstructor = SequenceReconstructor(A, B, K, Xi, Psi, h, iet_max)
  valid_sequences: List[Tuple[int, ...]] = []

  if verbose:
    print(f"--- Layer 1 ---")

  for k in K_set:
    seq = (k,)
    if reconstructor.find_compatible_state(seq) is not None:
      valid_sequences.append(seq)

  valid_base_transitions: Set[Tuple[int, int]] = set()

  for length in range(2, target_l + 1):
    if verbose:
      print(f"--- Layer {length} ---")

    next_valid_sequences: List[Tuple[int, ...]] = []
    candidates_count = len(valid_sequences) * len(K_set)
    smt_calls = 0

    for seq in valid_sequences:
      for k_next in K_set:
        if length > 2 and (seq[-1], k_next) not in valid_base_transitions:
          continue

        candidate = seq + (k_next,)
        smt_calls += 1

        if reconstructor.find_compatible_state(candidate) is not None:
          next_valid_sequences.append(candidate)
          if length == 2:
            valid_base_transitions.add((seq[-1], k_next))

    dropped = candidates_count - len(next_valid_sequences)
    pruned_by_topology = candidates_count - smt_calls

    if verbose:
      print(
          f"   Candidates: {candidates_count} | Valid: {len(next_valid_sequences)}")
      print(
          f"   Topological Pruning Avoided {pruned_by_topology} SMT calls.")
      print(f"   Total Dropped: {dropped}")

    if not next_valid_sequences:
      if verbose:
        print("WARNING: No valid sequences found at this length!")
      break

    valid_sequences = next_valid_sequences

  feasible_nodes = valid_sequences
  feasible_nodes_set = set(feasible_nodes)
  G = nx.DiGraph()
  G.add_nodes_from(feasible_nodes)

  if verbose:
    print(
        f"=== [Edges] Generating edges (Extension to size {target_l + 1}) ===")

  count_edges = 0

  for node_a in feasible_nodes:
    for k_next in K_set:
      if (node_a[-1], k_next) not in valid_base_transitions:
        continue

      node_b = node_a[1:] + (k_next,)

      if node_b in feasible_nodes_set:
        transition_seq = node_a + (k_next,)
        if reconstructor.find_compatible_state(transition_seq) is not None:
          G.add_edge(
              node_a,
              node_b,
              weight=node_a[0],
              label=str(node_a[0])
          )
          count_edges += 1

  if verbose:
    print("\n=== Final Report ===")
    print(f"Final Nodes: {len(feasible_nodes)}")
    print(f"Edges:       {count_edges}")

  return G


class SequenceRegionAnalyzer:
  """
  Analyzes and determines the analytical angular regions (cones) in the phase plane.

  This class identifies the set of initial angles that generate a specific symbolic
  sequence for linear homogeneous systems.
  """

  def __init__(self, reconstructor):
    """
    Args:
        reconstructor: Instance of SequenceReconstructor containing
                       pre-computed Phi and M matrices.
    """
    self.rec = reconstructor

  def _get_angular_interval(self, M):
    """
    Computes the angular intervals theta in [0, pi) where the quadratic form x'Mx >= 0.
    """
    a = M[1, 1]
    b = M[0, 1] + M[1, 0]
    c = M[0, 0]

    delta = b**2 - 4*a*c

    if delta < 0:
      return [(0, np.pi)] if c >= 0 else []

    t1 = (-b - np.sqrt(delta)) / (2 * a) if a != 0 else -c/b
    t2 = (-b + np.sqrt(delta)) / (2 * a) if a != 0 else -c/b

    angles = sorted([np.arctan(t1) % np.pi, np.arctan(t2) % np.pi])

    mid_test = angles[0] + (angles[1] - angles[0]) / 2
    test_vec = np.array([np.cos(mid_test), np.sin(mid_test)])

    if test_vec.T @ M @ test_vec >= 0:
      return [(angles[0], angles[1])]
    else:
      return [(0, angles[0]), (angles[1], np.pi)]

  def find_region(self, sequence):
    """
    Identifies the valid angular intervals [theta_min, theta_max] for a given symbolic sequence.

    Args:
        sequence (tuple): The target sequence of inter-event steps (k).

    Returns:
        list: List of tuples representing valid angular intervals in radians.
    """
    G_acc = np.eye(2)
    final_intervals = [(0, np.pi)]

    for m_target in sequence:
      for k in range(1, m_target):
        M_eff = G_acc.T @ self.rec.M_cache[k] @ G_acc
        new_interval = self._get_angular_interval(M_eff)
        final_intervals = self._intersect(
            final_intervals, new_interval)

      if m_target < self.rec.m_max:
        M_eff_trig = G_acc.T @ self.rec.M_cache[m_target] @ G_acc
        new_interval = self._get_angular_interval(-M_eff_trig)
        final_intervals = self._intersect(
            final_intervals, new_interval)

      G_acc = self.rec.Phi_cache[m_target] @ G_acc

    return final_intervals

  def _intersect(self, intervals_a, intervals_b):
    res = []
    for start_a, end_a in intervals_a:
      for start_b, end_b in intervals_b:
        s = max(start_a, start_b)
        e = min(end_a, end_b)
        if s < e:
          res.append((s, e))
    return res


def verify_region_robustness(target_seq, region_list, config, prob_results, experiment_file, n_samples=100, eps=1e-6):
  """
  Validates the analytical invariance region using statistical sampling and high-fidelity simulation.

  This function checks if sampled points within the calculated region produce the expected
  symbolic sequence.

  Args:
      target_seq (tuple): The symbolic sequence to verify.
      region_list (list): List of (theta_min, theta_max) tuples.
      config (dict): Configuration dictionary containing design_params.
      prob_results (dict): Dictionary containing controller and ETM matrices.
      experiment_file (str): Path to the simulation configuration file.
      n_samples (int): Number of points to sample within the region.
      eps (float): Safety margin to mitigate boundary numerical sensitivities.
  """
  if not region_list:
    print("Error: The provided region list is empty.")
    return

  sampling_period = config["design_params"]['h']
  theta_min, theta_max = region_list[0]

  theta_min_safe = theta_min + eps
  theta_max_safe = theta_max - eps

  if theta_min_safe >= theta_max_safe:
    print("Warning: Epsilon margin exceeds region width. Using center point only.")
    sampled_angles = np.array([(theta_min + theta_max) / 2])
  else:
    sampled_angles = np.linspace(theta_min_safe, theta_max_safe, n_samples)

  stats = {"success": 0, "failure": 0}
  failures = []

  print(f"--- Initiating Robustness Verification (eps={eps}) ---")
  print(f"Target Sequence: {target_seq}")
  print(
      f"Safe Range: [{np.degrees(theta_min_safe):.4f}°, {np.degrees(theta_max_safe):.4f}°]\n")

  for i, theta in enumerate(sampled_angles):
    x0_sample = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

    try:
      events = recurrence_model_setm(
          x0_sample, experiment_file, prob_results)
      inter_event_times = nm.compute_deltas(events)

      sim_seq = np.round(
          np.array(inter_event_times[1:]) / sampling_period).astype(int)
      obtained_seq = tuple(sim_seq[:len(target_seq)])

      if obtained_seq == target_seq:
        stats["success"] += 1
      else:
        stats["failure"] += 1
        failures.append({
            "sample_idx": i,
            "angle_deg": np.degrees(theta),
            "got": obtained_seq
        })

    except Exception as e:
      stats["failure"] += 1
      print(f"Execution error at sample {i}: {e}")

  total_effective = len(sampled_angles)
  accuracy = (stats["success"] / total_effective) * 100

  print("-" * 45)
  print(f"VERIFICATION COMPLETE")
  print(f"Reliability: {accuracy:.2f}%")
  print(f"OK: {stats['success']} | FAIL: {stats['failure']}")
  print("-" * 45)

  if failures:
    print("\nAnalysis of Divergences:")
    for fail in failures[:3]:
      print(
          f"Sample {fail['sample_idx']}: θ = {fail['angle_deg']:.6f}° -> Got {fail['got']}")


def extract_k_val(node: Any) -> int:
  """
  Extracts the inter-event time scalar 'k' from a node representation.
  The weight corresponds strictly to the first element of the sequence
  per the formal definition of l-complete traffic models.
  """
  val = node[0] if isinstance(node, (tuple, list)) else node
  return int(val)


def karp_minimum_mean_cycle(
    G: nx.DiGraph,
    h_scalar: float,
    maximize: bool = False
) -> Tuple[Optional[float], List[Any]]:
  """
  Implements Karp's algorithm (1978) to find the Minimum or Maximum Mean Weight Cycle.
  Computes the critical mean and mathematically reconstructs the sequence of nodes comprising the cycle.
  """
  n = G.number_of_nodes()
  if n == 0:
    return None, []

  nodes = list(G.nodes())
  node_to_idx = {node: i for i, node in enumerate(nodes)}

  F = np.full((n + 1, n), np.inf if not maximize else -np.inf)
  parent = np.full((n + 1, n), -1, dtype=int)

  F[0, :] = 0.0

  edges = [(node_to_idx[u], node_to_idx[v], extract_k_val(u) * h_scalar)
           for u, v in G.edges()]

  for k in range(1, n + 1):
    for u, v, weight in edges:
      w = -weight if maximize else weight
      if not np.isinf(F[k - 1, u]):
        new_cost = F[k - 1, u] + w
        if (not maximize and new_cost < F[k, v]) or (maximize and new_cost > F[k, v]):
          F[k, v] = new_cost
          parent[k, v] = u

  best_mean = np.inf if not maximize else -np.inf
  best_v = -1

  for v in range(n):
    if np.isinf(F[n, v]):
      continue

    v_max_val = -np.inf if not maximize else np.inf
    for k in range(n):
      if np.isinf(F[k, v]):
        continue

      val = (F[n, v] - F[k, v]) / (n - k)
      if (not maximize and val > v_max_val) or (maximize and val < v_max_val):
        v_max_val = val

    if (not maximize and v_max_val < best_mean) or (maximize and v_max_val > best_mean):
      best_mean = v_max_val
      best_v = v

  if best_v == -1:
    return None, []

  path = []
  curr = best_v
  for k in range(n, -1, -1):
    path.append(curr)
    curr = parent[k, curr]

  visited = {}
  cycle_nodes = []
  for i, node_idx in enumerate(path):
    if node_idx in visited:
      cycle_indices = path[visited[node_idx]:i]
      cycle_nodes = [nodes[idx] for idx in cycle_indices]
      break
    visited[node_idx] = i

  cycle_nodes.reverse()

  return abs(best_mean), cycle_nodes


def compute_analytical_metrics(
    G: nx.DiGraph,
    h: float,
    reconstructor: Optional[Any] = None,
    verbose: bool = True
) -> Dict[str, Any]:
  """
  Generates a comprehensive analytical report of ETC traffic metrics, strictly implementing
  the quantitative automata formulations by Gleizer & Mazo Jr. (2023).
  """
  num_nodes = G.number_of_nodes()
  if num_nodes == 0:
    raise ValueError("Graph G topology is empty.")

  h_scalar = float(h)
  if h_scalar <= 0.0:
    raise ValueError("Temporal scalar 'h' must be strictly positive.")

  num_edges = G.number_of_edges()
  all_ks = [extract_k_val(n) for n in G.nodes()]

  inf_val = min(all_ks) * h_scalar
  sup_val = max(all_ks) * h_scalar

  sccs = list(nx.strongly_connected_components(G))
  recurrent_ks = []
  max_scc_size = 0
  recurrent_nodes = set()

  for scc in sccs:
    if len(scc) > 1 or (len(scc) == 1 and G.has_edge(list(scc)[0], list(scc)[0])):
      max_scc_size = max(max_scc_size, len(scc))
      recurrent_nodes.update(scc)
      recurrent_ks.extend([extract_k_val(n) for n in scc])

  inf_liminf = min(recurrent_ks) * h_scalar if recurrent_ks else None
  sup_limsup = max(recurrent_ks) * h_scalar if recurrent_ks else None

  entropy = 0.0
  if recurrent_nodes:
    Gr = G.subgraph(recurrent_nodes)

    if Gr.number_of_nodes() < 500:
      adj_dense = nx.adjacency_matrix(Gr).toarray()
      vals = np.linalg.eigvals(adj_dense)
      rho = float(np.max(np.abs(vals)))
    else:
      adj_sparse = nx.adjacency_matrix(Gr).astype(float)
      try:
        vals = sla.eigs(adj_sparse, k=1, which='LM',
                        return_eigenvectors=False)
        rho = float(np.abs(vals[0]))
      except sla.ArpackNoConvergence:
        adj_dense = adj_sparse.toarray()
        vals = np.linalg.eigvals(adj_dense)
        rho = float(np.max(np.abs(vals)))

    entropy = float(np.log2(rho)) if rho > (1.0 + 1e-9) else 0.0

  is_chaotic = entropy > 1e-6

  inf_lim_avg = None
  sup_lim_avg = None
  rob_inf_lim_avg = None
  is_mac_stable = False
  savings_pct = None

  if hasattr(reconstructor, 'M_cache') and reconstructor.M_cache:
    def get_m_k(k: int) -> Optional[np.ndarray]:
      cache = reconstructor.M_cache
      if isinstance(cache, dict):
        return cache.get(k)
      if isinstance(cache, (list, tuple, np.ndarray)) and 0 <= k < len(cache):
        return cache[k]
      return None

    # Extract limits using Karp's algorithm in polynomial time
    min_mean, mac_cycle = karp_minimum_mean_cycle(
        G, h_scalar, maximize=False)
    max_mean, _ = karp_minimum_mean_cycle(G, h_scalar, maximize=True)

    if min_mean is not None:
      inf_lim_avg = min_mean
      sup_lim_avg = max_mean

      if inf_lim_avg > 1e-9:
        savings_pct = (1.0 - (h_scalar / inf_lim_avg)) * 100.0

      # Verify Schur stability specifically for the MAC
      if mac_cycle:
        sample_matrix = get_m_k(all_ks[0])
        dim = sample_matrix.shape[0] if sample_matrix is not None else 2
        M_sigma = np.eye(dim)
        valid = True

        for n in mac_cycle:
          M_k = get_m_k(extract_k_val(n))
          if M_k is None:
            valid = False
            break
          M_sigma = M_k @ M_sigma

        if valid:
          spec_rad = float(
              np.max(np.abs(np.linalg.eigvals(M_sigma))))
          is_mac_stable = spec_rad < (1.0 - 1e-9)
          if is_mac_stable:
            rob_inf_lim_avg = inf_lim_avg

  if verbose:
    print(f"\n{'='*60}")
    print(f"ANALYTICAL TRAFFIC REPORT (Karp 1978 DP Implementation)")
    print(f"{'='*60}")
    print(
        f"Nodes: {num_nodes} | Edges: {num_edges} | Max SCC: {max_scc_size}")
    print(f"Inf: {inf_val} | Sup: {sup_val}")
    print(f"Entropy H(T_l): {entropy:.4f} bits | Chaotic: {is_chaotic}")
    print(f"InfLimInf: {inf_liminf} | SupLimSup: {sup_limsup}")
    print(f"InfLimAvg (MAC): {inf_lim_avg} | SupLimAvg: {sup_lim_avg}")
    print(f"MAC Schur Stable: {is_mac_stable}")
    print(
        f"Robust InfLimAvg: {rob_inf_lim_avg if rob_inf_lim_avg else 'Undefined (MAC is Unstable/Chaotic)'}")
    print(f"{'='*60}\n")

  return {
      "NumNodes": num_nodes,
      "NumEdges": num_edges,
      "Entropy": entropy,
      "Inf": inf_val,
      "Sup": sup_val,
      "InfLimInf": inf_liminf,
      "SupLimSup": sup_limsup,
      "InfLimAvg": inf_lim_avg,
      "SupLimAvg": sup_lim_avg,
      "RobInfLimAvg": rob_inf_lim_avg,
      "IsMACStable": is_mac_stable,
      "SavingsPct": savings_pct
  }


def export_metrics_to_csv(metrics, filename="metrics.csv"):
  """
  Exports a metrics dictionary to a CSV file, appending if the file exists.

  This function handles numpy types and ensures that if the file already exists,
  the new data aligns with the existing columns.

  Args:
      metrics (dict): The dictionary containing metric names and values.
      filename (str): The target CSV file path.
  """

  # 1. Clean numpy types for serialization
  clean_metrics = {}
  for k, v in metrics.items():
    if hasattr(v, 'item'):
      clean_metrics[k] = v.item()
    else:
      clean_metrics[k] = v

  file_exists = os.path.isfile(filename)
  fieldnames = list(clean_metrics.keys())

  # 2. If file exists, read header to ensure column alignment
  if file_exists:
    try:
      with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
        if existing_header:
          fieldnames = existing_header
    except Exception as e:
      print(f"Warning: Could not read existing CSV header: {e}")

  # 3. Write data
  with open(filename, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

    if not file_exists or f.tell() == 0:
      writer.writeheader()

    writer.writerow(clean_metrics)
